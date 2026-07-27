import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import json
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.pointnet2_sem_seg import get_model, get_loss
from data_utils.heath_dataset import HeideDataset, NUM_CLASSES, CLASS_NAMES

BATCH_SIZE    = 8
EPOCHS        = 64
LEARNING_RATE = 1e-4
TILE_DIR      = '/Users/luis/Documents/BA/training_tiles'
LOG_DIR       = './logs/heath_run2'
PRETRAINED    = './logs/heath_run1/best_model.pth'


def calculate_iou(pred, target, num_classes):
    iou_list = []
    for c in range(num_classes):
        tp = np.sum((pred == c) & (target == c))
        fp = np.sum((pred == c) & (target != c))
        fn = np.sum((pred != c) & (target == c))
        denom = tp + fp + fn
        iou_list.append(np.nan if denom == 0 else tp / denom)
    return np.array(iou_list)


def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Nutze Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Nutze NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")
        print("Nutze Standard-Prozessor (CPU)")

    with open(os.path.join(TILE_DIR, 'splits.json')) as f:
        splits = json.load(f)

    train_ds = HeideDataset(TILE_DIR, splits['train'], augment=True)
    val_ds   = HeideDataset(TILE_DIR, splits['val'],   augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False)

    raw_counts = np.array([91358, 27120, 404802, 556455,
                           329, 45694, 3380, 1574904])
    weights = 1.0 / np.sqrt(raw_counts + 1e-6)
    weights = weights / weights.sum() * NUM_CLASSES
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

    print("\nKlassengewichte:")
    for name, w in zip(CLASS_NAMES, weights):
        print(f"  {name:<12} {w:.4f}")

    model = get_model(NUM_CLASSES).to(device)

    print(f"\nLade vortrainiertes Modell: {PRETRAINED}")
    model.load_state_dict(torch.load(PRETRAINED, map_location=device))

    criterion = get_loss().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                 weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6)

    os.makedirs(LOG_DIR, exist_ok=True)
    checkpoint_path = os.path.join(LOG_DIR, 'checkpoint_latest.pth')
    best_model_path = os.path.join(LOG_DIR, 'best_model.pth')
    writer = SummaryWriter(LOG_DIR)
    best_miou = 0.0

    print(f"\nStarte Feintuning fuer {EPOCHS} Epochen (lr={LEARNING_RATE})...")
    for epoch in range(1, EPOCHS + 1):

        current_lr = optimizer.param_groups[0]['lr']

        model.train()
        train_loss = 0.0

        for points, labels in tqdm(train_loader,
                                   desc=f"Epoche {epoch}/{EPOCHS} [Train]"):
            points = points.float().to(device)
            labels = labels.long().to(device)
            inp    = points.permute(0, 2, 1).contiguous()

            optimizer.zero_grad()
            seg_pred, trans_feat = model(inp)

            loss = criterion(
                seg_pred.contiguous().view(-1, NUM_CLASSES),
                labels.view(-1),
                trans_feat,
                weights_tensor
            )
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        avg_loss = train_loss / len(train_loader)
        writer.add_scalar('Loss/Train', avg_loss, epoch)
        writer.add_scalar('LR', current_lr, epoch)

        model.eval()
        all_ious = []

        with torch.no_grad():
            for points, labels in tqdm(val_loader,
                                       desc=f"Epoche {epoch}/{EPOCHS} [Val]"):
                points    = points.float().to(device)
                labels_np = labels.numpy()
                inp       = points.permute(0, 2, 1).contiguous()

                seg_pred, _ = model(inp)
                pred_np     = seg_pred.argmax(dim=2).cpu().numpy()

                for b in range(pred_np.shape[0]):
                    all_ious.append(
                        calculate_iou(pred_np[b], labels_np[b], NUM_CLASSES))

        iou_array = np.nanmean(np.stack(all_ious), axis=0)
        miou      = np.nanmean(iou_array)

        writer.add_scalar('Metric/mIoU', miou, epoch)
        for i, name in enumerate(CLASS_NAMES):
            writer.add_scalar(f'IoU/{name}', iou_array[i], epoch)

        print(f"\nErgebnis Epoche {epoch}: "
              f"Loss = {avg_loss:.4f} | mIoU = {miou:.4f} | lr = {current_lr:.2e}")
        for name, val in zip(CLASS_NAMES, iou_array):
            print(f"  {name:<12} IoU = {val:.4f}")

        torch.save({
            'epoch':           epoch,
            'model_state':     model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'best_miou':       best_miou,
        }, checkpoint_path)

        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), best_model_path)
            print("  -> Neues bestes Modell gespeichert.")

    writer.close()
    print(f"\nFeintuning abgeschlossen. Bestes mIoU: {best_miou:.4f}")


if __name__ == '__main__':
    main()
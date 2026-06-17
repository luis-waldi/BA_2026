"""
train_heath.py
Trainiert PointNet++ auf dem Heide-Datensatz zur semantischen Segmentierung.
"""
import os
# MPS-Fallback aktivieren: erlaubt CPU-Ausweichen fuer Ops, die Apple Silicon
# (noch) nicht unterstuetzt. MUSS vor dem torch-Import gesetzt werden.
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import json
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Importiere PointNet++ Architektur und den DataLoader
from models.pointnet2_sem_seg import get_model, get_loss
from data_utils.heath_dataset import HeideDataset, NUM_CLASSES, CLASS_NAMES

# Hyperparameter (Stellschrauben fuer das Training)
BATCH_SIZE = 8       # Wie viele Kacheln gleichzeitig gelernt werden
EPOCHS = 32          # Wie oft das Netzwerk den gesamten Datensatz sieht
LEARNING_RATE = 1e-3 # Wie schnell das Netzwerk lernt
TILE_DIR = '/Users/luis/Documents/BA/training_tiles'
LOG_DIR = './logs/heath_run1'


def calculate_iou(pred, target, num_classes):
    """Berechnet die Intersection over Union (IoU) fuer jede Klasse."""
    iou_list = []
    for c in range(num_classes):
        tp = np.sum((pred == c) & (target == c))
        fp = np.sum((pred == c) & (target != c))
        fn = np.sum((pred != c) & (target == c))

        denom = tp + fp + fn
        if denom == 0:
            iou_list.append(np.nan)  # Klasse in dieser Kachel nicht vorhanden
        else:
            iou_list.append(tp / denom)
    return np.array(iou_list)


def main():
    # 1. Hardware-Check (Apple Silicon, CUDA oder CPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Nutze Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Nutze NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")
        print("Nutze Standard-Prozessor (CPU)")

    # 2. Daten laden
    with open(os.path.join(TILE_DIR, 'splits.json')) as f:
        splits = json.load(f)

    train_ds = HeideDataset(TILE_DIR, splits['train'], augment=True)
    val_ds = HeideDataset(TILE_DIR, splits['val'], augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Klassengewichte (aus den Daten berechnet, damit seltene Klassen
    # wie "soil" oder "other" nicht untergehen)
    raw_counts = np.array([91358, 27120, 404802, 556455,
                           329, 45694, 3380, 1574904])
    weights = 1.0 / (raw_counts + 1e-6)
    weights = weights / weights.sum() * NUM_CLASSES
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

    # 3. Modell, Loss und Optimierer initialisieren
    model = get_model(NUM_CLASSES).to(device)
    criterion = get_loss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Logger fuer Live-Auswertung (Verzeichnis VOR dem Writer anlegen)
    os.makedirs(LOG_DIR, exist_ok=True)
    writer = SummaryWriter(LOG_DIR)
    best_miou = 0.0

    # 4. Trainings-Loop
    print(f"\nStarte Training fuer {EPOCHS} Epochen...")
    for epoch in range(1, EPOCHS + 1):

        # --- TRAINING ---
        model.train()
        train_loss = 0.0

        for points, labels in tqdm(train_loader,
                                   desc=f"Epoche {epoch}/{EPOCHS} [Train]"):
            points = points.float().to(device)
            labels = labels.long().to(device)

            # PointNet++ erwartet EINEN Tensor im Format (Batch, Channels, Points).
            # Spalten: x_c, y_c, Z, R, G, B  ->  6 Kanaele.
            # Das Modell trennt XYZ (erste 3) und Features intern selbst.
            inp = points.permute(0, 2, 1).contiguous()  # (B, 6, N)

            optimizer.zero_grad()

            # Modell-Output: seg_pred (B, N, num_classes) als log_softmax,
            #                trans_feat fuer die Loss-Funktion
            seg_pred, trans_feat = model(inp)

            # Fuer den Loss in (B*N, C) umformen - KEIN permute noetig,
            # da seg_pred bereits (B, N, C) ist.
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)
            loss = criterion(seg_pred, labels.view(-1),
                             trans_feat, weights_tensor)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_loss = train_loss / len(train_loader)
        writer.add_scalar('Loss/Train', avg_loss, epoch)

        # --- VALIDIERUNG (Testen, ohne zu lernen) ---
        model.eval()
        all_ious = []

        with torch.no_grad():
            for points, labels in tqdm(val_loader,
                                       desc=f"Epoche {epoch}/{EPOCHS} [Val]"):
                points = points.float().to(device)
                labels_np = labels.numpy()

                inp = points.permute(0, 2, 1).contiguous()  # (B, 6, N)
                seg_pred, _ = model(inp)                     # (B, N, C)

                # Hoechste Wahrscheinlichkeit gewinnt -> ueber Klassen-Dim (2)
                pred_np = seg_pred.argmax(dim=2).cpu().numpy()  # (B, N)

                for b in range(pred_np.shape[0]):
                    all_ious.append(
                        calculate_iou(pred_np[b], labels_np[b], NUM_CLASSES))

        iou_array = np.nanmean(np.stack(all_ious), axis=0)
        miou = np.nanmean(iou_array)

        writer.add_scalar('Metric/mIoU', miou, epoch)
        for i, name in enumerate(CLASS_NAMES):
            writer.add_scalar(f'IoU/{name}', iou_array[i], epoch)

        print(f"\nErgebnis Epoche {epoch}: Loss = {avg_loss:.4f} "
              f"| mIoU = {miou:.4f}")
        for name, val in zip(CLASS_NAMES, iou_array):
            print(f"  {name:<10} IoU = {val:.4f}")

        # Bestes Modell speichern
        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(),
                       os.path.join(LOG_DIR, 'best_model.pth'))
            print("  -> Neues bestes Modell gespeichert.")

    writer.close()
    print(f"\nTraining abgeschlossen. Bestes mIoU: {best_miou:.4f}")


if __name__ == '__main__':
    main()
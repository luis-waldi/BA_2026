import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import json
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from collections import Counter
from tqdm import tqdm

from data_utils.heath_dataset_v2 import HeideDatasetV2, NUM_CLASSES, CLASS_NAMES

sys.path.append('./models')
from models.pointnet2_sem_seg import get_model

TILE_DIR    = os.path.join(DATA_ROOT, 'processed', 'training_tiles_v3')
INIT_MODEL  = './logs/heath_run2/best_model.pth'
LOG_DIR     = './logs/heath_run3'

EPOCHS      = 64
BATCH_SIZE  = 8
LR          = 1e-4
ETA_MIN     = 1e-6
NUM_WORKERS = 4

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Device: {device}')

with open(os.path.join(TILE_DIR, 'splits.json')) as f:
    splits = json.load(f)

train_list = splits['train']
val_list   = splits['val']
print(f'Train: {len(train_list)}  Val: {len(val_list)}')

train_ds = HeideDatasetV2(TILE_DIR, train_list, augment=True)
val_ds   = HeideDatasetV2(TILE_DIR, val_list,   augment=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, drop_last=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0)

print('Berechne Klassengewichte (1/sqrt(count)) ...')
counts = Counter()
for fname in train_list:
    data = np.loadtxt(os.path.join(TILE_DIR, fname), dtype=np.float32)
    lbl  = data[:, 6].astype(np.int64)
    counts.update(lbl.tolist())

count_arr = np.array([counts.get(c, 0) for c in range(NUM_CLASSES)], dtype=np.float64)
count_arr = np.maximum(count_arr, 1)
class_w   = 1.0 / np.sqrt(count_arr)
class_w   = class_w / class_w.sum() * NUM_CLASSES
class_weights = torch.tensor(class_w, dtype=torch.float32, device=device)

print('Klassengewichte:')
for i, n in enumerate(CLASS_NAMES):
    print(f'  {n:10s} count={int(count_arr[i]):>9d}  w={class_w[i]:.4f}')

model = get_model(NUM_CLASSES).to(device)

if os.path.exists(INIT_MODEL):
    print(f'Lade Warmstart-Gewichte: {INIT_MODEL}')
    ckpt = torch.load(INIT_MODEL, map_location=device)
    state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state)
else:
    print('WARNUNG: kein Warmstart-Modell gefunden, starte von Scratch')

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=ETA_MIN)

def weighted_loss(seg_pred, target, point_w):
    seg_pred = seg_pred.reshape(-1, NUM_CLASSES)
    target   = target.reshape(-1)
    point_w  = point_w.reshape(-1)

    loss_pp = F.nll_loss(seg_pred, target,
                         weight=class_weights,
                         reduction='none')
    loss = (loss_pp * point_w).sum() / (point_w.sum() + 1e-8)
    return loss

def compute_iou(conf_mat):
    ious = []
    for c in range(NUM_CLASSES):
        tp = conf_mat[c, c]
        fp = conf_mat[:, c].sum() - tp
        fn = conf_mat[c, :].sum() - tp
        denom = tp + fp + fn
        ious.append(tp / denom if denom > 0 else float('nan'))
    return ious

os.makedirs(LOG_DIR, exist_ok=True)
writer = SummaryWriter(LOG_DIR)
best_miou = 0.0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for points, labels, pw in tqdm(train_loader, desc=f'Epoche {epoch+1}/{EPOCHS} [Train]'):
        points = points.to(device)
        labels = labels.to(device)
        pw     = pw.to(device)

        inp = points.transpose(2, 1).contiguous()

        optimizer.zero_grad()
        seg_pred, _ = model(inp)
        loss = weighted_loss(seg_pred, labels, pw)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)
    scheduler.step()

    model.eval()
    conf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    val_loss = 0.0
    with torch.no_grad():
        for points, labels, pw in tqdm(val_loader, desc=f'Epoche {epoch+1}/{EPOCHS} [Val]'):
            points = points.to(device)
            labels = labels.to(device)
            pw     = pw.to(device)
            inp = points.transpose(2, 1).contiguous()

            seg_pred, _ = model(inp)
            val_loss += weighted_loss(seg_pred, labels, pw).item()

            pred = seg_pred.argmax(dim=2)
            p = pred.cpu().numpy().reshape(-1)
            t = labels.cpu().numpy().reshape(-1)
            for ti, pi in zip(t, p):
                conf_mat[ti, pi] += 1

    val_loss /= len(val_loader)
    ious = compute_iou(conf_mat)
    miou = np.nanmean(ious)

    lr_now = optimizer.param_groups[0]['lr']
    print(f'\nEpoche {epoch+1}/{EPOCHS}  lr={lr_now:.2e}')
    print(f'  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  mIoU={miou:.4f}')
    for i, n in enumerate(CLASS_NAMES):
        print(f'    {n:10s} IoU={ious[i]:.4f}')

    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val',   val_loss,   epoch)
    writer.add_scalar('mIoU/val',   miou,       epoch)
    writer.add_scalar('LR',         lr_now,     epoch)
    for i, n in enumerate(CLASS_NAMES):
        writer.add_scalar(f'IoU/{n}', ious[i], epoch)

    if miou > best_miou:
        best_miou = miou
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'miou': miou,
        }, os.path.join(LOG_DIR, 'best_model.pth'))
        print(f'  -> Neues bestes Modell gespeichert (mIoU={miou:.4f})')

writer.close()
print(f'\nFertig. Bestes mIoU: {best_miou:.4f}')
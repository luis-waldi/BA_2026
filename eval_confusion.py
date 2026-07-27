import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

sys.path.append('./models')
from models.pointnet2_sem_seg import get_model
from data_utils.heath_dataset_v2 import HeideDatasetV2, NUM_CLASSES, CLASS_NAMES

TILE_DIR   = '/Users/luis/Documents/BA/data/processed/training_tiles_v3'
MODEL_PATH = './log/sem_seg/run_v4_full/checkpoints/best_model.pth'

device = torch.device('mps' if torch.backends.mps.is_available()
                      else 'cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# Auswertung auf dem Test-Split
with open(os.path.join(TILE_DIR, 'splits.json')) as f:
    splits = json.load(f)
test_list = splits['test']
print(f'Test-Tiles: {len(test_list)}')

test_ds     = HeideDatasetV2(TILE_DIR, test_list, augment=False)
test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0)

model = get_model(NUM_CLASSES).to(device)
ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
model.load_state_dict(state)
model.eval()

saved_iou = ckpt.get('class_avg_iou', ckpt.get('miou'))
info = f'Epoche {ckpt.get("epoch", "?")}'
if saved_iou is not None:
    info += f', mIoU={saved_iou:.4f}'
print(f'Modell geladen: {MODEL_PATH} ({info})')

# Confusion Matrix ueber den Test-Split
conf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
with torch.no_grad():
    for points, labels, pw in test_loader:
        points = points.to(device)
        labels = labels.to(device)
        inp    = points.transpose(2, 1).contiguous()
        seg_pred, _ = model(inp)
        pred = seg_pred.argmax(dim=2)
        p = pred.cpu().numpy().reshape(-1)
        t = labels.cpu().numpy().reshape(-1)
        for ti, pi in zip(t, p):
            conf_mat[ti, pi] += 1

# Rohe Confusion Matrix
print('\n--- Confusion Matrix (Zeile=True, Spalte=Pred) ---')
header = f'{"":12s}' + ''.join(f'{n:12s}' for n in CLASS_NAMES)
print(header)
print('-' * (12 + 12 * NUM_CLASSES))
for i, name in enumerate(CLASS_NAMES):
    row = f'{name:12s}' + ''.join(f'{conf_mat[i, j]:12d}' for j in range(NUM_CLASSES))
    print(row)

# Normalisiert je wahrer Klasse
print('\n--- Confusion Matrix normalisiert (% der wahren Klasse) ---')
print(header)
print('-' * (12 + 12 * NUM_CLASSES))
row_sums = conf_mat.sum(axis=1, keepdims=True).clip(min=1)
conf_norm = conf_mat / row_sums * 100
for i, name in enumerate(CLASS_NAMES):
    row = f'{name:12s}' + ''.join(f'{conf_norm[i, j]:11.1f}%' for j in range(NUM_CLASSES))
    print(row)

# IoU je Klasse + mIoU
print('\n--- IoU pro Klasse ---')
ious = []
for c in range(NUM_CLASSES):
    tp    = conf_mat[c, c]
    fp    = conf_mat[:, c].sum() - tp
    fn    = conf_mat[c, :].sum() - tp
    denom = tp + fp + fn
    iou   = tp / denom if denom > 0 else float('nan')
    ious.append(iou)
    total_true = conf_mat[c, :].sum()
    print(f'  {CLASS_NAMES[c]:12s}  IoU={iou:.4f}  (TP={tp:7d} / {total_true:7d} wahre Punkte)')
miou = np.nanmean(ious)
print(f'\n  mIoU = {miou:.4f}')

# Groesste Verwechslungen
print('\n--- Top-10 Verwechslungen (False Positives) ---')
errors = []
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        if i != j and conf_mat[i, j] > 0:
            errors.append((conf_mat[i, j], CLASS_NAMES[i], CLASS_NAMES[j]))
errors.sort(reverse=True)
print(f'  {"Punkte":>10s}  {"True":12s} -> {"Pred":12s}')
for count, true_cls, pred_cls in errors[:10]:
    print(f'  {count:10d}  {true_cls:12s} -> {pred_cls:12s}')

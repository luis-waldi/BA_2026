import os
import sys
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

sys.path.append('./models')
from models.pointnet2_sem_seg import get_model
from data_utils.heath_dataset_v2 import HeideDatasetV2, NUM_CLASSES, CLASS_NAMES, SCHEMA


def parse_args():
    p = argparse.ArgumentParser('Auswertung PointNet++ Heide')
    p.add_argument('--tile_dir', type=str,
                   default='/Users/luis/Documents/BA/data/processed/training_tiles_v3')
    p.add_argument('--model_path', type=str,
                   default='./log/sem_seg/run01/checkpoints/best_model.pth')
    p.add_argument('--split', type=str, default='test', help='train, val oder test')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--out', type=str, default=None,
                   help='Protokolldatei [default: <run>_eval_<split>.txt]')
    return p.parse_args()


class Tee:
    """Ausgabe gleichzeitig auf die Konsole und in eine Datei."""

    def __init__(self, path):
        self.stream = sys.stdout
        self.file = open(path, 'w')

    def write(self, text):
        self.stream.write(text)
        self.file.write(text)

    def flush(self):
        self.stream.flush()
        self.file.flush()


args = parse_args()

# Laufname aus dem Checkpoint-Pfad: log/sem_seg/<run>/checkpoints/best_model.pth
run_name = os.path.basename(os.path.dirname(os.path.dirname(args.model_path)))
out_path = args.out or f'{run_name}_eval_{args.split}.txt'
sys.stdout = Tee(out_path)

device = torch.device('mps' if torch.backends.mps.is_available()
                      else 'cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

with open(os.path.join(args.tile_dir, 'splits.json')) as f:
    splits = json.load(f)
tile_list = splits[args.split]
print(f'{args.split}-Tiles: {len(tile_list)}')

ds     = HeideDatasetV2(args.tile_dir, tile_list, augment=False)
loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

print(f'Schema: {SCHEMA}  Features je Punkt: {ds.n_feat}')
model = get_model(NUM_CLASSES, in_channel=ds.n_feat).to(device)
ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
model.load_state_dict(state)
model.eval()

saved_iou = ckpt.get('class_avg_iou', ckpt.get('miou'))
info = f'Epoche {ckpt.get("epoch", "?")}'
if saved_iou is not None:
    info += f', mIoU={saved_iou:.4f}'
print(f'Modell geladen: {args.model_path} ({info})')

# Confusion Matrix ueber den Split
conf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
with torch.no_grad():
    for points, labels, pw in loader:
        points = points.to(device)
        labels = labels.to(device)
        inp    = points.transpose(2, 1).contiguous()
        seg_pred, _ = model(inp)
        pred = seg_pred.argmax(dim=2)
        p = pred.cpu().numpy().reshape(-1)
        t = labels.cpu().numpy().reshape(-1)
        for ti, pi in zip(t, p):
            if ti < NUM_CLASSES:          
                conf_mat[ti, pi] += 1

# Kennzahlen je Klasse aus der Confusion Matrix 
seen_class    = conf_mat.sum(axis=1)                       
correct_class = np.diag(conf_mat)                          
deno_class    = conf_mat.sum(axis=0) + conf_mat.sum(axis=1) - correct_class  

overall_acc = correct_class.sum() / max(conf_mat.sum(), 1)
class_acc   = correct_class / np.maximum(seen_class, 1)    
class_iou   = correct_class / np.maximum(deno_class, 1)    
mean_acc = class_acc.mean()
mIoU     = class_iou.mean()

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

# Accuracy und IoU je Klasse
print('\n--- Accuracy und IoU je Klasse ---')
print(f'  {"Klasse":12s} {"Accuracy":>10s} {"IoU":>10s} {"wahre Punkte":>14s}')
for c in range(NUM_CLASSES):
    print(f'  {CLASS_NAMES[c]:12s} {class_acc[c]:10.4f} {class_iou[c]:10.4f} {int(seen_class[c]):14d}')

print(f'\n  Overall Accuracy      = {overall_acc:.4f}')
print(f'  mittlere Accuracy     = {mean_acc:.4f}')
print(f'  mIoU (Mittel je Klasse) = {mIoU:.4f}')

# Groesste Verwechslungen
print('\n--- Top-10 Verwechslungen ---')
errors = []
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        if i != j and conf_mat[i, j] > 0:
            errors.append((conf_mat[i, j], CLASS_NAMES[i], CLASS_NAMES[j]))
errors.sort(reverse=True)
print(f'  {"Punkte":>10s}  {"True":12s} -> {"Pred":12s}')
for count, true_cls, pred_cls in errors[:10]:
    print(f'  {count:10d}  {true_cls:12s} -> {pred_cls:12s}')

# Gehoelz gegen Rest, unabhaengig vom trainierten Schema. Erlaubt den
# Vergleich mit dem binaeren Modell und ist die fuer den Verbuschungsgrad
# relevante Groesse.
if SCHEMA in ('taxo6', 'taxo8'):
    geh = [i for i, n in enumerate(CLASS_NAMES) if n in ('bush', 'tree')]
    rest = [i for i in range(NUM_CLASSES) if i not in geh]
    tp = conf_mat[np.ix_(geh, geh)].sum()
    fn = conf_mat[np.ix_(geh, rest)].sum()
    fp = conf_mat[np.ix_(rest, geh)].sum()
    tn = conf_mat[np.ix_(rest, rest)].sum()
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    print('\n--- Gehoelz (bush+tree) gegen Rest, nachtraeglich zusammengefasst ---')
    print(f'  IoU Gehoelz  = {tp / max(tp + fp + fn, 1):.4f}')
    print(f'  IoU Rest     = {tn / max(tn + fp + fn, 1):.4f}')
    print(f'  Precision    = {prec:.4f}')
    print(f'  Recall       = {rec:.4f}')
    print(f'  F1           = {2 * prec * rec / max(prec + rec, 1e-9):.4f}')

sys.stdout.flush()
print(f'\nProtokoll geschrieben: {out_path}')

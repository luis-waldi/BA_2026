"""Inferenz auf ungelabelten Befliegungsgebieten.

Liest die von InferenzPreprocessing.R erzeugten normalisierten LAZ-Bloecke,
zerlegt sie in dieselben Fenster wie beim Training und sagt fuer jeden Punkt
eine Vegetationsklasse vorher.

Zwei Punkte, die den Unterschied zur reinen Auswertung ausmachen:

Der Stride von 5 m bei 10 m Fenstern bedeutet, dass jeder Ort in bis zu vier
Fenstern liegt. Statt vier harte Vorhersagen per Mehrheit auszuzaehlen, werden
die Wahrscheinlichkeiten gemittelt und erst danach das Argmax gebildet. Eine
sichere Vorhersage zaehlt so mehr als eine unsichere, und Randfehler der
Fenster glaetten sich.

Fenster an den Grenzen eines Verarbeitungsblocks brauchen Punkte des
Nachbarblocks, sonst sieht das Netz dort eine halbe Nachbarschaft. Deshalb
wird jeder Block zusammen mit einem Saum aus den angrenzenden Bloecken
geladen, vorhergesagt werden aber nur die Punkte des Blocks selbst.

Aufruf:
    HEATH_SCHEMA=taxo6 HEATH_RADII=wide python predict_heath.py \
        --block_dir /home/l/lwaldeye/data/inferenz_110kv \
        --model_path ./log/sem_seg/run07_radien/checkpoints/best_model.pth \
        --out_dir /home/l/lwaldeye/data/vorhersage_110kv
"""

import os
import re
import glob
import argparse
import importlib

import numpy as np
import torch
import laspy
from tqdm import tqdm

os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

from data_utils.heath_dataset_v2 import NUM_CLASSES, CLASS_NAMES, SCHEMA

TILE_SIZE = 10.0    # m, wie im Trainingsexport
STRIDE = 5.0        # m, 50 Prozent Ueberlappung
N_POINTS = 8192     # Punkte je Fenster
HALO = TILE_SIZE    # m, Saum aus den Nachbarbloecken
COLOR_MAX = 65535.0  # wie im R-Export


def parse_args():
    p = argparse.ArgumentParser('Inferenz PointNet++ Heide')
    p.add_argument('--block_dir', type=str, required=True,
                   help='Verzeichnis mit den normalisierten LAZ-Bloecken')
    p.add_argument('--out_dir', type=str, required=True)
    p.add_argument('--model', type=str, default='pointnet2_sem_seg')
    p.add_argument('--model_path', type=str, required=True)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--write_laz', action='store_true',
                   help='zusaetzlich eine LAZ je Block fuer die Sichtpruefung')
    return p.parse_args()


def read_block(path):
    """Punkte eines Blocks als Merkmalsmatrix in der Reihenfolge des Trainings."""
    las = laspy.read(path)
    n = len(las.points)
    if n == 0:
        return None, None
    xy = np.empty((n, 2), dtype=np.float64)
    xy[:, 0] = las.x
    xy[:, 1] = las.y
    feat = np.empty((n, 7), dtype=np.float32)
    feat[:, 0] = las.z                       # normalisierte Hoehe
    feat[:, 1] = np.asarray(las.red) / COLOR_MAX
    feat[:, 2] = np.asarray(las.green) / COLOR_MAX
    feat[:, 3] = np.asarray(las.blue) / COLOR_MAX
    feat[:, 4] = np.asarray(las.nir) / COLOR_MAX
    feat[:, 5] = np.asarray(las.z_rel)
    feat[:, 6] = np.asarray(las.z_range)
    return xy, feat


def block_origin(path):
    """Blockursprung aus dem Dateinamen norm_{XLEFT}_{YBOTTOM}.laz."""
    m = re.search(r'norm_(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)', os.path.basename(path))
    if m is None:
        raise ValueError(f'Blockursprung nicht aus dem Namen ableitbar: {path}')
    return float(m.group(1)), float(m.group(2))


def window_origins(lo, hi):
    """Fensterursprunge auf einem globalen Vielfachen von STRIDE, damit die
    Fenster ueber Blockgrenzen hinweg auf demselben Raster liegen. Der
    Bereich ist so gewaehlt, dass jeder Punkt zwischen lo und hi in
    mindestens einem, im Regelfall in zwei Fenstern je Achse liegt."""
    first = np.floor((lo - TILE_SIZE) / STRIDE) * STRIDE
    last = np.ceil(hi / STRIDE) * STRIDE
    return np.arange(first, last + STRIDE, STRIDE)


def build_cell_index(xy, gx0, gy0):
    """Punkte in Zellen der Kantenlaenge STRIDE einsortieren. Der Ursprung
    liegt auf einem Vielfachen von STRIDE, dadurch deckt ein 10-m-Fenster
    genau zwei Zellen je Achse ab."""
    cx = np.floor((xy[:, 0] - gx0) / STRIDE).astype(np.int64)
    cy = np.floor((xy[:, 1] - gy0) / STRIDE).astype(np.int64)
    ncy = int(cy.max()) + 2
    key = cx * ncy + cy
    order = np.argsort(key, kind='stable')
    ks = key[order]
    uk, start = np.unique(ks, return_index=True)
    end = np.append(start[1:], len(ks))
    return order, dict(zip(uk.tolist(), zip(start.tolist(), end.tolist()))), ncy


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'mps' if torch.backends.mps.is_available() else 'cpu')

    MODEL = importlib.import_module(f'models.{args.model}')
    model = MODEL.get_model(NUM_CLASSES, in_channel=9).to(device)
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get('model_state_dict', ckpt))
    model.eval()

    print(f'Device: {device}')
    print(f'Modell: {args.model}  Schema: {SCHEMA}  Klassen: {CLASS_NAMES}')
    print(f'Radien: {MODEL.RADII_NAME} {MODEL.RADII}')
    print(f'Checkpoint: {args.model_path} (Epoche {ckpt.get("epoch", "?")})')

    blocks = sorted(glob.glob(os.path.join(args.block_dir, '*.laz')))
    if not blocks:
        raise SystemExit(f'Keine LAZ-Bloecke in {args.block_dir}')
    origins = {b: block_origin(b) for b in blocks}
    print(f'Bloecke: {len(blocks)}')

    ges_punkte = 0
    ges_ohne = 0

    for path in tqdm(blocks, desc='Bloecke'):
        xy0, feat0 = read_block(path)
        if xy0 is None:
            continue
        bx, by = origins[path]
        # Blockkern aus den tatsaechlichen Punkten, nicht aus dem Namen, denn
        # der letzte Block einer Reihe kann schmaler sein.
        x_lo, y_lo = xy0[:, 0].min(), xy0[:, 1].min()
        x_hi, y_hi = xy0[:, 0].max(), xy0[:, 1].max()

        # Saum aus den Nachbarbloecken dazuladen. Vorhergesagt wird nur der
        # eigene Block, die Saumpunkte liefern nur Nachbarschaft.
        xy_list, feat_list = [xy0], [feat0]
        for other in blocks:
            if other == path:
                continue
            ox, oy = origins[other]
            if (ox > x_hi + HALO or ox + 100 < x_lo - HALO or
                    oy > y_hi + HALO or oy + 100 < y_lo - HALO):
                continue
            xy1, feat1 = read_block(other)
            if xy1 is None:
                continue
            keep = ((xy1[:, 0] > x_lo - HALO) & (xy1[:, 0] < x_hi + HALO) &
                    (xy1[:, 1] > y_lo - HALO) & (xy1[:, 1] < y_hi + HALO))
            if keep.any():
                xy_list.append(xy1[keep])
                feat_list.append(feat1[keep])

        n_own = len(xy0)
        xy = np.concatenate(xy_list)
        feat = np.concatenate(feat_list)

        prob_sum = np.zeros((n_own, NUM_CLASSES), dtype=np.float32)
        n_seen = np.zeros(n_own, dtype=np.int32)

        # Punkte in ein grobes Raster einsortieren, damit die Fensterabfrage
        # nicht jedes Mal ueber alle Punkte laeuft.
        gx0 = np.floor((xy[:, 0].min() - STRIDE) / STRIDE) * STRIDE
        gy0 = np.floor((xy[:, 1].min() - STRIDE) / STRIDE) * STRIDE
        order, cells, ncy = build_cell_index(xy, gx0, gy0)

        batch_pts, batch_idx = [], []

        def flush():
            if not batch_pts:
                return
            arr = torch.from_numpy(np.stack(batch_pts)).to(device)
            with torch.no_grad():
                logits, _ = model(arr.transpose(2, 1).contiguous())
                prob = torch.exp(logits).cpu().numpy()   # Modell liefert log_softmax
            for p, idx in zip(prob, batch_idx):
                own = idx < n_own
                if own.any():
                    np.add.at(prob_sum, idx[own], p[own])
                    np.add.at(n_seen, idx[own], 1)
            batch_pts.clear()
            batch_idx.clear()

        for xi in window_origins(x_lo, x_hi):
            cx = int(round((xi - gx0) / STRIDE))
            for yi in window_origins(y_lo, y_hi):
                cy = int(round((yi - gy0) / STRIDE))
                # Ein 10-m-Fenster deckt genau zwei Zellen je Achse ab
                cand = []
                for dx in (0, 1):
                    for dy in (0, 1):
                        se = cells.get((cx + dx) * ncy + (cy + dy))
                        if se is not None:
                            cand.append(order[se[0]:se[1]])
                if not cand:
                    continue
                idx = np.concatenate(cand)
                sub = xy[idx]
                inside = ((sub[:, 0] >= xi) & (sub[:, 0] < xi + TILE_SIZE) &
                          (sub[:, 1] >= yi) & (sub[:, 1] < yi + TILE_SIZE))
                idx = idx[inside]
                if len(idx) < 100 or not (idx < n_own).any():
                    continue

                take = np.random.choice(len(idx), N_POINTS, replace=len(idx) < N_POINTS)
                sel = idx[take]
                pts = np.empty((N_POINTS, 9), dtype=np.float32)
                pts[:, 0] = xy[sel, 0] - (xi + TILE_SIZE / 2)
                pts[:, 1] = xy[sel, 1] - (yi + TILE_SIZE / 2)
                pts[:, 2:] = feat[sel]

                batch_pts.append(pts)
                batch_idx.append(sel)
                if len(batch_pts) == args.batch_size:
                    flush()
        flush()

        pred = np.full(n_own, 255, dtype=np.uint8)
        conf = np.zeros(n_own, dtype=np.float32)
        hit = n_seen > 0
        mean = prob_sum[hit] / n_seen[hit, None]
        pred[hit] = mean.argmax(axis=1).astype(np.uint8)
        conf[hit] = mean.max(axis=1)

        ges_punkte += n_own
        ges_ohne += int((~hit).sum())

        base = os.path.splitext(os.path.basename(path))[0]
        np.savez_compressed(
            os.path.join(args.out_dir, base + '_pred.npz'),
            x=xy0[:, 0], y=xy0[:, 1], z=feat0[:, 0],
            pred=pred, conf=conf, n_seen=n_seen)

        if args.write_laz:
            src = laspy.read(path)
            src.classification = np.where(pred == 255, 0, pred).astype(np.uint8)
            src.write(os.path.join(args.out_dir, base + '_pred.laz'))

    print(f'\nPunkte gesamt: {ges_punkte:,}')
    print(f'ohne Vorhersage (kein Fenster getroffen): {ges_ohne:,} '
          f'({100 * ges_ohne / max(ges_punkte, 1):.2f} %)')
    print(f'Ergebnisse in {args.out_dir}')


if __name__ == '__main__':
    main()

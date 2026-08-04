# Training PointNet++ zur semantischen Segmentierung der Heidevegetation.
# Trainingsablauf angelehnt an yanx27/Pointnet_Pointnet2_pytorch (train_semseg.py).
# Angepasst an eigenen Datensatz (8 bzw. 6 Klassen, 6 Kanaele x,y,z,R,G,B, NIR), Apple Silicon
# (MPS) und einen zusaetzlich per-Punkt gewichteten Loss.
# Modell: PointNet++.
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import sys
import json
import shutil
import logging
import argparse
import datetime
import importlib
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import provider
from data_utils.heath_dataset_v2 import (HeideDatasetV2, NUM_CLASSES, CLASS_NAMES,
                                         IGNORE_INDEX, SCHEMA)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'models'))

TILE_DIR_DEFAULT = '/Users/luis/Documents/BA/data/processed/training_tiles_v3'


def inplace_relu(m):
    if m.__class__.__name__.find('ReLU') != -1:
        m.inplace = True


def parse_args():
    p = argparse.ArgumentParser('PointNet++ Heide semseg')
    p.add_argument('--model', type=str, default='pointnet2_sem_seg', help='Modelldatei in models/')
    p.add_argument('--tile_dir', type=str, default=TILE_DIR_DEFAULT)
    p.add_argument('--log_dir', type=str, default=None, help='Name unter log/sem_seg/ [default: Zeitstempel]')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--epoch', type=int, default=64)
    p.add_argument('--learning_rate', type=float, default=1e-3)
    p.add_argument('--optimizer', type=str, default='Adam', help='Adam oder SGD')
    p.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay')
    p.add_argument('--step_size', type=int, default=10, help='Epochen je LR-Abfall')
    p.add_argument('--lr_decay', type=float, default=0.7)
    p.add_argument('--num_workers', type=int, default=0, help='0 auf Apple Silicon empfohlen')
    p.add_argument('--loss', type=str, default='weighted', choices=['weighted', 'focal'])
    p.add_argument('--gamma', type=float, default=2.0, help='Focusing-Parameter fuer Focal Loss')
    return p.parse_args()


def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def compute_class_weights(labels_list, device):
    """Klassengewichte 1/sqrt(count) gegen die Klassen-Imbalance."""
    counts = Counter()
    for lbl in labels_list:
        counts.update(lbl.tolist())
    arr = np.array([counts.get(c, 0) for c in range(NUM_CLASSES)], dtype=np.float64)
    arr = np.maximum(arr, 1)
    w = 1.0 / np.sqrt(arr)
    w = w / w.sum() * NUM_CLASSES
    return torch.tensor(w, dtype=torch.float32, device=device), arr


def main(args):
    device = get_device()

    # ---- Verzeichnisse (log/sem_seg/<name>/{checkpoints,logs}) ----
    timestr = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    experiment_dir = Path('./log/sem_seg')
    experiment_dir.mkdir(parents=True, exist_ok=True)
    experiment_dir = experiment_dir.joinpath(args.log_dir if args.log_dir else timestr)
    experiment_dir.mkdir(exist_ok=True)
    ckpt_dir = experiment_dir.joinpath('checkpoints'); ckpt_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs'); log_dir.mkdir(exist_ok=True)

    # ---- Logging ----
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(fh)

    def log_string(s):
        logger.info(s)
        print(s)

    log_string(args)
    log_string('Device: %s' % device)

    # ---- Daten (Split aus splits.json) ----
    with open(os.path.join(args.tile_dir, 'splits.json')) as f:
        splits = json.load(f)
    train_list, val_list = splits['train'], splits['val']

    # Augmentierung erfolgt im Loop ueber provider, daher hier augment=False
    train_ds = HeideDatasetV2(args.tile_dir, train_list, augment=False)
    val_ds = HeideDatasetV2(args.tile_dir, val_list, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
    log_string('Train: %d  Val: %d' % (len(train_ds), len(val_ds)))

    # ---- Klassengewichte ----
    class_weights, counts = compute_class_weights(train_ds.labels_list, device)
    log_string('Klassengewichte:')
    for i, n in enumerate(CLASS_NAMES):
        log_string('  %-10s count=%9d  w=%.4f' % (n, int(counts[i]), class_weights[i]))

    # ---- Modell laden ----
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('models/pointnet2_utils.py', str(experiment_dir))

    log_string('Schema: %s  Features je Punkt: %d  Radien: %s %s'
               % (SCHEMA, train_ds.n_feat, MODEL.RADII_NAME, MODEL.RADII))
    classifier = MODEL.get_model(NUM_CLASSES, in_channel=train_ds.n_feat).to(device)
    classifier.apply(inplace_relu)

    def weights_init(m):
        cn = m.__class__.__name__
        if cn.find('Conv2d') != -1 or cn.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load(str(ckpt_dir) + '/best_model.pth', map_location=device)
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Vorhandenes Modell gefunden, setze Training fort (Resume).')
    except Exception:
        log_string('Kein Modell gefunden, Training von Scratch.')
        start_epoch = 0
        classifier = classifier.apply(weights_init)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(classifier.parameters(), lr=args.learning_rate,
                                     betas=(0.9, 0.999), eps=1e-8, weight_decay=args.decay_rate)
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            m.momentum = momentum

    LR_CLIP = 1e-5
    MOM_ORIG = 0.1
    MOM_DECAY = 0.5
    MOM_STEP = args.step_size

    # Loss: klassengewichtetes NLL, zusaetzlich per-Punkt gewichtet, um unsichere
    # Labels (Schatten, Ueberbelichtung, geringe Confidence) abzuschwaechen.
    # Focal Loss zusaetzlich, der leichte, sichere Punkte daempft
    # und den Fokus auf schwere/seltene Punkte legt.
    def weighted_loss(seg_pred, target, point_w):
        seg_pred = seg_pred.reshape(-1, NUM_CLASSES)
        target = target.reshape(-1)
        point_w = point_w.reshape(-1)
        loss_pp = F.nll_loss(seg_pred, target, weight=class_weights,
                             reduction='none', ignore_index=IGNORE_INDEX)
        if args.loss == 'focal':
            tgt = target.clamp(0, NUM_CLASSES - 1)  # ignorierte Labels sind hier egal, loss_pp=0
            logpt = seg_pred.gather(1, tgt.unsqueeze(1)).squeeze(1)  # log p_t
            loss_pp = (1.0 - logpt.exp()) ** args.gamma * loss_pp
        point_w = point_w * (target != IGNORE_INDEX).float()  # ignorierte Punkte nicht mitzaehlen
        return (loss_pp * point_w).sum() / (point_w.sum() + 1e-8)

    best_iou = 0.0

    for epoch in range(start_epoch, args.epoch):
        log_string('**** Epoch %d/%d ****' % (epoch + 1, args.epoch))
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LR_CLIP)
        for g in optimizer.param_groups:
            g['lr'] = lr
        momentum = max(MOM_ORIG * (MOM_DECAY ** (epoch // MOM_STEP)), 0.01)
        classifier.apply(lambda m: bn_momentum_adjust(m, momentum))
        log_string('lr=%.6f  bn_momentum=%.3f' % (lr, momentum))

        # ---- Training ----
        classifier.train()
        loss_sum = 0.0
        total_correct = 0
        total_seen = 0
        for points, target, pw in tqdm(train_loader, total=len(train_loader), smoothing=0.9):
            # Augmentierung nur auf die xyz-Kanaele: Rotation um Z + leichter Jitter
            pts = points.numpy()
            pts[:, :, :3] = provider.rotate_point_cloud_z(pts[:, :, :3])
            pts[:, :, :3] = provider.jitter_point_cloud(pts[:, :, :3], sigma=0.01, clip=0.05)
            points = torch.Tensor(pts)

            points = points.float().to(device)
            target = target.long().to(device)
            pw = pw.float().to(device)
            points = points.transpose(2, 1)          

            optimizer.zero_grad()
            seg_pred, trans_feat = classifier(points)  
            loss = weighted_loss(seg_pred, target, pw)
            loss.backward()
            optimizer.step()

            pred = seg_pred.argmax(dim=2)
            valid = target != IGNORE_INDEX
            total_correct += ((pred == target) & valid).sum().item()
            total_seen += valid.sum().item()
            loss_sum += loss.item()

        log_string('Train  loss=%.4f  OA=%.4f' %
                   (loss_sum / len(train_loader), total_correct / total_seen))

        # ---- Validierung ----
        classifier.eval()
        with torch.no_grad():
            loss_sum = 0.0
            total_correct = 0
            total_seen = 0
            seen_c = [0] * NUM_CLASSES
            corr_c = [0] * NUM_CLASSES
            deno_c = [0] * NUM_CLASSES
            for points, target, pw in tqdm(val_loader, total=len(val_loader), smoothing=0.9):
                points = points.float().to(device)
                target = target.long().to(device)
                pw = pw.float().to(device)
                points = points.transpose(2, 1)

                seg_pred, trans_feat = classifier(points)
                loss_sum += weighted_loss(seg_pred, target, pw).item()

                pred = seg_pred.argmax(dim=2).cpu().numpy()
                gt = target.cpu().numpy()
                valid = gt != IGNORE_INDEX
                total_correct += np.sum((pred == gt) & valid)
                total_seen += np.sum(valid)
                for l in range(NUM_CLASSES):
                    seen_c[l] += np.sum(gt == l)
                    corr_c[l] += np.sum((pred == l) & (gt == l))
                    deno_c[l] += np.sum((pred == l) | (gt == l))

            iou = np.array(corr_c) / (np.array(deno_c, dtype=np.float64) + 1e-6)
            mIoU = float(np.mean(iou))
            log_string('Val    loss=%.4f  OA=%.4f  mIoU=%.4f' %
                       (loss_sum / len(val_loader), total_correct / total_seen, mIoU))
            s = '------- IoU je Klasse --------\n'
            for l in range(NUM_CLASSES):
                s += '  %-10s IoU=%.3f\n' % (CLASS_NAMES[l], iou[l])
            log_string(s)

            if mIoU >= best_iou:
                best_iou = mIoU
                savepath = str(ckpt_dir) + '/best_model.pth'
                torch.save({
                    'epoch': epoch,
                    'class_avg_iou': mIoU,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, savepath)
                log_string('  -> bestes Modell gespeichert (mIoU=%.4f)' % mIoU)
            log_string('Best mIoU: %.4f' % best_iou)


if __name__ == '__main__':
    main(parse_args())

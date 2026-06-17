import os
import numpy as np
from torch.utils.data import Dataset

NUM_CLASSES = 8
CLASS_NAMES = ['bush','deadwood','graminoid','heath',
               'other','sand','soil','tree']

# weight_calc.py (Referenz fuer heath_dataset_v2.py)
CONFIDENCE_MAX    = 10     # Skala 2-10, 9 fehlt in Daten – irrelevant
SHADOW_WEIGHT     = 0.3
OVEREXPOSED_WEIGHT = 0.3
MIN_WEIGHT        = 0.05

def point_weight(shadow: bool, overexposed: bool, confidence: int) -> float:
    w = confidence / CONFIDENCE_MAX          # [0.2, 1.0]
    if shadow:      w *= SHADOW_WEIGHT       # *= 0.3
    if overexposed: w *= OVEREXPOSED_WEIGHT  # *= 0.3
    return max(w, MIN_WEIGHT)               # clip auf 0.05

class HeideDatasetV2(Dataset):
    def __init__(self, tile_dir, tile_list, augment=False):
        self.tile_dir = tile_dir
        self.tile_list = tile_list
        self.augment = augment

    def __len__(self):
        return len(self.tile_list)

    def __getitem__(self, idx):
        path = os.path.join(self.tile_dir, self.tile_list[idx])
        data = np.loadtxt(path, dtype=np.float32)  # (N, 10)

        points = data[:, :6]                        # x,y,Z,R,G,B
        labels = data[:, 6].astype(np.int64)
        shadow      = data[:, 7]
        overexposed = data[:, 8]
        confidence  = data[:, 9]

        # Per-Punkt-Gewicht berechnen (vektorisiert, kein Python-Loop)
        w = confidence / CONFIDENCE_MAX                              # [0.2, 1.0]
        w = np.where(shadow      > 0.5, w * SHADOW_WEIGHT,      w)  # shadow-Punkte abwerten
        w = np.where(overexposed > 0.5, w * OVEREXPOSED_WEIGHT, w)  # overexposed abwerten
        w = np.clip(w, MIN_WEIGHT, 1.0).astype(np.float32)          # untere Grenze 0.05

        if self.augment:
            points = self._augment(points)

        return points, labels, w

    @staticmethod
    def _augment(points):
            angle = np.random.uniform(0, 2*np.pi)
            c, s = np.cos(angle), np.sin(angle)
            rot = np.array([[c,-s],[s,c]], dtype=np.float32)
            points[:, :2] = points[:, :2] @ rot.T
            points[:, :3] += np.random.normal(0,0.01,
                            size=points[:, :3].shape).astype(np.float32)
            return points
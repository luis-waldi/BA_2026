import os
import numpy as np
from torch.utils.data import Dataset

NUM_CLASSES = 8
CLASS_NAMES = ['bush', 'deadwood', 'graminoid', 'heath',
               'other', 'sand', 'soil', 'tree']

# Per-Punkt-Qualitaetsgewicht aus shadow/overexposed/confidence
CONFIDENCE_MAX     = 10     # Skala 2-10, 9 fehlt in Daten – irrelevant
SHADOW_WEIGHT      = 0.3
OVEREXPOSED_WEIGHT = 0.3
MIN_WEIGHT         = 0.05


class HeideDatasetV2(Dataset):
    def __init__(self, tile_dir, tile_list, augment=False):
        self.tile_dir = tile_dir
        self.tile_list = tile_list
        self.augment = augment

        # Alle Kacheln einmal vorladen und im RAM halten, statt jede .txt bei
        # jedem Zugriff neu einzulesen.
        self.points_list = []   # je Kachel: (N, 6)  x,y,z,R,G,B
        self.labels_list = []   # je Kachel: (N,)    Klassenlabel 0..7
        self.weight_list = []   # je Kachel: (N,)    Per-Punkt-Gewicht

        for fname in tile_list:
            data = np.loadtxt(os.path.join(tile_dir, fname), dtype=np.float32)

            # Spaltenzahl bestimmt die Featureanzahl: 6 nur RGB oder 7 mit NIR.
            # Danach folgen label, shadow, overexposed, confidence.
            n_feat = 7 if data.shape[1] >= 11 else 6

            points = data[:, :n_feat].copy()            # x,y,z,R,G,B[,NIR]
            labels = data[:, n_feat].astype(np.int64)
            shadow      = data[:, n_feat + 1]
            overexposed = data[:, n_feat + 2]
            confidence  = data[:, n_feat + 3]

            # Per-Punkt-Gewicht, einmalig vorberechnet
            w = confidence / CONFIDENCE_MAX                              # [0.2, 1.0]
            w = np.where(shadow      > 0.5, w * SHADOW_WEIGHT,      w)   # Schatten abwerten
            w = np.where(overexposed > 0.5, w * OVEREXPOSED_WEIGHT, w)  # Ueberbelichtung abwerten
            w = np.clip(w, MIN_WEIGHT, 1.0).astype(np.float32)          # untere Grenze 0.05

            self.points_list.append(points)
            self.labels_list.append(labels)
            self.weight_list.append(w)

    def __len__(self):
        return len(self.tile_list)

    def __getitem__(self, idx):
        points = self.points_list[idx]
        labels = self.labels_list[idx]
        w      = self.weight_list[idx]

        # Augmentierung standardmaessig aus (laeuft im Trainingsskript).
        if self.augment:
            points = self._augment(points.copy())

        return points, labels, w

    @staticmethod
    def _augment(points):
        angle = np.random.uniform(0, 2 * np.pi)
        c, s = np.cos(angle), np.sin(angle)
        rot = np.array([[c, -s], [s, c]], dtype=np.float32)
        points[:, :2] = points[:, :2] @ rot.T
        points[:, :3] += np.random.normal(0, 0.01,
                          size=points[:, :3].shape).astype(np.float32)
        return points

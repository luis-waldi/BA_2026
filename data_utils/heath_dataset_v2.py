import os
import numpy as np
from torch.utils.data import Dataset

IGNORE_INDEX = 255

# Klassenschema ueber die Umgebungsvariable HEATH_SCHEMA umschaltbar, damit
# Trainings- und Auswertungsskripte unveraendert bleiben:
#   HEATH_SCHEMA=taxo8   8 Klassen, Originalschema der Baseline (run01-run03)
#   HEATH_SCHEMA=taxo6   6 Klassen, sand+soil zu ground, other ignoriert
#   HEATH_SCHEMA=binary  2 Klassen, Gehoelz (bush+tree) gegen alles andere
# Rohlabels: 0 bush, 1 deadwood, 2 graminoid, 3 heath, 4 other, 5 sand,
# 6 soil, 7 tree.
_SCHEMAS = {
    # bush, deadwood, graminoid, heath, other, sand, soil, tree
    'taxo8': (['bush', 'deadwood', 'graminoid', 'heath', 'other', 'sand',
               'soil', 'tree'],
              [0, 1, 2, 3, 4, 5, 6, 7]),
    'taxo6': (['bush', 'deadwood', 'graminoid', 'heath', 'ground', 'tree'],
              [0, 1, 2, 3, IGNORE_INDEX, 4, 4, 5]),
    'binary': (['rest', 'gehoelz'],
               [1, 0, 0, 0, IGNORE_INDEX, 0, 0, 1]),
}

SCHEMA = os.environ.get('HEATH_SCHEMA', 'taxo6')
if SCHEMA not in _SCHEMAS:
    raise ValueError(f'Unbekanntes HEATH_SCHEMA "{SCHEMA}", erlaubt: {list(_SCHEMAS)}')

CLASS_NAMES = _SCHEMAS[SCHEMA][0]
NUM_CLASSES = len(CLASS_NAMES)
_LABEL_MAP = np.array(_SCHEMAS[SCHEMA][1], dtype=np.int64)

# Per-Punkt-Qualitaetsgewicht aus shadow/overexposed/confidence
CONFIDENCE_MAX     = 10     
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
        self.points_list = []   
        self.labels_list = []   
        self.weight_list = []   

        for fname in tile_list:
            data = np.loadtxt(os.path.join(tile_dir, fname), dtype=np.float32)

            # Die letzten vier Spalten sind immer label, shadow, overexposed
            # und confidence. Alles davor sind Merkmale. Damit funktionieren
            # 6 (RGB), 7 (RGB+NIR) und 9 (RGB+NIR+z_rel+z_range) ohne Aenderung.
            n_feat = data.shape[1] - 4

            points = data[:, :n_feat].copy()            
            labels = _LABEL_MAP[data[:, n_feat].astype(np.int64)]  
            shadow      = data[:, n_feat + 1]
            overexposed = data[:, n_feat + 2]
            confidence  = data[:, n_feat + 3]

            # Per-Punkt-Gewicht, einmalig vorberechnet
            w = confidence / CONFIDENCE_MAX                              
            w = np.where(shadow      > 0.5, w * SHADOW_WEIGHT,      w)   # Schatten abwerten
            w = np.where(overexposed > 0.5, w * OVEREXPOSED_WEIGHT, w)  # Ueberbelichtung abwerten
            w = np.clip(w, MIN_WEIGHT, 1.0).astype(np.float32)          

            self.points_list.append(points)
            self.labels_list.append(labels)
            self.weight_list.append(w)
            self.n_feat = n_feat 

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

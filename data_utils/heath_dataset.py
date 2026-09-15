import os
import numpy as np
from torch.utils.data import Dataset

# Klassen: 0=Busch, 1=Totholz, 2=Gras, 3=Heide, 4=Other, 5=Sand, 6=Soil, 7=Baum
NUM_CLASSES = 8
CLASS_NAMES = ['bush', 'deadwood', 'graminoid', 'heath', 'other', 'sand', 'soil', 'tree']

class HeideDataset(Dataset):
    """
    Liest vorberechnete 10x10m-Kacheln aus .txt-Dateien.
    Spalten: x_centered | y_centered | Z | R | G | B | label
    """
    def __init__(self, tile_dir, tile_list, augment=False):
        self.tile_dir = tile_dir
        self.tile_list = tile_list  # Liste von Dateinamen, z.B. ['tile_0001.txt', ...]
        self.augment = augment

    def __len__(self):
        return len(self.tile_list)

    def __getitem__(self, idx):
        # 1. Datei laden
        path = os.path.join(self.tile_dir, self.tile_list[idx])
        data = np.loadtxt(path, dtype=np.float32)
        
        # 2. Features (X,Y,Z, R,G,B) und Labels trennen
        points = data[:, :6] 
        labels = data[:, 6].astype(np.int64)
        
        # 3. Data Augmentation (nur beim Training)
        if self.augment:
            points = self._augment(points)
            
        return points, labels

    @staticmethod
    def _augment(points):
        # Zufällige Rotation NUR um die Z-Achse (Bäume wachsen immer nach oben!)
        angle = np.random.uniform(0, 2 * np.pi)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rot_matrix = np.array([[cos_a, -sin_a], 
                               [sin_a,  cos_a]], dtype=np.float32)
        
        # Wende Rotation auf X und Y an
        points[:, :2] = np.dot(points[:, :2], rot_matrix.T)
        
        # Gauss-Jitter (Minimales Rauschen hinzufügen, ca. 1cm)
        points[:, :3] += np.random.normal(0, 0.01, size=points[:, :3].shape).astype(np.float32)
        
        return points

# --- Kleiner Test, ob alles klappt ---
if __name__ == '__main__':
    import json
    # Pfad anpassen, falls er bei dir anders heißt!
    TILE_DIR = os.path.join(DATA_ROOT, 'processed', 'training_tiles_v3')
    
    with open(os.path.join(TILE_DIR, 'splits.json')) as f:
        splits = json.load(f)
        
    dataset = HeideDataset(TILE_DIR, splits['train'], augment=True)
    points, labels = dataset[0]
    
    print(f"Erfolgreich geladen!")
    print(f"Punkte-Shape: {points.shape} (Sollte 4096, 6 sein)")
    print(f"Labels-Shape: {labels.shape} (Sollte 4096, sein)")
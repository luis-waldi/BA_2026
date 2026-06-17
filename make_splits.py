"""
make_splits.py
Erzeugt eine splits.json mit train/val/test-Listen (70/15/15).
Räumliche Trennung: Kacheln werden nach Index sortiert, dann in Blöcke aufgeteilt.
"""
import os
import json
import math

TILE_DIR = "/Users/luis/Documents/BA/training_tiles_v2"

# Alle vorhandenen Kacheln einlesen und sortieren
all_tiles = sorted([f for f in os.listdir(TILE_DIR) if f.endswith(".txt")])
n = len(all_tiles)
print(f"Gesamt: {n} Kacheln im Verzeichnis gefunden.")

if n == 0:
    print("WARNUNG: Keine .txt-Kacheln gefunden! Prüfe deinen TILE_DIR Pfad.")
else:
    # Grenzen für 70% Train, 15% Val, 15% Test berechnen
    n_train = math.floor(n * 0.70)
    n_val = math.floor(n * 0.15)
    
    train_tiles = all_tiles[:n_train]
    val_tiles = all_tiles[n_train : n_train + n_val]
    test_tiles = all_tiles[n_train + n_val :]

    print(f"Train: {len(train_tiles)} | Val: {len(val_tiles)} | Test: {len(test_tiles)}")

    splits = {
        "train": train_tiles,
        "val": val_tiles,
        "test": test_tiles
    }

    out_path = os.path.join(TILE_DIR, "splits.json")
    with open(out_path, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"Erfolgreich gespeichert: {out_path}")
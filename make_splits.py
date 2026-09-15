# Erzeugt splits.json (train/val/test).
# Split je raeumlich getrenntem Bereich: alle Patches eines Bereichs landen komplett in einem Split, damit ueberlappende
# Patches nicht ueber die Split-Grenze reichen.
import os
import json
import argparse
import collections

parser = argparse.ArgumentParser()
parser.add_argument("--tile_dir", default=os.path.join(DATA_ROOT, 'processed', 'training_tiles_v5'))
TILE_DIR = parser.parse_args().tile_dir

all_tiles = sorted(f for f in os.listdir(TILE_DIR)
                   if f.endswith(".txt") and not f.startswith("."))
n = len(all_tiles)
print(f"Gesamt: {n} Kacheln gefunden.")

if n == 0:
    print("WARNUNG: keine .txt-Kacheln gefunden, TILE_DIR pruefen.")
    raise SystemExit

# Patches nach Bereich gruppieren
regions = collections.defaultdict(list)
for f in all_tiles:
    regions[f.split("__")[0]].append(f)

sizes = {r: len(v) for r, v in regions.items()}
print("Bereiche (Patches):", sizes)

# Groesste Bereiche zuerst, jeweils dem Split mit dem groessten
# Rueckstand auf sein Ziel zuweisen. Ganze Bereiche bleiben zusammen.
targets = {"train": 0.70, "val": 0.15, "test": 0.15}
assigned = {"train": [], "val": [], "test": []}
counts = {"train": 0, "val": 0, "test": 0}

for region in sorted(regions, key=lambda r: -sizes[r]):
    deficits = {s: targets[s] * n - counts[s] for s in targets}
    pick = max(deficits, key=deficits.get)
    assigned[pick].append(region)
    counts[pick] += sizes[region]

splits = {s: sorted(f for r in assigned[s] for f in regions[r]) for s in assigned}

for s in ("train", "val", "test"):
    pct = 100 * counts[s] / n
    print(f"{s}: {counts[s]} Patches ({pct:.0f}%) aus Bereichen {assigned[s]}")

out_path = os.path.join(TILE_DIR, "splits.json")
with open(out_path, "w") as f:
    json.dump(splits, f, indent=2)
print(f"Erfolgreich gespeichert: {out_path}")

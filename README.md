# Verbuschungsgrad von Heideflächen aus SfM-Punktwolken

Code zur Bachelorarbeit *Quantifizierung von Verbuschung in
Natura-2000-Heideflächen: Ein Vergleich von 2D- und 3D-Deep-Learning-Verfahren
zur drohnengestützten Habitatbewertung* (Universität Münster, Institut für
Geoinformatik).

Vollständige Verarbeitungskette von der photogrammetrischen Rohpunktwolke bis
zum flächendeckenden Verbuschungsgrad: Vorverarbeitung und Label-Transfer in R,
Training und Inferenz von PointNet++ in PyTorch, Auswertung gegen eine
Biotoptypenkartierung.

## Herkunft des Codes

Das PointNet++-Grundgerüst stammt aus
[yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)
und steht unter der MIT-Lizenz (siehe `LICENSE`). Es wurde unverändert als
eigener Commit übernommen, bevor die erste eigene Codezeile entstand.

Übernommen: `models/`, `provider.py`, `visualizer/`, Teile von `data_utils/`
sowie die Skripte für ModelNet, ShapeNet und S3DIS (`train_classification.py`,
`train_semseg.py`, `train_partseg.py` und die zugehörigen Testskripte). Diese
Dateien werden in der Arbeit nicht verwendet und bleiben nur zur
Nachvollziehbarkeit der Herkunft erhalten.

Eigene Entwicklung: die R-Vorverarbeitung, `data_utils/heath_dataset_v2.py`, die
Trainings-, Auswertungs- und Inferenzskripte für den Heidedatensatz sowie die
beiden Auswertungsskripte zum Verbuschungsgrad.

## Verarbeitungskette

| Schritt | Skript | Ergebnis |
| --- | --- | --- |
| 1. Trainingsdaten aufbereiten | `preprocessing_R/LabelMapping.R` | Kacheln als Textdateien |
| 2. Datenaufteilung | `make_splits.py` | `splits.json` |
| 3. Training | `train_heath_v4.py` | Modellgewichte in `log/sem_seg/<lauf>/` |
| 4. Auswertung auf dem Testteil | `eval_confusion.py` | Konfusionsmatrix und Kennzahlen |
| 5. Inferenzgebiete aufbereiten | `preprocessing_R/InferenzPreprocessing.R` | normalisierte LAZ-Blöcke |
| 6. Inferenz | `predict_heath.py` | Vorhersage je Punkt |
| 7a. Auswertung je Polygon | `verbuschungsgrad.py` | Vergleich mit der Kartierung |
| 7b. Flächendeckende Karte | `verbuschungskarte.py` | GeoTIFF des Verbuschungsgrades |

`preprocessing_R/hoehen_analyse.R` leitet die Höhenschwellen der Labelkorrektur
aus der Höhenverteilung je Klasse ab, Vorstufe zu Schritt 1.

### 1. Vorverarbeitung und Label-Transfer

`LabelMapping.R` erzeugt aus Rohpunktwolken und annotierten Polygonen die
Trainingskacheln: Bodenklassifikation, Höhennormalisierung, Label-Transfer,
höhenbasierte Labelkorrektur, lokale Höhenmerkmale, Kachelung.

Jede Kachel ist eine Textdatei mit 8.192 Zeilen und 13 Spalten:

```
x  y  z  R  G  B  NIR  z_rel  z_range  label  shadow  overexposed  confidence
```

Der Dataloader leitet die Zahl der Eingabekanäle als Spaltenzahl minus vier ab.

### 2. Datenaufteilung

```bash
python make_splits.py --tile_dir <pfad/zu/den/kacheln>
```

Verteilt räumlich getrennte Blöcke im Verhältnis 70/15/15.

### 3. Training

```bash
HEATH_SCHEMA=taxo6 HEATH_RADII=wide \
python train_heath_v4.py --tile_dir <pfad> --log_dir run07_radien
```

Zwei Umgebungsvariablen steuern die Varianten der Ablationsstudie:

| Variable | Werte | Bedeutung |
| --- | --- | --- |
| `HEATH_SCHEMA` | `taxo8`, `taxo6`, `binary` | Klassenschema mit 8, 6 oder 2 Klassen |
| `HEATH_RADII` | `default`, `wide` | Nachbarschaftsradien (0,1/0,2/0,4/0,8 m oder 0,25/0,5/1,0/2,0 m) |

Weitere Parameter über die Kommandozeile, unter anderem `--batch_size`,
`--epoch`, `--learning_rate`, `--loss` (`weighted` oder `focal`) und `--model`
(`pointnet2_sem_seg` oder `pointnet2_sem_seg_msg`).

### 4. Auswertung

```bash
HEATH_SCHEMA=taxo6 HEATH_RADII=wide \
python eval_confusion.py --tile_dir <pfad> \
    --model_path log/sem_seg/run07_radien/checkpoints/best_model.pth
```

Liefert Konfusionsmatrix, IoU, Precision, Recall und F1 je Klasse,
Gesamtgenauigkeit, mittlere IoU und die Zusammenfassung Gehölz gegen Rest.
Umgebungsvariablen und Kanalzahl müssen zum Training passen, sonst lässt sich
der Checkpoint nicht laden.

### 5. Inferenzgebiete aufbereiten

`InferenzPreprocessing.R` wendet dieselben Schritte auf ungelabelte
Befliegungsgebiete an, blockweise wegen der Datenmenge. Ergebnis sind
georeferenzierte LAZ-Blöcke mit den beiden Höhenmerkmalen als Punktattributen.

### 6. Inferenz

```bash
HEATH_SCHEMA=taxo6 HEATH_RADII=wide \
python predict_heath.py \
    --block_dir <pfad/zu/den/blöcken> \
    --model_path log/sem_seg/run07_radien/checkpoints/best_model.pth \
    --out_dir <ausgabepfad>
```

Mittelt die Klassenwahrscheinlichkeiten über alle überlappenden Fenster und lädt
jeden Block mit einem Saum aus den Nachbarblöcken. Mit `--write_laz` entsteht
zusätzlich eine klassifizierte Punktwolke.

### 7. Verbuschungsgrad

Vergleich mit der Kartierung:

```bash
python verbuschungsgrad.py \
    --pred_dir <vorhersagen> --shapefile <kartierung.shp> \
    --out <ausgabe> --max_hoehe 2 3 5 8
```

Flächendeckende Karte:

```bash
python verbuschungskarte.py \
    --pred_dir <vorhersagen> --out <ausgabe> --zelle 5 --max_hoehe 5
```

Beide projizieren die Punktvorhersagen in ein Raster von 0,25 m, in dem jede
Zelle Klasse und Höhe ihres höchsten Punktes übernimmt. Als verbuscht gilt eine
Zelle mit Klasse `bush` oder `tree` unterhalb der über `--max_hoehe` gesetzten
Grenze. `verbuschungskarte.py` schreibt zwei GeoTIFFs in EPSG 25832, den
Verbuschungsgrad je Zelle in Prozent und die häufigste Kronenklasse.

## Voraussetzungen

Python 3.9 mit PyTorch, NumPy, laspy, tqdm, shapely, pyshp, rasterio, openpyxl
und Matplotlib. Training auf einer CUDA-fähigen Grafikkarte, zusätzlich wird
Apple Silicon über MPS unterstützt.

R mit `lidR`, `sf`, `terra` und `dplyr`.

## Daten

Punktwolken, Orthomosaike und Annotationen liegen außerhalb des Repositorys und
sind nicht Teil der Veröffentlichung. Die Rohdaten stammen aus einer Befliegung
der Deutschen Bundesstiftung Umwelt, die Referenzdaten aus der
Biotoptypenkartierung der DBU-Naturerbefläche Authausener Wald von 2014.
Trainingsartefakte sind über die `.gitignore` ausgeschlossen.

Alle Skripte erwarten die Daten unter dem Verzeichnis, das die Umgebungsvariable
`HEATH_DATA` benennt, mit Rückfall auf `./data`:

```
data/
├── raw/          Rohpunktwolken, DGM
├── labeled/      annotierte GPKG-Dateien
└── processed/    Zwischenergebnisse und Trainingskacheln
```

```bash
export HEATH_DATA=/pfad/zu/den/daten
```

## Lizenz

MIT, siehe `LICENSE`. Der Lizenztext und der Urheberrechtsvermerk von yanx27
bleiben erhalten.

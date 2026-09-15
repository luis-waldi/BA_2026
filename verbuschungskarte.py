"""Flaechendeckende Verbuschungskarte aus den Punktvorhersagen.

Erzeugt aus den von predict_heath.py geschriebenen Vorhersagen zwei
georeferenzierte Raster:

  <out>_verbuschung.tif   Verbuschungsgrad je Zelle in Prozent
  <out>_kronenklasse.tif  haeufigste Kronenklasse je Zelle

Das Verfahren ist dasselbe wie in verbuschungsgrad.py, nur ist die
Bezugsflaeche hier eine Rasterzelle statt eines Kartierungspolygons. In zwei
Stufen: ein feines Raster von 0,25 m uebernimmt je Zelle die Klasse und die
Hoehe des obersten Punktes, das ist die Projektion der Vegetation von oben.
Darueber liegt das Auswerteraster, dessen Zellen den Anteil der als Gehoelz
projizierten Feinzellen tragen.

Als Gehoelz zaehlt eine Feinzelle, wenn ihre Kronenklasse bush oder tree ist
und die Kronenhoehe unter der Grenze liegt. Ohne diese Grenze wuerde der
angrenzende Altbestand mitgezaehlt, dessen Kronen ueber die offene Flaeche
ragen. Verbuschung meint aufkommende Pioniergehoelze, nicht Wald.

Zellen mit zu wenigen belegten Feinzellen bleiben leer, damit Randbereiche
mit duenner Punktabdeckung keine Scheinwerte liefern.

Aufruf:
    python verbuschungskarte.py \
        --pred_dir $HEATH_DATA/processed/vorhersage_110kv \
        --pred_dir $HEATH_DATA/processed/vorhersage_heidewildnis \
        --out $HEATH_DATA/ergebnisse/karte_authausen
"""

import os
import glob
import argparse

import numpy as np

CLASS_NAMES = ['bush', 'deadwood', 'graminoid', 'heath', 'ground', 'tree']
BUSH, TREE = 0, 5
LEER = 255
EPSG = 25832   # ETRS89 / UTM Zone 32N, wie die Kartierung und die Wolken


def parse_args():
    p = argparse.ArgumentParser('Verbuschungskarte')
    p.add_argument('--pred_dir', action='append', required=True)
    p.add_argument('--out', type=str, required=True, help='Pfad ohne Endung')
    p.add_argument('--fein', type=float, default=0.25,
                   help='Rasterweite der Projektion von oben in m')
    p.add_argument('--zelle', type=float, default=5.0,
                   help='Kantenlaenge der Auswertezelle in m')
    p.add_argument('--max_hoehe', type=float, default=5.0,
                   help='Kronenhoehe, bis zu der Gehoelz als Verbuschung zaehlt')
    p.add_argument('--min_anteil', type=float, default=0.5,
                   help='Mindestanteil belegter Feinzellen je Auswertezelle')
    return p.parse_args()


def kronenraster(dateien, fein):
    lo = [np.inf, np.inf]
    hi = [-np.inf, -np.inf]
    for f in dateien:
        d = np.load(f)
        lo = [min(lo[0], d['x'].min()), min(lo[1], d['y'].min())]
        hi = [max(hi[0], d['x'].max()), max(hi[1], d['y'].max())]
    gx0 = np.floor(lo[0] / fein) * fein
    gy0 = np.floor(lo[1] / fein) * fein
    nx = int(np.ceil((hi[0] - gx0) / fein)) + 1
    ny = int(np.ceil((hi[1] - gy0) / fein)) + 1
    print(f'Feinraster {nx} x {ny} Zellen a {fein} m')

    best_z = np.full(nx * ny, -np.inf, dtype=np.float32)
    best_c = np.full(nx * ny, LEER, dtype=np.uint8)
    for f in dateien:
        d = np.load(f)
        x, y, z, p = d['x'], d['y'], d['z'], d['pred']
        m = p != LEER
        if not m.any():
            continue
        x, y, z, p = x[m], y[m], z[m], p[m]
        cid = np.floor((x - gx0) / fein).astype(np.int64) * ny \
            + np.floor((y - gy0) / fein).astype(np.int64)
        o = np.lexsort((-z, cid))
        cid, z, p = cid[o], z[o], p[o]
        erste = np.r_[True, cid[1:] != cid[:-1]]
        cid, z, p = cid[erste], z[erste], p[erste]
        neu = z > best_z[cid]
        best_z[cid[neu]] = z[neu]
        best_c[cid[neu]] = p[neu]
    return best_c, best_z, gx0, gy0, nx, ny


def schreibe_raster(pfad, arr, gx0, gy0, zelle, nodata):
    """GeoTIFF ueber rasterio, ersatzweise ESRI-ASCII-Grid.

    Das Ausgabearray liegt in Bildkoordinaten, die erste Zeile ist also der
    Nordrand. Deshalb wird die Y-Achse gedreht.
    """
    bild = np.flipud(arr.T)
    try:
        import rasterio
        from rasterio.transform import from_origin
        tr = from_origin(gx0, gy0 + arr.shape[1] * zelle, zelle, zelle)
        with rasterio.open(
                pfad + '.tif', 'w', driver='GTiff',
                height=bild.shape[0], width=bild.shape[1], count=1,
                dtype=bild.dtype, crs=f'EPSG:{EPSG}', transform=tr,
                nodata=nodata, compress='deflate') as dst:
            dst.write(bild, 1)
        return pfad + '.tif'
    except ImportError:
        with open(pfad + '.asc', 'w') as fh:
            fh.write(f'ncols {bild.shape[1]}\nnrows {bild.shape[0]}\n'
                     f'xllcorner {gx0}\nyllcorner {gy0}\n'
                     f'cellsize {zelle}\nNODATA_value {nodata}\n')
            np.savetxt(fh, bild, fmt='%.2f')
        with open(pfad + '.prj', 'w') as fh:
            fh.write(f'EPSG:{EPSG}')
        return pfad + '.asc'


def main():
    args = parse_args()
    dateien = sorted(f for d in args.pred_dir
                     for f in glob.glob(os.path.join(d, '*_pred.npz')))
    if not dateien:
        raise SystemExit('Keine *_pred.npz gefunden')
    print(f'Vorhersagedateien: {len(dateien)}')

    best_c, best_z, gx0, gy0, nx, ny = kronenraster(dateien, args.fein)
    belegt = best_c != LEER
    print(f'belegte Feinzellen: {belegt.sum():,} '
          f'entspricht {belegt.sum() * args.fein ** 2 / 1e4:.1f} ha')

    # Feinzellen den Auswertezellen zuordnen
    k = int(round(args.zelle / args.fein))
    if abs(k * args.fein - args.zelle) > 1e-9:
        raise SystemExit('Die Auswertezelle muss ein Vielfaches der Feinzelle sein')
    mx, my = nx // k, ny // k
    print(f'Auswerteraster {mx} x {my} Zellen a {args.zelle} m')

    c = best_c[:mx * k * ny].reshape(mx * k, ny)[:, :my * k]
    z = best_z[:mx * k * ny].reshape(mx * k, ny)[:, :my * k]
    c = c.reshape(mx, k, my, k)
    z = z.reshape(mx, k, my, k)

    ist_belegt = (c != LEER).sum(axis=(1, 3))
    ist_gehoelz = (((c == BUSH) | (c == TREE)) & (z < args.max_hoehe)).sum(axis=(1, 3))

    genug = ist_belegt >= args.min_anteil * k * k
    verb = np.full((mx, my), np.nan, dtype=np.float32)
    verb[genug] = 100 * ist_gehoelz[genug] / ist_belegt[genug]

    # Haeufigste Kronenklasse je Auswertezelle
    haeufig = np.full((mx, my), LEER, dtype=np.uint8)
    zaehler = np.zeros((mx, my, len(CLASS_NAMES)), dtype=np.int32)
    for kl in range(len(CLASS_NAMES)):
        zaehler[:, :, kl] = (c == kl).sum(axis=(1, 3))
    hat = zaehler.sum(axis=2) > 0
    haeufig[hat] = zaehler.argmax(axis=2)[hat].astype(np.uint8)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    p1 = schreibe_raster(args.out + '_verbuschung', verb, gx0, gy0,
                         args.zelle, nodata=-9999)
    p2 = schreibe_raster(args.out + '_kronenklasse', haeufig, gx0, gy0,
                         args.zelle, nodata=LEER)

    gueltig = verb[np.isfinite(verb)]
    print(f'\nZellen mit Wert: {gueltig.size:,} '
          f'({gueltig.size * args.zelle ** 2 / 1e4:.1f} ha)')
    print(f'Verbuschungsgrad Median {np.median(gueltig):.1f} %, '
          f'Mittel {gueltig.mean():.1f} %, '
          f'Anteil ueber 30 %: {100 * np.mean(gueltig > 30):.1f} %')
    for kl, name in enumerate(CLASS_NAMES):
        anteil = 100 * np.mean(haeufig[haeufig != LEER] == kl)
        print(f'  {name:10s} als haeufigste Klasse in {anteil:5.1f} % der Zellen')
    print(f'\nGeschrieben: {p1}\nGeschrieben: {p2}')


if __name__ == '__main__':
    main()

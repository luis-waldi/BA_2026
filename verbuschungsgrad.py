"""Verbuschungsgrad je Kartierungspolygon aus den Punktvorhersagen.

Nimmt die von predict_heath.py erzeugten Vorhersagen, projiziert sie von oben
in ein feines Raster und berechnet fuer jedes Polygon der Biotopkartierung den
Gehoelzanteil. Dieser wird dem kartierten Feld VERBUP gegenuebergestellt.

Warum die Projektion von oben und nicht der Punktanteil: die Punktdichte einer
SfM-Wolke ist nicht gleichmaessig. Ein Busch mit Aesten und Textur liefert je
Quadratmeter ein Vielfaches der Punkte von flachem Sandboden, und unter
Gehoelzen fehlen Punkte durch Verdeckung. Ein reiner Punktanteil wuerde die
Verbuschung deshalb systematisch ueberschaetzen. Das Raster rechnet "wie viele
Punkte" in "wie viel Flaeche" um, und Flaeche ist die Groesse, nach der die
Definition des Verbuschungsgrades fragt.

Je Rasterzelle wird die Klasse des hoechsten Punktes uebernommen. Was von oben
sichtbar ist, bedeckt den Boden. Ein Heidepunkt unter einem Buschzweig zaehlt
nicht als offene Heide, denn er wird gerade beschattet.

Zwei Besonderheiten der Referenzdaten werden mitgefuehrt:

Die Kartierung stammt von 2014, die Befliegung von 2025. Polygone, deren
Bemerkungsfeld eine kurz zuvor erfolgte Pflegemassnahme nennt, also
Freistellung, Gehoelzentnahme, Mulchen oder Abschieben, sind mit hoher
Wahrscheinlichkeit inzwischen wieder aufgewachsen. Ihr Referenzwert von null
war 2014 korrekt, gilt aber fuer 2025 nicht mehr. Diese Polygone werden
gekennzeichnet und die Kennzahlen zusaetzlich ohne sie berechnet.

Aus der Begleittabelle wird ausserdem uebernommen, ob Verbuschung als
Beeintraechtigung vermerkt wurde (Code 17.1.3).

Aufruf:
    python verbuschungsgrad.py \
        --pred_dir $HEATH_DATA/vorhersage_110kv \
        --shapefile ../authausen_auswertung/Attributtabelle_AuthausenerWald_2014 \
        --xlsx ../authausen_auswertung/ergaenzendeAttribute_AuthausenerWald_2014.xlsx \
        --out ergebnisse/verbuschung_authausen
"""

import os
import re
import csv
import glob
import argparse

import numpy as np
import shapefile
import shapely
import openpyxl
from shapely.geometry import shape

# Klassenschema von run07
CLASS_NAMES = ['bush', 'deadwood', 'graminoid', 'heath', 'ground', 'tree']
BUSH, TREE = 0, 5
LEER = 255

# Biotop-Hauptgruppen, in denen Verbuschung ueberhaupt vorkommen kann.
# Waelder, Gewaesser und Rohbodenbiotope tragen in der Kartierung
# durchgehend VERBUP = 0, dort ist die Groesse nicht definiert.
OFFENLAND = ('Heiden und Magerrasen', 'Moore und Sümpfe',
             'Staudenfluren und Säume', 'Grünland')

# Hinweise auf eine kurz vor der Kartierung erfolgte Pflegemassnahme
PFLEGE = re.compile(
    r'freigestellt|geh.lzentnahme|geh.lzentfernung|gemulcht|gefr.st|'
    r'entkusselt|abgeschoben|ber.umt', re.IGNORECASE)


def parse_args():
    p = argparse.ArgumentParser('Verbuschungsgrad je Kartierungspolygon')
    p.add_argument('--pred_dir', action='append', required=True,
                   help='Verzeichnis mit *_pred.npz, mehrfach angebbar')
    p.add_argument('--shapefile', type=str, required=True,
                   help='Pfad ohne Endung')
    p.add_argument('--xlsx', type=str, default=None,
                   help='Begleittabelle mit Biotopgruppe und Beeintraechtigungen')
    p.add_argument('--out', type=str, required=True, help='Ausgabepfad ohne Endung')
    p.add_argument('--cell', type=float, default=0.25, help='Rasterweite in m')
    p.add_argument('--min_abdeckung', type=float, default=80.0,
                   help='Mindestanteil des Polygons mit Punkten, in Prozent')
    p.add_argument('--min_zellen', type=int, default=200,
                   help='Mindestzahl belegter Zellen je Polygon')
    p.add_argument('--max_hoehe', type=float, nargs='+', default=[3, 5, 8, 12],
                   help='Hoehengrenzen in m fuer die Verbuschungsdefinition. '
                        'Eine Zelle zaehlt als verbuscht, wenn ihre Kronenklasse '
                        'bush oder tree ist und die Kronenhoehe darunter liegt.')
    return p.parse_args()


def kronenraster(dateien, cell):
    """Je Rasterzelle die Klasse des hoechsten vorhergesagten Punktes.

    Die Dateien werden nacheinander gelesen und in ein gemeinsames Raster
    eingetragen, damit nie alle Punkte gleichzeitig im Speicher liegen.
    """
    lo = [np.inf, np.inf]
    hi = [-np.inf, -np.inf]
    for f in dateien:
        d = np.load(f)
        lo = [min(lo[0], d['x'].min()), min(lo[1], d['y'].min())]
        hi = [max(hi[0], d['x'].max()), max(hi[1], d['y'].max())]

    gx0 = np.floor(lo[0] / cell) * cell
    gy0 = np.floor(lo[1] / cell) * cell
    nx = int(np.ceil((hi[0] - gx0) / cell)) + 1
    ny = int(np.ceil((hi[1] - gy0) / cell)) + 1
    print(f'Raster {nx} x {ny} Zellen a {cell} m ({nx * ny / 1e6:.1f} Mio)')

    best_z = np.full(nx * ny, -np.inf, dtype=np.float32)
    best_c = np.full(nx * ny, LEER, dtype=np.uint8)

    for f in dateien:
        d = np.load(f)
        x, y, z, p = d['x'], d['y'], d['z'], d['pred']
        m = p != LEER
        if not m.any():
            continue
        x, y, z, p = x[m], y[m], z[m], p[m]
        cid = np.floor((x - gx0) / cell).astype(np.int64) * ny \
            + np.floor((y - gy0) / cell).astype(np.int64)
        # Je Zelle den hoechsten Punkt zuerst, dann den ersten je Zelle nehmen
        o = np.lexsort((-z, cid))
        cid, z, p = cid[o], z[o], p[o]
        erste = np.r_[True, cid[1:] != cid[:-1]]
        cid, z, p = cid[erste], z[erste], p[erste]
        neu = z > best_z[cid]
        best_z[cid[neu]] = z[neu]
        best_c[cid[neu]] = p[neu]

    belegt = int((best_c != LEER).sum())
    print(f'belegte Zellen: {belegt:,} entspricht {belegt * cell ** 2 / 1e4:.1f} ha')
    return best_c, best_z, gx0, gy0, nx, ny


def lies_referenz(shp_pfad, xlsx_pfad):
    """Polygone mit VERBUP, Biotopgruppe, Bemerkung und Verbuschungsvermerk."""
    sf = shapefile.Reader(shp_pfad, encoding='utf-8', encodingErrors='replace')
    feld = [f[0] for f in sf.fields[1:]]
    idx = {k: feld.index(k) for k in
           ('BKFLID', 'VERBUP', 'HC_FFH', 'LBTNAME', 'BEMERK', 'HC_EZG_B')}

    gruppe, vermerk = {}, set()
    if xlsx_pfad:
        wb = openpyxl.load_workbook(xlsx_pfad, read_only=True, data_only=True)
        for r in list(wb.worksheets[0].iter_rows(values_only=True))[1:]:
            gruppe[r[0]] = r[1] or ''
            texte = [str(r[k]) for k in (3, 6, 9, 12, 15) if k < len(r) and r[k]]
            if any('17.1.3' in t for t in texte):
                vermerk.add(r[0])

    polygone = []
    for sr in sf.iterShapeRecords():
        g = shape(sr.shape.__geo_interface__)
        r = sr.record
        if not g.area:
            continue
        bk = r[idx['BKFLID']]
        grp = gruppe.get(bk, '')
        if gruppe and grp not in OFFENLAND:
            continue
        bem = str(r[idx['BEMERK']] or '').replace('\r', ' ').replace('\n', ' ')
        polygone.append(dict(
            id=bk, geom=g, flaeche=g.area,
            verbup=float(r[idx['VERBUP']] or 0),
            ffh=str(r[idx['HC_FFH']] or ''),
            typ=str(r[idx['LBTNAME']] or ''),
            ez=str(r[idx['HC_EZG_B']] or ''),
            gruppe=grp,
            bemerkung=bem,
            gepflegt_2014=bool(PFLEGE.search(bem)),
            verbuschung_vermerkt=bk in vermerk))
    return polygone


def spaltenname(h):
    return f'verb_h{h:g}_pct'


def auswerten(polygone, best_c, best_z, gx0, gy0, nx, ny, cell,
              min_abd, min_zellen, hoehen):
    ergebnis = []
    for p in polygone:
        x0, y0, x1, y1 = p['geom'].bounds
        cx0 = max(int(np.floor((x0 - gx0) / cell)), 0)
        cy0 = max(int(np.floor((y0 - gy0) / cell)), 0)
        cx1 = min(int(np.ceil((x1 - gx0) / cell)), nx - 1)
        cy1 = min(int(np.ceil((y1 - gy0) / cell)), ny - 1)
        if cx1 < cx0 or cy1 < cy0:
            continue

        ix, iy = np.meshgrid(np.arange(cx0, cx1 + 1),
                             np.arange(cy0, cy1 + 1), indexing='ij')
        ix, iy = ix.ravel(), iy.ravel()
        zellid = ix * ny + iy
        cls = best_c[zellid]
        belegt = cls != LEER
        if belegt.sum() < min_zellen:
            continue

        # Nur Zellen, deren Mittelpunkt im Polygon liegt
        xs = gx0 + (ix[belegt] + 0.5) * cell
        ys = gy0 + (iy[belegt] + 0.5) * cell
        drin = shapely.contains_xy(p['geom'], xs, ys)
        cls = cls[belegt][drin]
        hoehe = best_z[zellid[belegt][drin]]
        if len(cls) < min_zellen:
            continue

        abdeckung = 100 * len(cls) * cell ** 2 / p['flaeche']
        if abdeckung < min_abd:
            continue

        anteil = lambda k: 100 * float((cls == k).mean())
        gehoelz = np.isin(cls, (BUSH, TREE))
        zeile = dict(
            BKFLID=p['id'], gruppe=p['gruppe'], ffh=p['ffh'], typ=p['typ'],
            ez_gesamt=p['ez'], flaeche_m2=round(p['flaeche'], 1),
            zellen=len(cls), abdeckung_pct=round(abdeckung, 1),
            verbup_ref=p['verbup'],
            gehoelz_pct=round(anteil(BUSH) + anteil(TREE), 2),
            bush_pct=round(anteil(BUSH), 2),
            tree_pct=round(anteil(TREE), 2))
        # Hoehenbegrenzte Verbuschung. Aufkommende Gehoelze sind niedrig,
        # der angrenzende Altbestand ragt nur mit der Krone ueber die
        # Polygongrenze und wird durch die Grenze ausgeschlossen.
        for h in hoehen:
            zeile[spaltenname(h)] = round(
                100 * float((gehoelz & (hoehe < h)).mean()), 2)
        # Mittlere Kronenhoehe der Gehoelzzellen, hilft bei der Deutung
        zeile['gehoelz_h_median'] = (round(float(np.median(hoehe[gehoelz])), 2)
                                     if gehoelz.any() else 0.0)
        zeile.update(
            gepflegt_2014=int(p['gepflegt_2014']),
            verbuschung_vermerkt=int(p['verbuschung_vermerkt']),
            bemerkung=p['bemerkung'][:250])
        ergebnis.append(zeile)
    return ergebnis


def kennzahlen(zeilen, spalte):
    """RMSE, Bias, Bestimmtheitsmass und Korrelation gegen VERBUP.

    RMSE ist die mittlere Fehlergroesse in Prozentpunkten, Bias der
    systematische Versatz mit Vorzeichen, das Bestimmtheitsmass der Anteil der
    erklaerten Streuung. Dazu zwei Korrelationen: die gewoehnliche nach Pearson
    misst den Gleichlauf der Werte, die Rangkorrelation nach Spearman nur den
    Gleichlauf der Reihenfolge. Pearson reagiert bei kleiner Stichprobe stark
    auf einzelne Extremwerte, Spearman nicht. Weichen beide deutlich
    voneinander ab, traegt ein einzelnes Polygon das Ergebnis.
    """
    y = np.array([r['verbup_ref'] for r in zeilen], dtype=float)
    yh = np.array([r[spalte] for r in zeilen], dtype=float)
    if len(y) < 3:
        return None
    e = yh - y
    ss = float(((y - y.mean()) ** 2).sum())

    def rang(v):
        o = np.argsort(np.argsort(v)).astype(float)
        # Bindungen mitteln, sonst verzerrt die Reihenfolge gleicher Werte
        for w in np.unique(v):
            m = v == w
            if m.sum() > 1:
                o[m] = o[m].mean()
        return o

    ok = y.std() > 0 and yh.std() > 0
    return dict(
        n=len(y), rmse=float(np.sqrt((e ** 2).mean())), bias=float(e.mean()),
        r2=1 - float((e ** 2).sum()) / ss if ss > 0 else float('nan'),
        korr=float(np.corrcoef(y, yh)[0, 1]) if ok else float('nan'),
        rang=float(np.corrcoef(rang(y), rang(yh))[0, 1]) if ok else float('nan'))


def main():
    args = parse_args()
    dateien = sorted(f for d in args.pred_dir
                     for f in glob.glob(os.path.join(d, '*_pred.npz')))
    if not dateien:
        raise SystemExit('Keine *_pred.npz gefunden')
    print(f'Vorhersagedateien: {len(dateien)}')

    best_c, best_z, gx0, gy0, nx, ny = kronenraster(dateien, args.cell)
    polygone = lies_referenz(args.shapefile, args.xlsx)
    print(f'Offenlandpolygone in der Kartierung: {len(polygone)}')

    zeilen = auswerten(polygone, best_c, best_z, gx0, gy0, nx, ny, args.cell,
                       args.min_abdeckung, args.min_zellen, args.max_hoehe)
    zeilen.sort(key=lambda r: -r['flaeche_m2'])
    print(f'ausgewertete Polygone: {len(zeilen)}')

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    csv_pfad = args.out + '_polygone.csv'
    with open(csv_pfad, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(zeilen[0].keys()))
        w.writeheader()
        w.writerows(zeilen)

    txt_pfad = args.out + '_kennzahlen.txt'
    with open(txt_pfad, 'w', encoding='utf-8') as fh:
        def sag(s=''):
            print(s)
            fh.write(s + '\n')

        sag(f'Rasterweite {args.cell} m, Mindestabdeckung {args.min_abdeckung} %')
        sag(f'Polygone: {len(zeilen)}')
        gepflegt = [r for r in zeilen if r['gepflegt_2014']]
        sag(f'davon 2014 als frisch gepflegt kartiert: {len(gepflegt)}')
        sag()

        gruppen = {}
        for r in zeilen:
            gruppen.setdefault(r['gruppe'], []).append(r)
        auswahl = list(gruppen.items())
        auswahl.append(('nur FFH 4030', [r for r in zeilen if r['ffh'] == '4030']))
        auswahl.append(('alle Offenlandbiotope', zeilen))
        auswahl.append(('alle, ohne 2014 gepflegte',
                        [r for r in zeilen if not r['gepflegt_2014']]))
        auswahl.append(('FFH 4030, ohne 2014 gepflegte',
                        [r for r in zeilen
                         if r['ffh'] == '4030' and not r['gepflegt_2014']]))

        masse = [('bush_pct', 'nur bush'), ('gehoelz_pct', 'bush+tree')]
        masse += [(spaltenname(h), f'bush+tree <{h:g}m') for h in args.max_hoehe]

        kopf = (f'{"Auswahl":32s} {"n":>3s} {"Mass":15s} {"RMSE":>6s} '
                f'{"Bias":>7s} {"R2":>8s} {"Pears":>6s} {"Spear":>6s}')
        sag(kopf)
        sag('-' * len(kopf))
        for name, sub in auswahl:
            for spalte, label in masse:
                k = kennzahlen(sub, spalte)
                if k:
                    sag(f'{name[:32]:32s} {k["n"]:3d} {label:15s} '
                        f'{k["rmse"]:6.1f} {k["bias"]:+7.1f} '
                        f'{k["r2"]:8.2f} {k["korr"]:6.2f} {k["rang"]:6.2f}')
            sag()

        hsp = [spaltenname(h) for h in args.max_hoehe]
        sag('Polygone einzeln, sortiert nach kartiertem Wert')
        sag(f'{"BKFLID":11s} {"Grp":10s} {"FFH":5s} {"VERBUP":>6s} '
            f'{"bush":>6s} {"tree":>6s} '
            + ' '.join(f'{"<" + f"{h:g}m":>6s}' for h in args.max_hoehe)
            + f' {"hMed":>5s} {"Pfl":>3s}  Typ')
        for r in sorted(zeilen, key=lambda r: -r['verbup_ref']):
            sag(f'{r["BKFLID"]:11s} {r["gruppe"][:10]:10s} {r["ffh"]:5s} '
                f'{r["verbup_ref"]:6.0f} {r["bush_pct"]:5.1f}% '
                f'{r["tree_pct"]:5.1f}% '
                + ' '.join(f'{r[s]:5.1f}%' for s in hsp)
                + f' {r["gehoelz_h_median"]:5.1f} '
                f'{"ja" if r["gepflegt_2014"] else "":>3s}  {r["typ"][:30]}')

    print(f'\nGeschrieben: {csv_pfad}')
    print(f'Geschrieben: {txt_pfad}')


if __name__ == '__main__':
    main()

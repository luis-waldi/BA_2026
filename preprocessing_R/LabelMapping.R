# LabelMapping.R: Preprocessing der Trainingsdaten
# Daten ausserhalb des Repos unter /Users/luis/Documents/BA/data/:
#   raw/       Rohdaten (Punktwolke .laz, DGM, tile .tif)
#   labeled/   gelabelte GPKGs
#   processed/ Outputs (rds, LAZ, training_tiles_v3)


library(sf)
library(terra)
library(lidR)

las <- readLAS("/Users/luis/Documents/BA/data/raw/Punktwolke_AuthausenerWald_Schneise_I_20250522.laz")
#las

las <- remove_lasattribute(las, "normal x")
las <- remove_lasattribute(las, "normal y")
las <- remove_lasattribute(las, "normal z")

writeLAS(las, "/Users/luis/Documents/BA/data/processed/Punktwolke_ohne_normals.laz")


las2 <- readLAS("/Users/luis/Documents/BA/data/processed/Punktwolke_ohne_normals.laz")
names(las2@data)

#Testbereich

las <- readLAS("/Users/luis/Documents/BA/data/raw/Punktwolke_AuthausenerWald_Schneise_I_20250522.laz")
polys <- st_read("/Users/luis/Documents/BA/data/labeled/Authausen_02_tile_3_segments_labeled_JS.gpkg")

#gleiches CRS?

st_crs(las)$epsg
st_crs(polys)$epsg

#Gucken ob Punkte grundsätzlich in den Polygonen liegen 
bb <- st_bbox(polys)

las_crop_tile <- clip_rectangle(
  las,
  xleft   = bb["xmin"],
  ybottom = bb["ymin"],
  xright  = bb["xmax"],
  ytop    = bb["ymax"]
)

d <- las_crop_tile@data

plot(d$X, d$Y, asp = 1, pch = 20, cex = 0.2)
plot(st_geometry(polys), border = "red", add = TRUE)


#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------

#Mapping der Polygonlables auf die Punkte
#vorher Punktwolke auf Bereich der tile1 .tif file zuschneiden

library(lidR)
library(sf)
library(dplyr)

dir_labeled <- "/Users/luis/Documents/BA/data/labeled"
dir_raw     <- "/Users/luis/Documents/BA/data/raw"

# Punktwolken je Gebiet
laz_schneise_I <- file.path(dir_raw, "Punktwolke_AuthausenerWald_Schneise_I_20250522.laz")
laz_110kv      <- file.path(dir_raw, "Punktwolke_AuthausenerWald_110kVTrassePSAPlatz_20250521.laz")
laz_cuxhaven   <- file.path(dir_raw, "Punktwolke_Cuxhavener_Küstenheiden_2025.laz")

# Tile -> zugehoerige Punktwolke
tiles <- list(
  list(gpkg = "Authausen_02_tile_1_segments_labeled_JS.gpkg",     laz = laz_schneise_I),
  list(gpkg = "Authausen_02_tile_2_segments_labeled_JS.gpkg",     laz = laz_schneise_I),
  list(gpkg = "Authausen_02_tile_3_segments_labeled_JS.gpkg",     laz = laz_schneise_I),
  list(gpkg = "Authausen_02_tile_4_segments_labeled_JS.gpkg",     laz = laz_schneise_I),
  list(gpkg = "Authausen_02_tile_5_segments_labeled_JS_neu.gpkg", laz = laz_schneise_I),
  list(gpkg = "Authausen_02_tile_6_segments_labeled_JS_neu.gpkg", laz = laz_schneise_I),
  list(gpkg = "Authausen_02_tile_7_segments_labeled_JS.gpkg",     laz = laz_schneise_I),
  list(gpkg = "Authausen_02_tile_8_segments_labeled_JS.gpkg",     laz = laz_schneise_I),
  list(gpkg = "Authausen_06_tile_75_segments_labeled_JS.gpkg",    laz = laz_110kv),
  list(gpkg = "Authausen_06_tile_190_segments_labeled_JS.gpkg",   laz = laz_110kv),
  list(gpkg = "Authausen_06_tile_226_segments_labeled_JS.gpkg",   laz = laz_110kv),
  list(gpkg = "Cuxhaven_01_tile_10_segments_labeled_JS.gpkg",     laz = laz_cuxhaven),
  list(gpkg = "Cuxhaven_01_tile_66_segments_labeled_JS.gpkg",     laz = laz_cuxhaven),
  list(gpkg = "Cuxhaven_01_tile_225_segments_labeled_JS.gpkg",    laz = laz_cuxhaven),
  list(gpkg = "Cuxhaven_01_tile_270_segments_labeled_JS.gpkg",    laz = laz_cuxhaven),
  list(gpkg = "Cuxhaven_01_tile_406_segments_labeled_JS.gpkg",    laz = laz_cuxhaven)
)

# Nur diese Spalten behalten. Verschiedene Wolken haben unterschiedliche
# Extra-Attribute (z.B. Normalen), sonst bricht rbind unten.
keep_cols <- c("X", "Y", "Z", "R", "G", "B",
               "class", "shadow", "overexposed", "confidence")

tile_list <- list()

for (i in seq_along(tiles)) {
  gpkg <- tiles[[i]]$gpkg
  laz  <- tiles[[i]]$laz
  cat("Verarbeite Tile", i, ":", gpkg, "\n")

  polys <- st_read(file.path(dir_labeled, gpkg), quiet = TRUE)
  polys <- st_make_valid(polys)
  bb <- st_bbox(polys)

  las_tile <- readLAS(laz, filter = paste(
    "-keep_xy", bb["xmin"], bb["ymin"], bb["xmax"], bb["ymax"]))

  # Hoehe pro Tile normalisieren. Global ueber die
  # weit auseinanderliegenden Gebiete waere die CSF-Flaeche riesig und
  # extrem langsam. Z ist danach Hoehe ueber Boden.
  las_tile <- classify_ground(las_tile, csf(cloth_resolution = 0.5,
                                            class_threshold  = 0.3,
                                            rigidness        = 2))
  las_tile <- normalize_height(las_tile, knnidw(k = 10, p = 2))
  las_tile <- filter_poi(las_tile, Z >= -1.0)   

  # 4 Attribute nacheinander mappen
  las_lab <- merge_spatial(las_tile, polys, "class")
  las_lab <- merge_spatial(las_lab,  polys, "shadow")
  las_lab <- merge_spatial(las_lab,  polys, "overexposed")
  las_lab <- merge_spatial(las_lab,  polys, "confidence")

  # Nur gelabelte Punkte. Leerer String "" extra abfangen.
  las_train <- filter_poi(las_lab, !is.na(class) & class != "")

  tile_list[[i]] <- las_train@data[, ..keep_cols]
}

# Alle Tiles zusammenfuehren
df_all <- do.call(rbind, tile_list)

# Klassen-ID per festem Woerterbuch, nicht as.factor.
# as.factor kodiert alphabetisch aus den vorhandenen Klassen, IDs verrutschen
# bei neuen oder fehlenden Klassen. Feste Liste haelt IDs stabil.
# water als id 9 reserviert, entsteht erst mit echten water-Punkten.
class_levels <- c("bush", "deadwood", "graminoid", "heath",
                  "other", "sand", "soil", "tree", "water")
df_all$class_id <- match(df_all$class, class_levels)

# Unbekannte Klassennamen werden zu NA.
if (any(is.na(df_all$class_id))) {
  bad <- unique(df_all$class[is.na(df_all$class_id)])
  stop("Unbekannte Klassen, nicht in class_levels: ", paste(bad, collapse = ", "))
}

cat("\nKlassen-Mapping:\n")
print(table(df_all$class, df_all$class_id))

# Verteilung der neuen Attribute pruefen
cat("\nShadow:\n");      print(table(df_all$shadow,      useNA = "ifany"))
cat("\nOverexposed:\n"); print(table(df_all$overexposed, useNA = "ifany"))
cat("\nConfidence:\n");  print(table(df_all$confidence,  useNA = "ifany"))

saveRDS(df_all, "/Users/luis/Documents/BA/data/processed/trainingspunkte_alle_tiles_v3.rds")


#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------

#Umwandeln der .rds file in eine LAZ File

library(lidR)

df_all <- readRDS("/Users/luis/Documents/BA/data/processed/trainingspunkte_alle_tiles_v3.rds")

# Flags in 0/1 umwandeln
shadow_vec  <- as.integer(df_all$shadow      %in% c(TRUE, "true", "True", "1", 1))
overexp_vec <- as.integer(df_all$overexposed %in% c(TRUE, "true", "True", "1", 1))
conf_vec    <- as.numeric(df_all$confidence)

# Standard-LAS bauen
df_std <- df_all
df_std$class       <- NULL
df_std$shadow      <- NULL
df_std$overexposed <- NULL
df_std$confidence  <- NULL

# class_id als Classification ins df schreiben, bevor das LAS gebaut wird.
# Bei einem fertigen LAS koennte Classification nicht mehr neu angelegt werden.
df_std$Classification <- as.integer(df_all$class_id)
df_std$class_id       <- NULL

las_all <- LAS(df_std)
st_crs(las_all) <- 25832

# Extra-Attribute registrieren
las_all <- add_lasattribute(las_all, shadow_vec,  "shadow",      "Schatten 0/1")
las_all <- add_lasattribute(las_all, overexp_vec, "overexposed", "Ueberbelichtet 0/1")
las_all <- add_lasattribute(las_all, conf_vec,    "confidence",  "Label-Confidence")

writeLAS(las_all, "/Users/luis/Documents/BA/data/processed/trainingspunkte_alle_tiles_classified_v3.laz")

# Attribute muessen erhalten sein
las_check <- readLAS("/Users/luis/Documents/BA/data/processed/trainingspunkte_alle_tiles_classified_v3.laz")
print(names(las_check@data))   # shadow, overexposed, confidence muessen dabei sein
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
#Höhen Schwellenwerte bestimmen um unrealistische Fehlklassifikationen zu korrigieren

library(lidR)

# Punktwolke laden. Z ist bereits pro Tile normalisiert, Classification enthaelt
# die semantische class_id.
las_norm <- readLAS("/Users/luis/Documents/BA/data/processed/trainingspunkte_alle_tiles_classified_v3.laz")

# --- Schwellenwert-Regeln ---
# Grenzen = 95. Perzentil der normierten Hoehe je Klasse (hoehen_analyse.R).
# bush per Oekologie (3.5 m), da p95 durch Baumueberhang verfaelscht.
# Aufwaerts zuerst, dann Abwaerts.

# AUFWÄRTS-KORREKTUREN (zu hoch fuer die Klasse -> tree), p95 aus 16 Tiles
las_norm$Classification[las_norm$Classification == 7 & las_norm$Z > 1.15] <- 8L  # soil, p95=1.14
las_norm$Classification[las_norm$Classification == 6 & las_norm$Z > 1.4]  <- 8L  # sand, p95=1.36
las_norm$Classification[las_norm$Classification == 3 & las_norm$Z > 1.2]  <- 8L  # graminoid, p95=1.20
las_norm$Classification[las_norm$Classification == 4 & las_norm$Z > 1.3]  <- 8L  # heath, p95=1.26
las_norm$Classification[las_norm$Classification == 1 & las_norm$Z > 3.5]  <- 8L  # bush, oekologisch

# ABWÄRTS-KORREKTUREN (tiefe tree-Punkte = Unterwuchs)
las_norm$Classification[las_norm$Classification == 8 & las_norm$Z < 0.5] <- 4L          # -> heath
las_norm$Classification[las_norm$Classification == 8 & las_norm$Z >= 0.5 & las_norm$Z < 2.0] <- 1L  # -> bush

# 7. Finale Datei speichern
writeLAS(las_norm, "/Users/luis/Documents/BA/data/processed/tiles_csf_knnidw_final_v3.laz")             

print(table(las_norm$Classification))


#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
#LAZ File für den Export vorbereiten

library(lidR)
library(sf)
las_norm <- readLAS("/Users/luis/Documents/BA/data/processed/tiles_csf_knnidw_final_v3.laz")

output_dir <- "/Users/luis/Documents/BA/data/processed/training_tiles_v3/"
dir.create(output_dir, showWarnings = FALSE)
# alte Tiles entfernen, damit ein neuer Lauf sauber startet
file.remove(list.files(output_dir, pattern = "\\.txt$", full.names = TRUE))

tile_size <- 10; stride <- 5; n_points <- 4096
# label_map: class_id -> 0-indexiertes Label fuer PointNet++.
# An die Klassenreihenfolge in class_levels gekoppelt.
# water reserviert, Label 8 erst mit water-Daten.
label_map <- c("1"=0,"2"=1,"3"=2,"4"=3,"5"=4,"6"=5,"7"=6,"8"=7,"9"=8)

# Sliding-Window je zusammenhaengendem Bereich.
# Schneise I (tile 1-8) grenzt aneinander und laeuft als ein durchgehendes
# Raster, so entstehen keine Ueberlappungen an Tile-Grenzen. Alle uebrigen
# Tiles liegen isoliert und bilden je einen eigenen Bereich.
# Der Bereichsname wird spaeter dem Dateinamen vorangestellt (rn__NNNN.txt),
# damit make_splits.py ganze Bereiche einem Split zuordnen kann.
dir_labeled <- "/Users/luis/Documents/BA/data/labeled"
regions <- list(
  schneiseI = c("Authausen_02_tile_1_segments_labeled_JS.gpkg",
                "Authausen_02_tile_2_segments_labeled_JS.gpkg",
                "Authausen_02_tile_3_segments_labeled_JS.gpkg",
                "Authausen_02_tile_4_segments_labeled_JS.gpkg",
                "Authausen_02_tile_5_segments_labeled_JS_neu.gpkg",
                "Authausen_02_tile_6_segments_labeled_JS_neu.gpkg",
                "Authausen_02_tile_7_segments_labeled_JS.gpkg",
                "Authausen_02_tile_8_segments_labeled_JS.gpkg"),
  authausen06_t75  = c("Authausen_06_tile_75_segments_labeled_JS.gpkg"),
  authausen06_t190 = c("Authausen_06_tile_190_segments_labeled_JS.gpkg"),
  authausen06_t226 = c("Authausen_06_tile_226_segments_labeled_JS.gpkg"),
  cux_t10  = c("Cuxhaven_01_tile_10_segments_labeled_JS.gpkg"),
  cux_t66  = c("Cuxhaven_01_tile_66_segments_labeled_JS.gpkg"),
  cux_t225 = c("Cuxhaven_01_tile_225_segments_labeled_JS.gpkg"),
  cux_t270 = c("Cuxhaven_01_tile_270_segments_labeled_JS.gpkg"),
  cux_t406 = c("Cuxhaven_01_tile_406_segments_labeled_JS.gpkg")
)

tile_count <- 0; skipped <- 0

buffer <- tile_size  # Puffer zwischen Bloecken

for (rn in names(regions)) {
  gps <- regions[[rn]]
  # gemeinsame Bounding-Box aller Tiles dieses Bereichs
  bxs  <- lapply(gps, function(g) st_bbox(st_read(file.path(dir_labeled, g), quiet = TRUE)))
  xmin <- floor(min(sapply(bxs, function(b) b["xmin"])))
  xmax <- ceiling(max(sapply(bxs, function(b) b["xmax"])))
  ymin <- floor(min(sapply(bxs, function(b) b["ymin"])))
  ymax <- ceiling(max(sapply(bxs, function(b) b["ymax"])))

  # Punkte dieses Bereichs einmal herausfiltern, dann lokal auswerten
  las_region <- filter_poi(las_norm,
      X >= xmin & X <= xmax & Y >= ymin & Y <= ymax)
  if (nrow(las_region@data) == 0) next

  # Schneise I in 3 X-Bloecke mit Puffer teilen, damit das Gebiet in
  # train/val/test vertreten ist. Der Puffer verhindert, dass ein Fenster ueber
  # eine Blockgrenze reicht. Andere Bereiche = ein Block.
  if (rn == "schneiseI") {
    u  <- (xmax - xmin) - 2 * buffer           
    wa <- u * 0.70; wb <- u * 0.15             
    blocks <- list(
      list(name = "schneiseI_a", x0 = xmin,                       x1 = xmin + wa),
      list(name = "schneiseI_b", x0 = xmin + wa + buffer,         x1 = xmin + wa + buffer + wb),
      list(name = "schneiseI_c", x0 = xmin + wa + 2*buffer + wb,  x1 = xmax)
    )
  } else {
    blocks <- list(list(name = rn, x0 = xmin, x1 = xmax))
  }

  for (blk in blocks) {
    k <- 0  # Patch-Zaehler innerhalb des Blocks
    for (xi in seq(blk$x0, blk$x1 - tile_size, by = stride)) {
      for (yi in seq(ymin, ymax - tile_size, by = stride)) {

        tile <- filter_poi(las_region,
                           X >= xi & X < (xi+tile_size) & Y >= yi & Y < (yi+tile_size))
        if (nrow(tile@data) < 100) { skipped <- skipped + 1; next }

        if (nrow(tile@data) >= n_points) {
          idx <- sample(nrow(tile@data), n_points, replace = FALSE)
        } else {
          idx <- sample(nrow(tile@data), n_points, replace = TRUE)
        }
        pts <- tile@data[idx, ]

        x_centered <- pts$X - (xi + tile_size/2)
        y_centered <- pts$Y - (yi + tile_size/2)
        r_norm <- pts$R/65535; g_norm <- pts$G/65535; b_norm <- pts$B/65535
        labels <- label_map[as.character(pts$Classification)]

        # 3 Zusatzspalten ans Ende
        mat <- cbind(x_centered, y_centered, pts$Z,
                     r_norm, g_norm, b_norm,
                     labels,
                     pts$shadow, pts$overexposed, pts$confidence)

        tile_count <- tile_count + 1
        k <- k + 1
        # Blockname vorangestellt, damit make_splits.py je Block splitten kann
        fname <- sprintf("%s/%s__%04d.txt", output_dir, blk$name, k)
        write.table(mat, fname, row.names=FALSE, col.names=FALSE, sep=" ")
      }
    }
  }
}
cat("Tiles:", tile_count, "| uebersprungen:", skipped, "\n")
# LabelMapping.R: Preprocessing der Trainingsdaten
# Daten ausserhalb des Repos unter /Users/luis/Documents/BA/data/:
#   raw/       Rohdaten (Punktwolke .laz, DGM, tile .tif)
#   labeled/   gelabelte GPKGs
#   processed/ Outputs (rds, LAZ, training_tiles_v2)
# Pfade absolut. Grosse Daten nicht im Git.

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

laz_file <- "/Users/luis/Documents/BA/data/raw/Punktwolke_AuthausenerWald_Schneise_I_20250522.laz"
dir_labeled <- "/Users/luis/Documents/BA/data/labeled"

# Dateinamen der 6 Tiles 
gpkg_files <- c(
  "Authausen_02_tile_1_segments_labeled_JS.gpkg",
  "Authausen_02_tile_2_segments_labeled_JS.gpkg",
  "Authausen_02_tile_3_segments_labeled_JS.gpkg",
  "Authausen_02_tile_4_segments_labeled_JS.gpkg",
  "Authausen_02_tile_5_segments_labeled_JS_neu.gpkg",
  "Authausen_02_tile_6_segments_labeled_JS_neu.gpkg"
)

tile_list <- list()

for (i in seq_along(gpkg_files)) {
  cat("Verarbeite Tile", i, ":", gpkg_files[i], "\n")
  
  polys <- st_read(file.path(dir_labeled, gpkg_files[i]), quiet = TRUE)
  polys <- st_make_valid(polys)
  bb <- st_bbox(polys)
  
  las_tile <- readLAS(laz_file, filter = paste(
    "-keep_xy", bb["xmin"], bb["ymin"], bb["xmax"], bb["ymax"]))
  
  # ALLE 4 Attribute nacheinander mappen
  las_lab <- merge_spatial(las_tile, polys, "class")
  las_lab <- merge_spatial(las_lab,  polys, "shadow")
  las_lab <- merge_spatial(las_lab,  polys, "overexposed")
  las_lab <- merge_spatial(las_lab,  polys, "confidence")
  
  # Nur gelabelte Punkte. Leerer String "" ist kein NA, daher extra abfangen.
  las_train <- filter_poi(las_lab, !is.na(class) & class != "")
  
  tile_list[[i]] <- las_train@data
}

# Alle Tiles zusammenfuehren
df_all <- do.call(rbind, tile_list)

# Klassen-ID per festem Woerterbuch, nicht as.factor.
# as.factor kodiert alphabetisch aus den vorhandenen Klassen, IDs verrutschen
# bei neuen oder fehlenden Klassen. Feste Liste haelt IDs stabil, Neues hinten an.
# water als id 9 reserviert, entsteht erst mit echten water-Punkten.
class_levels <- c("bush", "deadwood", "graminoid", "heath",
                  "other", "sand", "soil", "tree", "water")
df_all$class_id <- match(df_all$class, class_levels)

# Unbekannte Klassennamen werden zu NA. Abbruch statt stiller Fehlkodierung.
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

saveRDS(df_all, "/Users/luis/Documents/BA/data/processed/trainingspunkte_alle_tiles_v2.rds")


#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------

#Umwandeln der .rds file in eine LAZ File
#classification feld bekommt nummer die einer Klasse zugeordnet ist 8


library(lidR)

df_all <- readRDS("/Users/luis/Documents/BA/data/processed/trainingspunkte_alle_tiles_v2.rds")

# Flags in 0/1 umwandeln (robust gegen logical ODER string)
shadow_vec  <- as.integer(df_all$shadow      %in% c(TRUE, "true", "True", "1", 1))
overexp_vec <- as.integer(df_all$overexposed %in% c(TRUE, "true", "True", "1", 1))
conf_vec    <- as.numeric(df_all$confidence)

# Standard-LAS bauen (Text-/Flag-Spalten vorher raus)
df_std <- df_all
df_std$class       <- NULL
df_std$shadow      <- NULL
df_std$overexposed <- NULL
df_std$confidence  <- NULL

las_all <- LAS(df_std)
las_all$Classification <- as.integer(df_all$class_id)
st_crs(las_all) <- 25832

# Extra-Attribute REGISTRIEREN (sonst beim writeLAS verloren!)
las_all <- add_lasattribute(las_all, shadow_vec,  "shadow",      "Schatten 0/1")
las_all <- add_lasattribute(las_all, overexp_vec, "overexposed", "Ueberbelichtet 0/1")
las_all <- add_lasattribute(las_all, conf_vec,    "confidence",  "Label-Confidence")

writeLAS(las_all, "/Users/luis/Documents/BA/data/processed/trainingspunkte_alle_tiles_classified_v2.laz")

# Kontrolle: Attribute muessen erhalten sein
las_check <- readLAS("/Users/luis/Documents/BA/data/processed/trainingspunkte_alle_tiles_classified_v2.laz")
print(names(las_check@data))   # shadow, overexposed, confidence muessen dabei sein
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
#Höhen Schwellenwerte bestimmen um unrealistische Fehlklassifikationen zu korrigieren

library(lidR)

# 1. Rohe Punktwolke laden
las <- readLAS("/Users/luis/Documents/BA/data/processed/trainingspunkte_alle_tiles_classified_v2.laz")  # <- v2

# 2. Original-Labels sichern
# Classification haelt hier die semantische class_id. classify_ground()
# ueberschreibt das Feld mit dem ASPRS-Bodencode (2 = Boden) fuer
# normalize_height(). Labels in Schritt 5 wiederhergestellt.
# Reihenfolge Schritt 2 bis 5 nicht vertauschen.
las@data$LabelOriginal <- las$Classification

# 3. Boden geometrisch erkennen
las <- classify_ground(las, csf(
  cloth_resolution = 0.5,
  class_threshold  = 0.3,
  rigidness        = 2
))

# 4. Höhe normalisieren 
las_norm <- normalize_height(las, knnidw(k = 10, p = 2))

# 5. Originale semantische Labels wiederherstellen
las_norm$Classification <- las_norm$LabelOriginal

# 6. Extreme Ausreißer nach unten entfernen 
las_norm <- filter_poi(las_norm, Z >= -1.0)

# --- Schwellenwert-Regeln ---
# Grenzen = 95. Perzentil der normierten Hoehe je Klasse (hoehen_analyse.R).
# bush per Oekologie (3.5 m), da p95 durch Baumueberhang verfaelscht.
# Aufwaerts zuerst, dann Abwaerts.

# AUFWÄRTS-KORREKTUREN (zu hoch fuer die Klasse -> tree)
las_norm$Classification[las_norm$Classification == 7 & las_norm$Z > 0.85] <- 8L  # soil, p95=0.84
las_norm$Classification[las_norm$Classification == 6 & las_norm$Z > 1.5]  <- 8L  # sand, p95=1.51
las_norm$Classification[las_norm$Classification == 3 & las_norm$Z > 1.5]  <- 8L  # graminoid, p95=1.55
las_norm$Classification[las_norm$Classification == 4 & las_norm$Z > 1.9]  <- 8L  # heath, p95=1.94
las_norm$Classification[las_norm$Classification == 1 & las_norm$Z > 3.5]  <- 8L  # bush, oekologisch
# deadwood (2) ohne Regel, Hoehe bimodal (liegend/stehend)

# ABWÄRTS-KORREKTUREN (tiefe tree-Punkte = Unterwuchs; Heuristik)
las_norm$Classification[las_norm$Classification == 8 & las_norm$Z < 0.5] <- 4L          # -> heath
las_norm$Classification[las_norm$Classification == 8 & las_norm$Z >= 0.5 & las_norm$Z < 2.0] <- 1L  # -> bush

# 7. Finale Datei speichern
writeLAS(las_norm, "/Users/luis/Documents/BA/data/processed/tiles_csf_knnidw_final_v2.laz")             # <- v2

print(table(las_norm$Classification))


#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
#LAZ File für den Export vorbereiten

library(lidR)
las_norm <- readLAS("/Users/luis/Documents/BA/data/processed/tiles_csf_knnidw_final_v2.laz")

output_dir <- "/Users/luis/Documents/BA/data/processed/training_tiles_v2/"
dir.create(output_dir, showWarnings = FALSE)

tile_size <- 10; stride <- 5; n_points <- 4096
# label_map: class_id -> 0-indexiertes Label fuer PointNet++.
# An Klassenreihenfolge gekoppelt (class_levels, Block 3).
# water (9 -> 8) reserviert, Label 8 erst mit water-Daten.
# num_classes im Config = vorhandene Klassen (8 ohne water, sonst leere Klasse).
label_map <- c("1"=0,"2"=1,"3"=2,"4"=3,"5"=4,"6"=5,"7"=6,"8"=7,"9"=8)

xmin <- floor(min(las_norm$X)); xmax <- ceiling(max(las_norm$X))
ymin <- floor(min(las_norm$Y)); ymax <- ceiling(max(las_norm$Y))

tile_count <- 0; skipped <- 0

for (xi in seq(xmin, xmax - tile_size, by = stride)) {
  for (yi in seq(ymin, ymax - tile_size, by = stride)) {
    
    tile <- filter_poi(las_norm,
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
    
    # NEU: 3 Zusatzspalten ans Ende
    mat <- cbind(x_centered, y_centered, pts$Z,
                 r_norm, g_norm, b_norm,
                 labels,
                 pts$shadow, pts$overexposed, pts$confidence)
    
    tile_count <- tile_count + 1
    fname <- sprintf("%s/tile_%04d.txt", output_dir, tile_count)
    write.table(mat, fname, row.names=FALSE, col.names=FALSE, sep=" ")
  }
}
cat("Tiles:", tile_count, "| uebersprungen:", skipped, "\n")
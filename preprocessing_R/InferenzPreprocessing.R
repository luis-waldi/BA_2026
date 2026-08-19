# InferenzPreprocessing.R: Vorbereitung ungelabelter Befliegungsgebiete
#
# Erzeugt aus einer Rohpunktwolke dieselbe Datengrundlage, auf der das Modell
# trainiert wurde: Hoehe ueber Boden statt absoluter Hoehe, dazu die beiden
# lokalen Hoehenmerkmale z_rel und z_range. Anders als LabelMapping.R laeuft
# das ohne Labels und ueber das gesamte Befliegungsgebiet.
#
# Die Zerlegung in Kacheln passiert nicht hier, sondern in predict.py. Bei
# 50 Prozent Fensterueberlappung waeren die Kacheln als Text ein Vielfaches
# der Punktwolke gross, als LAZ bleibt es bei ihrer Groessenordnung.
#
# Verarbeitet wird blockweise ueber die LAScatalog-Engine von lidR. Eine
# Bodenklassifikation ueber mehrere Quadratkilometer am Stueck ist weder
# speicher- noch zeittechnisch sinnvoll. Der Puffer verhindert Kanten an den
# Blockgrenzen, die Pufferpunkte werden vor dem Schreiben entfernt.

library(lidR)
library(sf)
library(terra)

# --- Konfiguration ---
in_laz  <- "/Users/luis/Documents/BA/data/raw/Punktwolke_AuthausenerWald_110kVTrassePSAPlatz_20250521.laz"
out_dir <- "/Users/luis/Documents/BA/data/processed/inferenz_110kv/"

chunk_size   <- 100   # m Kantenlaenge je Verarbeitungsblock
chunk_buffer <- 20    # m Puffer, deutlich groesser als das 2.5-m-Fenster von z_rel
HEIGHT_CLIP  <- 5     # m, wie im Trainingsexport

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

px_fun <- if ("pixel_metrics" %in% getNamespaceExports("lidR")) {
  lidR::pixel_metrics   # lidR >= 4
} else {
  lidR::grid_metrics    # lidR 3.x
}

# --- Lokale Hoehenmerkmale ---
# Identisch zu LabelMapping.R, damit die Eingangswerte bei der Inferenz
# dieselbe Bedeutung haben wie beim Training.
#   z_range  Hoehenspanne in der 0.5-m-Umgebung
#   z_rel    Hoehe ueber der medianen Hoehe der 2.5-m-Umgebung
add_height_features <- function(las) {
  m05 <- px_fun(las, ~list(zrange = max(Z) - min(Z), zmed = median(Z)),
                res = 0.5)
  if (!inherits(m05, "SpatRaster")) m05 <- terra::rast(m05)

  zmed_umgebung <- terra::focal(m05[["zmed"]], w = matrix(1, 5, 5),
                                fun = median, na.rm = TRUE)

  xy      <- cbind(las@data$X, las@data$Y)
  z_range <- terra::extract(m05[["zrange"]], xy)[, 1]
  z_med_u <- terra::extract(zmed_umgebung,   xy)[, 1]

  z_range <- ifelse(is.na(z_range), 0, z_range)
  z_rel   <- ifelse(is.na(z_med_u), 0, las@data$Z - z_med_u)

  las@data$z_range <- pmin(z_range, HEIGHT_CLIP)
  las@data$z_rel   <- pmax(pmin(z_rel, HEIGHT_CLIP), -HEIGHT_CLIP)
  las
}

# --- Verarbeitung eines Blocks ---
# Parameter der Bodenklassifikation und der Interpolation sind aus
# LabelMapping.R uebernommen, damit die Hoehennormalisierung identisch ist.
process_chunk <- function(chunk) {
  las <- readLAS(chunk)
  if (is.empty(las)) return(NULL)

  las <- classify_ground(las, csf(cloth_resolution = 0.5,
                                  class_threshold  = 0.3,
                                  rigidness        = 2))
  las <- normalize_height(las, knnidw(k = 10, p = 2))
  las <- filter_poi(las, Z >= -1.0)
  if (is.empty(las)) return(NULL)

  las <- add_height_features(las)

  # Pufferpunkte entfernen, erst danach die Attribute festschreiben
  las <- filter_poi(las, buffer == 0)
  if (is.empty(las)) return(NULL)

  # Ohne add_lasattribute gehen die beiden Merkmale beim Schreiben verloren.
  # Die Beschreibung darf hoechstens 32 Zeichen lang sein.
  las <- add_lasattribute(las, name = "z_rel",
                          desc = "Hoehe ueber Median 2.5 m")
  las <- add_lasattribute(las, name = "z_range",
                          desc = "Hoehenspanne 0.5 m")
  las
}

# --- Lauf ---
ctg <- readLAScatalog(in_laz)
opt_chunk_size(ctg)     <- chunk_size
opt_chunk_buffer(ctg)   <- chunk_buffer
opt_laz_compression(ctg) <- TRUE
opt_output_files(ctg)   <- paste0(out_dir, "norm_{XLEFT}_{YBOTTOM}")

cat("Ausdehnung:", paste(round(as.numeric(st_bbox(ctg))), collapse = " "), "\n")
cat("Punkte gesamt:", sum(ctg$Number.of.point.records), "\n")

res <- catalog_apply(ctg, process_chunk)

files <- list.files(out_dir, pattern = "\\.laz$", full.names = TRUE)
cat("Geschriebene Bloecke:", length(files), "\n")
cat("Gesamtgroesse:", round(sum(file.size(files)) / 1e6), "MB\n")

# --- Kontrolle ---
# Stichprobe eines Blocks, die Werte muessen in derselben Groessenordnung
# liegen wie im Trainingsexport (z_range Median rund 1 m, z_rel Median 0).
if (length(files) > 0) {
  probe <- readLAS(files[[ceiling(length(files) / 2)]])
  cat("Kontrollblock:", basename(files[[ceiling(length(files) / 2)]]),
      "|", nrow(probe@data), "Punkte\n")
  cat("  Z       Median/p95:",
      round(quantile(probe@data$Z, c(0.5, 0.95)), 2), "\n")
  cat("  z_range Median/p95:",
      round(quantile(probe@data$z_range, c(0.5, 0.95)), 2), "\n")
  cat("  z_rel   Median/p95:",
      round(quantile(probe@data$z_rel, c(0.5, 0.95)), 2), "\n")
  cat("  Attribute:", paste(names(probe@data), collapse = ", "), "\n")
}

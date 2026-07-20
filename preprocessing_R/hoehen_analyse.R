# hoehen_analyse.R: Herleitung der Hoehen-Schwellen fuer Block 5.
# Leitet die Schwellen aus der Hoehenverteilung je Klasse ab statt sie zu raten.
# Liest nur, schreibt Statistik-Tabelle und Boxplot, aendert keine Pipeline-Daten.
# Input classified_v2.laz: class_id im Feld Classification, Hoehe noch absolut.
# Normalisierung mit denselben Parametern wie Block 5.

library(lidR)
library(dplyr)

proc <- "/Users/luis/Documents/BA/data/processed"
out  <- file.path(proc, "hoehen_analyse")
dir.create(out, showWarnings = FALSE)

# 1. Klassifizierte Wolke laden
las <- readLAS(file.path(proc, "trainingspunkte_alle_tiles_classified_v2.laz"))

# 2. Semantische Labels sichern (classify_ground ueberschreibt Classification)
las@data$LabelOriginal <- las$Classification

# 3. Boden erkennen + Hoehe normalisieren
las <- classify_ground(las, csf(cloth_resolution = 0.5,
                                class_threshold  = 0.3,
                                rigidness        = 2))
las_norm <- normalize_height(las, knnidw(k = 10, p = 2))

# 4. Semantische Labels wiederherstellen, grobe Ausreisser raus
las_norm$Classification <- las_norm$LabelOriginal
las_norm <- filter_poi(las_norm, Z >= -1.0)

# 5. Klassennamen anhaengen
class_levels <- c("bush", "deadwood", "graminoid", "heath",
                  "other", "sand", "soil", "tree", "water")
df <- las_norm@data
df$class_name <- class_levels[df$Classification]

# 6. Hoehen-Quantile je Klasse. p95 der tiefen Klassen liefert die Obergrenzen,
#    der untere Bereich von tree/bush den Unterwuchs.
stats <- df %>%
  group_by(class_name) %>%
  summarise(n      = n(),
            p01    = quantile(Z, 0.01),
            p05    = quantile(Z, 0.05),
            p25    = quantile(Z, 0.25),
            median = median(Z),
            p75    = quantile(Z, 0.75),
            p90    = quantile(Z, 0.90),
            p95    = quantile(Z, 0.95),
            p99    = quantile(Z, 0.99),
            max    = max(Z)) %>%
  arrange(median)

stats <- as.data.frame(stats)
print(stats, digits = 3)
write.csv(stats, file.path(out, "hoehen_quantile_je_klasse.csv"), row.names = FALSE)

# 7. Boxplot (Ausreisser ausgeblendet, y auf sinnvollen Bereich begrenzt)
png(file.path(out, "hoehen_boxplot.png"), width = 1400, height = 900, res = 150)
boxplot(Z ~ class_name, data = df, outline = FALSE,
        ylim = c(-0.5, 6),
        ylab = "normierte Hoehe [m]", xlab = "",
        main = "Hoehenverteilung je Klasse (6 Tiles)",
        col = "grey85")
abline(h = c(0.5, 1.0, 1.5, 3.5), col = "red", lty = 2)  # aktuelle Schwellen
dev.off()

cat("\nFertig. Ergebnisse in:\n  ", out, "\n")

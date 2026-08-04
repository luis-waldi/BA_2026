# hoehen_analyse.R: Herleitung der Hoehen-Schwellen fuer die Korrektur in
# LabelMapping.R. Leitet die Schwellen aus der Hoehenverteilung je Klasse ab
# statt sie zu raten. Liest nur, schreibt Statistik-Tabelle und Boxplot.
# Input classified_v3.laz: Z ist bereits pro Tile normalisiert, Classification
# enthaelt die class_id. Keine erneute Normalisierung noetig.

set_lidr_threads(0)
library(lidR)
library(dplyr)

proc <- "/Users/luis/Documents/BA/data/processed"
out  <- file.path(proc, "hoehen_analyse")
dir.create(out, showWarnings = FALSE)

# Klassifizierte Wolke laden
las_norm <- readLAS(file.path(proc, "trainingspunkte_alle_tiles_classified_v3.laz"))

# Klassennamen anhaengen
class_levels <- c("bush", "deadwood", "graminoid", "heath",
                  "other", "sand", "soil", "tree", "water")
df <- las_norm@data
df$class_name <- class_levels[df$Classification]

# Hoehen-Quantile je Klasse. p95 der tiefen Klassen liefert die Obergrenzen,
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

# Boxplot
png(file.path(out, "hoehen_boxplot.png"), width = 1400, height = 900, res = 150)
boxplot(Z ~ class_name, data = df, outline = FALSE,
        ylim = c(-0.5, 6),
        ylab = "normierte Hoehe [m]", xlab = "",
        main = "Hoehenverteilung je Klasse",
        col = "grey85")
abline(h = c(0.5, 1.0, 1.5, 3.5), col = "red", lty = 2)  # aktuelle Schwellen
dev.off()

cat("\nFertig. Ergebnisse in:\n  ", out, "\n")

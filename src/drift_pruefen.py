# Datei: src/drift_pruefen.py – in VS Code einfügen und speichern
# Data Drift prüfen (ab Tag 19).
# Frage: Sehen die neuen Weine (z. B. aus ausgabe/letzte_anfragen.csv) aus wie die Trainingsdaten?
# Für jedes Merkmal vergleichen wir die beiden Verteilungen mit dem Kolmogorow-Smirnow-Test (KS-Test).
# Kleiner p-Wert (unter 0.01): Die Verteilungen sind sehr wahrscheinlich verschieden -> Drift.
#
# Aufruf:  python src/drift_pruefen.py ausgabe/letzte_anfragen.csv

import sys

import pandas as pd
from scipy.stats import ks_2samp

GRENZE = 0.01

if len(sys.argv) != 2:
    raise SystemExit("Aufruf: python src/drift_pruefen.py <datei.csv>")

training = pd.read_csv("data/prepared/train.csv").drop(columns="target")
# Unser Lastgenerator schickt dieselben Weine mehrmals. Wir zählen jeden Wein nur einmal.
# Sonst wirken kleine Zufallsunterschiede viel zu groß, und der Test meldet falschen Drift.
neu = pd.read_csv(sys.argv[1]).drop_duplicates()

print(f"Training: {len(training)} Weine, neu: {len(neu)} Weine\n")
print(f"{'Merkmal':30} {'Mittel Training':>16} {'Mittel neu':>11} {'p-Wert':>9}  Drift?")
drift = []
for spalte in training.columns:
    p_wert = ks_2samp(training[spalte], neu[spalte]).pvalue
    ist_drift = p_wert < GRENZE
    if ist_drift:
        drift.append(spalte)
    print(f"{spalte:30} {training[spalte].mean():16.2f} {neu[spalte].mean():11.2f} {p_wert:9.4f}  {'JA' if ist_drift else '-'}")

print()
if drift:
    print(f"DRIFT bei {len(drift)} Merkmal(en): {', '.join(drift)}")
    sys.exit(1)
print("Kein Drift gefunden.")
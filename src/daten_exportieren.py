# Datei: src/daten_exportieren.py – in VS Code einfügen und speichern
# Wir speichern den Wine-Datensatz als CSV-Datei im Projekt.
# Aufruf: python src/daten_exportieren.py [anzahl]
# Ohne Zahl speichern wir alle 178 Weine.

import sys
from pathlib import Path

from sklearn.datasets import load_wine

ZIEL = Path("data/raw/wein.csv")

# Alle Weine laden: 13 Merkmale und die Spalte target (Klasse 0, 1 oder 2).
tabelle = load_wine(as_frame=True).frame

# Wir mischen die Zeilen mit festem Zufall. So ist jede Teilmenge gemischt
# und das Ergebnis ist bei jedem Aufruf gleich.
tabelle = tabelle.sample(frac=1, random_state=42).reset_index(drop=True)

if len(sys.argv) > 1:
    anzahl = int(sys.argv[1])
    tabelle = tabelle.head(anzahl)

ZIEL.parent.mkdir(parents=True, exist_ok=True)
tabelle.to_csv(ZIEL, index=False)

print(f"Gespeichert: {ZIEL} mit {len(tabelle)} Weinen und {tabelle.shape[1]} Spalten")
print("Weine pro Klasse:", tabelle["target"].value_counts().sort_index().to_dict())
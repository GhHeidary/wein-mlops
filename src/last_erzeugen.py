# Datei: src/last_erzeugen.py – in VS Code einfügen und speichern
# Erzeugt Verkehr für die Wein-API (ab Tag 19), damit wir im Monitoring etwas sehen.
# Normal:    zufällige Testweine, dazu etwa 10 % ungültige Eingaben (Status 422).
# --drift:   "neue Lieferung" mit veränderten Werten (mehr Alkohol, weniger Flavanoide, mehr Farbe).
# --llm N:   jede N-te Anfrage geht an /erklaerung (0 = nie).
# Alle gültigen Weine, die wir schicken, speichern wir in ausgabe/letzte_anfragen.csv (für src/drift_pruefen.py).
#
# Beispiele:
#   python src/last_erzeugen.py --sekunden 60
#   python src/last_erzeugen.py --sekunden 60 --drift
#   python src/last_erzeugen.py --sekunden 60 --llm 20

import argparse
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import requests

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

parser = argparse.ArgumentParser(description="Verkehr für die Wein-API erzeugen")
parser.add_argument("--sekunden", type=int, default=60, help="So lange schicken wir Anfragen.")
parser.add_argument("--drift", action="store_true", help="Veränderte Weine schicken (Data Drift).")
parser.add_argument("--llm", type=int, default=0, help="Jede N-te Anfrage an /erklaerung (0 = nie).")
args = parser.parse_args()

# Zufall mit festem Startwert: Bei jedem Lauf dieselbe Folge von Weinen.
zufall = np.random.default_rng(42)
weine = pd.read_csv("data/prepared/test.csv").drop(columns="target")

if args.drift:
    # Die "neue Lieferung": andere Werte als in den Trainingsdaten.
    # clip: Wir bleiben in den erlaubten Grenzen der API (sonst Status 422).
    weine["alcohol"] = (weine["alcohol"] + 1.0).clip(upper=16)
    weine["flavanoids"] = (weine["flavanoids"] * 0.5).clip(lower=0.2)
    weine["color_intensity"] = (weine["color_intensity"] + 3.0).clip(upper=14)

ergebnisse = Counter()
gesendet = []
ende = time.time() + args.sekunden
nummer = 0
print(f"Sende Anfragen an {API_URL} für {args.sekunden} Sekunden (Drift: {args.drift}) ...")
while time.time() < ende:
    nummer += 1
    wein = weine.iloc[zufall.integers(len(weine))].to_dict()
    try:
        if zufall.random() < 0.1:
            # Etwa jede zehnte Anfrage ist ungültig: 99 % Alkohol.
            wein["alcohol"] = 99
            antwort = requests.post(f"{API_URL}/vorhersage", json=wein, timeout=10)
        elif args.llm and nummer % args.llm == 0:
            antwort = requests.post(f"{API_URL}/erklaerung", json=wein, timeout=180)
            gesendet.append(wein)
        else:
            antwort = requests.post(f"{API_URL}/vorhersage", json=wein, timeout=10)
            gesendet.append(wein)
        ergebnisse[antwort.status_code] += 1
    except requests.RequestException:
        # Die API antwortet nicht (z. B. gestoppt oder startet gerade). Wir zählen das und machen weiter.
        ergebnisse["keine Verbindung"] += 1
        time.sleep(1)
    time.sleep(0.1)

print(f"Fertig: {nummer} Anfragen, Statuscodes: {dict(sorted(ergebnisse.items(), key=str))}")
Path("ausgabe").mkdir(exist_ok=True)
pd.DataFrame(gesendet).to_csv("ausgabe/letzte_anfragen.csv", index=False)
print(f"Gespeichert: ausgabe/letzte_anfragen.csv ({len(gesendet)} gültige Weine)")
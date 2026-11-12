# Datei: api/app.py – in VS Code einfügen und speichern
# Unsere Modell-API mit FastAPI (ab Tag 14) – Stufe 3: mit Vorhersage.
# Start im Projektordner:  uvicorn api.app:app --reload

import json
from pathlib import Path

import joblib
import pandas as pd
import yaml
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Die drei Klassen im Wine-Datensatz sind drei Rebsorten aus Italien.
KLASSEN = {0: "Sorte 1", 1: "Sorte 2", 2: "Sorte 3"}

# Hier erzeugen wir die App. Titel, Beschreibung und Version sehen wir unter /docs.
app = FastAPI(title="Wein-API", description="Vorhersage der Weinsorte", version="0.1.0")

# Wir laden das Modell EINMAL beim Start – nicht bei jeder Anfrage.
# joblib kann beim Laden Code ausführen. Deshalb laden wir nur unser eigenes Modell.
modell = joblib.load("models/modell.joblib")


# Mit Pydantic beschreiben wir, wie ein Wein aussehen muss:
# 13 Felder, jedes Feld ist eine Kommazahl (float).
class Wein(BaseModel):
    """Die 13 Messwerte eines Weins – dieselben Spalten wie in data/raw/wein.csv."""

    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    # Der Spaltenname enthält "/" – in Python geht das nicht.
    # Mit alias heißt das Feld in JSON trotzdem so wie die Spalte.
    od280_od315_of_diluted_wines: float = Field(alias="od280/od315_of_diluted_wines")
    proline: float


# Der Decorator @app.get("/") verbindet die Adresse "/" mit der Funktion darunter.
@app.get("/")
def startseite():
    # Wir geben ein Python-Dictionary zurück. FastAPI macht daraus JSON.
    return {"nachricht": "Die Wein-API läuft", "dokumentation": "/docs"}


# /health beantwortet nur eine Frage: Läuft die API noch?
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/modell")
def modell_info():
    # Wir zeigen, welches Modell läuft: Parameter und Ergebnis aus der Pipeline.
    params = yaml.safe_load(Path("params.yaml").read_text(encoding="utf-8"))
    metriken = json.loads(Path("metrics.json").read_text(encoding="utf-8"))
    return {"parameter": params["trainieren"], "metriken": metriken}


@app.get("/klassen/{nummer}")
def klasse(nummer: int):
    # Pfad-Parameter: Die Nummer steht direkt im Pfad, zum Beispiel /klassen/2.
    # nummer: int bedeutet: FastAPI prüft, ob es eine ganze Zahl ist.
    return {"nummer": nummer, "name": KLASSEN.get(nummer, "unbekannt")}


@app.get("/beispiele")
def beispiele(anzahl: int = 2):
    # Query-Parameter: steht nach dem ?, zum Beispiel /beispiele?anzahl=3.
    # Ohne Angabe nehmen wir 2 Weine.
    weine = pd.read_csv("beispiele/neue_weine.csv").head(anzahl)
    return weine.to_dict(orient="records")


# POST: Der Client schickt Daten im Body. FastAPI prüft sie mit der Klasse Wein.
@app.post("/vorhersage")
def vorhersage(wein: Wein):
    # by_alias=True: Wir bekommen die Spaltennamen genau wie beim Training.
    daten = pd.DataFrame([wein.model_dump(by_alias=True)])
    # Das Modell erwartet eine Tabelle. Unsere Tabelle hat eine Zeile.
    klasse = int(modell.predict(daten)[0])
    wahrscheinlichkeit = float(modell.predict_proba(daten).max())
    return {
        "klasse": klasse,
        "name": KLASSEN[klasse],
        "wahrscheinlichkeit": round(wahrscheinlichkeit, 2),
    }
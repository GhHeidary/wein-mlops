# Datei: tests/test_metriken.py – in VS Code einfügen und speichern
# Ab Tag 19: Wir prüfen, ob die API ihre Metriken für Prometheus richtig zählt.
# Braucht models/modell.joblib. In CI läuft der Test deshalb im Job "pipeline".

import json
from pathlib import Path

from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app)
WEIN = json.loads(Path("beispiele/wein.json").read_text(encoding="utf-8"))


def wert(text: str, zeile: str) -> float:
    """Sucht in der Ausgabe von /metrics eine Zeile und gibt die Zahl am Ende zurück."""
    for z in text.splitlines():
        if z.startswith(zeile + " "):
            return float(z.split()[-1])
    return 0.0


def test_metrics_erreichbar():
    antwort = client.get("/metrics")
    assert antwort.status_code == 200
    assert "wein_http_anfragen_total" in antwort.text


def test_vorhersage_wird_gezaehlt():
    zeile = 'wein_vorhersagen_total{sorte="Sorte 1"}'
    vorher = wert(client.get("/metrics").text, zeile)
    client.post("/vorhersage", json=WEIN)
    nachher = wert(client.get("/metrics").text, zeile)
    assert nachher == vorher + 1


def test_fehler_wird_gezaehlt():
    # Ein Wein mit 99 % Alkohol ist ungültig -> 422. Auch das zählt die Middleware.
    # prometheus-client sortiert die Labels in der Ausgabe alphabetisch.
    zeile = 'wein_http_anfragen_total{methode="POST",pfad="/vorhersage",status="422"}'
    vorher = wert(client.get("/metrics").text, zeile)
    client.post("/vorhersage", json={**WEIN, "alcohol": 99})
    nachher = wert(client.get("/metrics").text, zeile)
    assert nachher == vorher + 1
# Datei: tests/test_llm.py – in VS Code einfügen und speichern
# Wir testen den Endpunkt /erklaerung – ohne echtes LLM.
# Statt das LLM zu fragen, ersetzen wir die Funktion llm.erklaeren durch eine Attrappe (Mock).
# So läuft der Test schnell, kostenlos und immer gleich – auch in CI ohne Ollama und ohne Key.
# Braucht models/modell.joblib. In CI läuft der Test deshalb im Job "pipeline".

import json
from pathlib import Path

from fastapi.testclient import TestClient

from api import llm
from api.app import app

client = TestClient(app)
WEIN = json.loads(Path("beispiele/wein.json").read_text(encoding="utf-8"))


def falsches_llm(sorte, wahrscheinlichkeit, merkmale):
    # Die Attrappe antwortet sofort mit einem festen Text.
    return {
        "text": f"Das Modell wählt {sorte}.",
        "llm_modell": "attrappe",
        "prompt_version": llm.PROMPT_VERSION,
        "tokens": 42,
        "sekunden": 0.01,
    }


def test_ohne_llm_503(monkeypatch):
    # Ohne LLM_BASE_URL gibt es kein LLM. Die API sagt das klar mit Status 503.
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    antwort = client.post("/erklaerung", json=WEIN)
    assert antwort.status_code == 503


def test_erklaerung_mit_attrappe(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://nicht-benutzt")
    monkeypatch.setattr(llm, "erklaeren", falsches_llm)
    antwort = client.post("/erklaerung", json=WEIN)
    assert antwort.status_code == 200
    daten = antwort.json()
    assert daten["vorhersage"]["name"] == "Sorte 1"
    assert daten["text"] == "Das Modell wählt Sorte 1."
    assert len(daten["merkmale"]) == 3


def test_llm_kaputt_502(monkeypatch):
    # Das LLM ist eingerichtet, antwortet aber nicht. Die API meldet Status 502.
    def kaputt(*args):
        raise ConnectionError("keine Verbindung")

    monkeypatch.setenv("LLM_BASE_URL", "http://nicht-benutzt")
    monkeypatch.setattr(llm, "erklaeren", kaputt)
    antwort = client.post("/erklaerung", json=WEIN)
    assert antwort.status_code == 502
    assert "ConnectionError" in antwort.json()["detail"]
# Datei: src/evaluieren_llm.py – in VS Code einfügen und speichern
# Evaluation der LLM-Erklärungen (ab Tag 18).
# Ein LLM antwortet nicht immer gleich. Wir prüfen deshalb jede Antwort mit einfachen Regeln:
#   1. sorte:    Nennt die Erklärung die vorhergesagte Sorte?
#   2. ehrlich:  Nennt sie KEINE andere Sorte?
#   3. merkmal:  Nennt sie mindestens ein Merkmal aus dem Prompt?
#   4. kurz:     Hat sie höchstens drei Sätze?
#   5. deutsch:  Ist sie auf Deutsch?
# Eine Antwort ist "bestanden", wenn alle fünf Regeln erfüllt sind.
# Das Ergebnis speichern wir in ausgabe/ und als Run in MLflow (Experiment wein-evaluation).

import json
import os
import re
from pathlib import Path

# MLflow 3.17 zeigt sonst einen langen Hinweis zu "Tracing". Den brauchen wir nicht.
# Wir schalten ihn vor dem Import von MLflow aus.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
SORTEN = ["Sorte 1", "Sorte 2", "Sorte 3"]

# Das LLM übersetzt Merkmale oft ins Deutsche. Wir suchen deshalb nach mehreren Wörtern.
MERKMAL_WOERTER = {
    "alcohol": ["alcohol", "alkohol"],
    "malic_acid": ["malic", "äpfelsäure", "apfelsäure"],
    "ash": ["ash", "asche"],
    "alcalinity_of_ash": ["alcalinity", "alkalinität", "alkalität"],
    "magnesium": ["magnesium"],
    "total_phenols": ["phenol"],
    "flavanoids": ["flavanoid", "flavonoid"],
    "nonflavanoid_phenols": ["nonflavanoid", "nicht-flavanoid"],
    "proanthocyanins": ["proanthocyan"],
    "color_intensity": ["color", "farb"],
    "hue": ["hue", "farbton"],
    "od280/od315_of_diluted_wines": ["od280", "od315"],
    "proline": ["prolin"],
}
DEUTSCHE_WOERTER = [" der ", " die ", " das ", " und ", " ist ", " ein", " hat "]


def pruefen(text: str, sorte: str, merkmale: list[str]) -> dict:
    klein = f" {text.lower()} "
    saetze = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    return {
        "sorte": sorte in text,
        "ehrlich": not any(andere in text for andere in SORTEN if andere != sorte),
        "merkmal": any(w in klein for m in merkmale for w in MERKMAL_WOERTER.get(m, [m.lower()])),
        "kurz": len(saetze) <= 3,
        "deutsch": sum(w in klein for w in DEUTSCHE_WOERTER) >= 2,
    }


weine = pd.read_csv("beispiele/neue_weine.csv").to_dict(orient="records")
ergebnisse = []
for nummer, wein in enumerate(weine, start=1):
    antwort = requests.post(f"{API_URL}/erklaerung", json=wein, timeout=180)
    if antwort.status_code != 200:
        raise SystemExit(f"Fehler {antwort.status_code}: {antwort.text}")
    daten = antwort.json()
    sorte = daten["vorhersage"]["name"]
    regeln = pruefen(daten["text"], sorte, list(daten["merkmale"]))
    bestanden = all(regeln.values())
    ergebnisse.append({"wein": nummer, "sorte": sorte, **daten, "regeln": regeln, "bestanden": bestanden})
    falsch = [name for name, ok in regeln.items() if not ok]
    print(f"Wein {nummer}: {sorte:8} {daten['sekunden']:6.2f} s  {'bestanden' if bestanden else 'NICHT bestanden: ' + ', '.join(falsch)}")

anzahl = len(ergebnisse)
metriken = {
    "bestanden_quote": round(sum(e["bestanden"] for e in ergebnisse) / anzahl, 2),
    "latenz_mittel_s": round(sum(e["sekunden"] for e in ergebnisse) / anzahl, 2),
    "tokens_mittel": round(sum(e["tokens"] or 0 for e in ergebnisse) / anzahl, 1),
}
# Für jede Regel: Wie viele Antworten erfüllen sie?
for regel in ergebnisse[0]["regeln"]:
    metriken[f"regel_{regel}"] = round(sum(e["regeln"][regel] for e in ergebnisse) / anzahl, 2)

llm_modell = ergebnisse[0]["llm_modell"]
prompt_version = ergebnisse[0]["prompt_version"]
print(f"LLM: {llm_modell}, Prompt {prompt_version}")
print(f"Bestanden: {metriken['bestanden_quote']:.0%}, Latenz im Mittel: {metriken['latenz_mittel_s']} s, "
      f"Tokens im Mittel: {metriken['tokens_mittel']}")

Path("ausgabe").mkdir(exist_ok=True)
Path("ausgabe/evaluation_llm.json").write_text(
    json.dumps({"metriken": metriken, "antworten": ergebnisse}, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
print("Gespeichert: ausgabe/evaluation_llm.json")

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("wein-evaluation")
with mlflow.start_run(run_name=f"llm-{llm_modell}-{prompt_version}"):
    mlflow.log_params({"llm_modell": llm_modell, "prompt_version": prompt_version, "weine": anzahl})
    mlflow.log_metrics(metriken)
    mlflow.set_tag("art", "llm")
    # Alle Antworten als Datei im Run. So können wir sie später lesen und vergleichen.
    mlflow.log_dict({"antworten": ergebnisse}, "antworten.json")
print(f"In MLflow gespeichert: Experiment wein-evaluation, Run llm-{llm_modell}-{prompt_version}")
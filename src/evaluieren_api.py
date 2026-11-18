# Datei: src/evaluieren_api.py – in VS Code einfügen und speichern
# Evaluation der laufenden API (ab Tag 18).
# Wir schicken alle Testweine über HTTP an die API – genau wie ein echter Nutzer.
# Dann prüfen wir drei Dinge:
#   1. Qualität:  Stimmt die Genauigkeit über die API mit metrics.json überein?
#   2. Latenz:    Wie schnell antwortet die API?
#   3. Robustheit: Lehnt die API falsche Eingaben richtig ab?
# Das Ergebnis speichern wir in ausgabe/ und als Run in MLflow (Experiment wein-evaluation).

import json
import os
import statistics
import time
from pathlib import Path

# MLflow 3.17 zeigt sonst einen langen Hinweis zu "Tracing". Den brauchen wir nicht.
# Wir schalten ihn vor dem Import von MLflow aus.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

# --- 1. Qualität und 2. Latenz -------------------------------------------------
test = pd.read_csv("data/prepared/test.csv")
richtig = 0
zeiten = []
for _, zeile in test.iterrows():
    wein = zeile.drop("target").to_dict()
    start = time.perf_counter()
    antwort = requests.post(f"{API_URL}/vorhersage", json=wein, timeout=10)
    zeiten.append((time.perf_counter() - start) * 1000)  # in Millisekunden
    antwort.raise_for_status()
    if antwort.json()["klasse"] == int(zeile["target"]):
        richtig += 1

genauigkeit_api = round(richtig / len(test), 4)
genauigkeit_pipeline = json.loads(Path("metrics.json").read_text(encoding="utf-8"))["genauigkeit"]

# p50 = Median: Die Hälfte der Anfragen ist schneller. p95: 95 von 100 Anfragen sind schneller.
p50 = round(statistics.median(zeiten), 1)
p95 = round(statistics.quantiles(zeiten, n=20)[-1], 1)

# --- 3. Robustheit: falsche Eingaben ------------------------------------------
gut = test.drop(columns="target").iloc[0].to_dict()
faelle = [
    ("Alkohol 99 %", "/vorhersage", {**gut, "alcohol": 99}, 422),
    ("Merkmal fehlt", "/vorhersage", {k: v for k, v in gut.items() if k != "proline"}, 422),
    ("Text statt Zahl", "/vorhersage", {**gut, "hue": "hoch"}, 422),
    ("leere Liste", "/vorhersage/batch", [], 400),
]
robust_ok = 0
for name, pfad, daten, erwartet in faelle:
    status = requests.post(f"{API_URL}{pfad}", json=daten, timeout=10).status_code
    ok = status == erwartet
    robust_ok += ok
    print(f"Robustheit: {name:16} -> {status} (erwartet {erwartet}) {'ok' if ok else 'FALSCH'}")

version = requests.get(f"{API_URL}/openapi.json", timeout=10).json()["info"]["version"]
ergebnis = {
    "api_version": version,
    "test_weine": len(test),
    "genauigkeit_api": genauigkeit_api,
    "genauigkeit_pipeline": genauigkeit_pipeline,
    "latenz_p50_ms": p50,
    "latenz_p95_ms": p95,
    "robustheit_quote": round(robust_ok / len(faelle), 2),
}

print(f"API-Version: {version}")
print(f"Genauigkeit über die API: {genauigkeit_api:.2%} (Pipeline: {genauigkeit_pipeline:.2%})")
print(f"Latenz: p50 {p50} ms, p95 {p95} ms")
print(f"Robustheit: {robust_ok} von {len(faelle)} Fällen richtig abgelehnt")

Path("ausgabe").mkdir(exist_ok=True)
Path("ausgabe/evaluation_api.json").write_text(json.dumps(ergebnis, indent=2) + "\n", encoding="utf-8")
print("Gespeichert: ausgabe/evaluation_api.json")

# --- In MLflow speichern ---------------------------------------------------------
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("wein-evaluation")
with mlflow.start_run(run_name=f"api-{version}"):
    mlflow.log_param("api_version", version)
    mlflow.log_param("test_weine", len(test))
    mlflow.log_metrics({k: v for k, v in ergebnis.items() if k not in ("api_version", "test_weine")})
    mlflow.set_tag("art", "api")
print(f"In MLflow gespeichert: Experiment wein-evaluation, Run api-{version}")
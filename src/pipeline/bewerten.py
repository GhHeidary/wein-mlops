# Datei: src/pipeline/bewerten.py – in VS Code einfügen und speichern
# Stufe 4 der Pipeline: Wir bewerten das Modell mit den Testdaten.
# Ab Tag 12 (optional): Ist MLFLOW_TRACKING_URI gesetzt, speichern wir das Ergebnis auch in MLflow.

import json
import os
import subprocess
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score

test = pd.read_csv("data/prepared/test.csv")
X_test = test.drop(columns="target")
y_test = test["target"]

# Wir laden nur Modelle, die wir selbst gespeichert haben (joblib kann Code ausführen).
modell = joblib.load("models/modell.joblib")
vorhersagen = modell.predict(X_test)

metriken = {
    "genauigkeit": round(accuracy_score(y_test, vorhersagen), 4),
    "f1_makro": round(f1_score(y_test, vorhersagen, average="macro"), 4),
    "test_weine": len(test),
}

Path("metrics.json").write_text(json.dumps(metriken, indent=2) + "\n", encoding="utf-8")
print(f"Genauigkeit: {metriken['genauigkeit']:.2%}, F1 (makro): {metriken['f1_makro']:.2%}")


def in_mlflow_speichern():
    """Speichert Parameter und Metriken als Run im MLflow-Experiment wein-pipeline."""
    # Läuft der Server nicht, wollen wir nicht minutenlang warten:
    # keine Wiederholungen, höchstens 10 Sekunden pro Anfrage.
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")
    # Wir importieren MLflow erst hier. Auf GitHub ist MLflow nicht installiert.
    import mlflow

    params = yaml.safe_load(Path("params.yaml").read_text(encoding="utf-8"))
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    # DVC setzt bei "dvc exp run" den Namen des Experiments als Umgebungsvariable.
    name = os.environ.get("DVC_EXP_NAME", "dvc-repro")

    mlflow.set_experiment("wein-pipeline")
    with mlflow.start_run(run_name=name):
        mlflow.log_params({f"vorbereiten.{k}": v for k, v in params["vorbereiten"].items()})
        mlflow.log_params({f"trainieren.{k}": v for k, v in params["trainieren"].items()})
        mlflow.log_metrics({"genauigkeit": metriken["genauigkeit"], "f1_makro": metriken["f1_makro"]})
        mlflow.set_tag("git_commit", commit)
    print(f"In MLflow gespeichert: Experiment wein-pipeline, Run {name}")


# Nur mit gesetzter Umgebungsvariable – sonst läuft die Pipeline ohne MLflow (zum Beispiel auf GitHub).
if os.environ.get("MLFLOW_TRACKING_URI"):
    try:
        in_mlflow_speichern()
    except Exception as fehler:
        # MLflow ist hier nur eine Zugabe: Die Pipeline soll trotzdem weiterlaufen.
        print(f"Hinweis: MLflow nicht erreichbar ({type(fehler).__name__}). metrics.json ist gespeichert.")
# Datei: src/pipeline/bewerten.py – in VS Code einfügen und speichern
# Stufe 4 der Pipeline: Wir bewerten das Modell mit den Testdaten.

import json
from pathlib import Path

import joblib
import pandas as pd
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
# Datei: src/pipeline/trainieren.py – in VS Code einfügen und speichern
# Stufe 3 der Pipeline: Wir trainieren das Modell und speichern es als Datei.

from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier

params = yaml.safe_load(Path("params.yaml").read_text(encoding="utf-8"))["trainieren"]

train = pd.read_csv("data/prepared/train.csv")
X_train = train.drop(columns="target")
y_train = train["target"]

modell = RandomForestClassifier(
    n_estimators=params["n_estimators"],
    max_depth=params["max_depth"],
    random_state=params["random_state"],
)
modell.fit(X_train, y_train)

ziel = Path("models/modell.joblib")
ziel.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(modell, ziel)

print(f"Modell trainiert: n_estimators={params['n_estimators']}, max_depth={params['max_depth']}")
print(f"Gespeichert: {ziel}")
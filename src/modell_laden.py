# Datei: src/modell_laden.py – in VS Code einfügen und speichern
# Wir laden den Champion aus der Model Registry und machen Vorhersagen.

import os

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

# Die Adresse kommt aus MLFLOW_TRACKING_URI. Ohne Variable nutzen wir mlflow.db.
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(TRACKING_URI)
print(f"Tracking-URI: {TRACKING_URI}")

# Wir laden das Modell über Name und Alias – keine Run-ID mehr.
modell_uri = "models:/wein-klassifikator@champion"
modell = mlflow.sklearn.load_model(modell_uri)
print(f"Modell geladen: {modell_uri}")

# Welche Version steckt gerade hinter @champion?
champion = MlflowClient().get_model_version_by_alias("wein-klassifikator", "champion")
print(f"Das ist Version {champion.version}")

# Wir nehmen dieselben Testdaten wie beim Training.
daten = load_wine(as_frame=True)
_, X_test, _, y_test = train_test_split(
    daten.data, daten.target, test_size=0.2, random_state=42, stratify=daten.target
)

vorhersagen = modell.predict(X_test.head(5))
print("-" * 40)
for nr, (vorhersage, echt) in enumerate(zip(vorhersagen, y_test.head(5)), start=1):
    ergebnis = "richtig" if vorhersage == echt else "falsch"
    print(f"Test {nr}: Vorhersage={vorhersage}, echt={echt} ({ergebnis})")
print("-" * 40)
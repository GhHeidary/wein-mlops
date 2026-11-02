# Datei: src/modell_laden.py – in VS Code einfügen und speichern
# Wir laden das beste Modell aus MLflow und machen Vorhersagen.

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

# Zuerst suchen wir den besten Run.
experiment = client.get_experiment_by_name("wein-klassifikation")
bester = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=[
        "metrics.genauigkeit DESC",
        "metrics.f1_makro DESC",
        "attributes.start_time DESC",
    ],
    max_results=1,
)[0]

# Jetzt laden wir das Modell über seine Adresse (Model URI).
modell_uri = f"runs:/{bester.info.run_id}/modell"
modell = mlflow.sklearn.load_model(modell_uri)
print(f"Modell geladen: {modell_uri}")

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
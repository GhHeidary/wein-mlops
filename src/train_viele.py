# Datei: src/train_viele.py – in VS Code einfügen und speichern
# Wir trainieren mehrere Modelle mit verschiedenen Hyperparametern.

import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# Die Adresse kommt aus MLFLOW_TRACKING_URI. Ohne Variable nutzen wir mlflow.db.
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(TRACKING_URI)
print(f"Tracking-URI: {TRACKING_URI}")
mlflow.set_experiment("wein-klassifikation")

VERTRAUTE_TYPEN = ["sklearn.tree._tree.Tree"]

daten = load_wine(as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    daten.data, daten.target, test_size=0.2, random_state=42, stratify=daten.target
)

# Diese Einstellungen wollen wir vergleichen.
hyperparameter = [
    {"n_estimators": 1, "max_depth": 1},
    {"n_estimators": 3, "max_depth": 2},
    {"n_estimators": 5, "max_depth": 3},
    {"n_estimators": 50, "max_depth": 5},
]

for hp in hyperparameter:
    # Für jede Einstellung starten wir einen eigenen Run.
    name = f"rf_n{hp['n_estimators']}_d{hp['max_depth']}"
    with mlflow.start_run(run_name=name):
        mlflow.log_param("modelltyp", "RandomForest")
        mlflow.log_params(hp)

        modell = RandomForestClassifier(**hp, random_state=42)
        modell.fit(X_train, y_train)
        y_pred = modell.predict(X_test)

        genauigkeit = accuracy_score(y_test, y_pred)
        f1_makro = f1_score(y_test, y_pred, average="macro")
        mlflow.log_metric("genauigkeit", genauigkeit)
        mlflow.log_metric("f1_makro", f1_makro)
        mlflow.sklearn.log_model(
            modell, name="modell", skops_trusted_types=VERTRAUTE_TYPEN
        )

        print(f"{name}: Genauigkeit={genauigkeit:.2%}, F1={f1_makro:.2%}")

print("Alle Modelle sind trainiert.")
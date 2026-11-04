# Datei: src/registrieren.py – in VS Code einfügen und speichern
# Wir registrieren das beste Modell in der Model Registry und wählen den Champion.

import os

import mlflow
from mlflow import MlflowClient

# Die Adresse kommt aus MLFLOW_TRACKING_URI. Ohne Variable nutzen wir mlflow.db.
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(TRACKING_URI)
print(f"Tracking-URI: {TRACKING_URI}")
client = MlflowClient()

MODELL_NAME = "wein-klassifikator"
METRIK = "genauigkeit"

# 1. Wir suchen den besten Run (wie in bestes_finden.py).
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
print(f"Bester Run: {bester.info.run_name} ({METRIK}={bester.data.metrics[METRIK]:.2%})")

# 2. Gibt es diesen Run schon als Version in der Registry?
#    Dann registrieren wir ihn nicht noch einmal.
vorhandene = client.search_model_versions(f"name='{MODELL_NAME}'")
gleiche = [v for v in vorhandene if v.run_id == bester.info.run_id]

if gleiche:
    version = gleiche[0].version
    print(f"Dieser Run ist schon registriert: Version {version}")
else:
    # Das Modell des Runs hat in MLflow 3 eine eigene ID (m-...).
    model_id = bester.outputs.model_outputs[0].model_id
    neu = mlflow.register_model(f"models:/{model_id}", MODELL_NAME)
    version = neu.version
    print(f"Neu registriert: {MODELL_NAME} Version {version}")

# 3. Wir vergleichen mit dem aktuellen Champion.
try:
    champion = client.get_model_version_by_alias(MODELL_NAME, "champion")
except mlflow.exceptions.MlflowException:
    champion = None

if champion is None:
    # Das erste Modell wird automatisch Champion.
    client.set_registered_model_alias(MODELL_NAME, "champion", version)
    print(f"Noch kein Champion. Version {version} ist jetzt @champion.")
elif champion.version == version:
    print(f"Version {version} ist schon @champion. Nichts zu tun.")
else:
    wert_champion = client.get_run(champion.run_id).data.metrics[METRIK]
    wert_neu = bester.data.metrics[METRIK]
    # Die neue Version ist zuerst nur Herausforderer.
    client.set_registered_model_alias(MODELL_NAME, "challenger", version)
    print(f"Version {version} ist jetzt @challenger.")
    print(f"Vergleich {METRIK}: Champion v{champion.version}={wert_champion:.2%}, "
          f"Challenger v{version}={wert_neu:.2%}")
    if wert_neu > wert_champion:
        client.set_registered_model_alias(MODELL_NAME, "champion", version)
        client.delete_registered_model_alias(MODELL_NAME, "challenger")
        print(f"Der Challenger ist besser. Version {version} ist jetzt @champion.")
    else:
        print(f"Der Challenger ist nicht besser. Version {champion.version} bleibt @champion.")
# Datei: src/bestes_finden.py – in VS Code einfügen und speichern
# Wir suchen das beste Modell mit Code, nicht per Hand.

import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

# Wir holen das Experiment über seinen Namen.
experiment = client.get_experiment_by_name("wein-klassifikation")

# Wir sortieren nach Genauigkeit, dann nach F1, dann nach Startzeit.
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=[
        "metrics.genauigkeit DESC",
        "metrics.f1_makro DESC",
        "attributes.start_time DESC",
    ],
    max_results=5,
)

print(f"Gefunden: {len(runs)} Runs (maximal 5 werden gezeigt)")
print("-" * 60)
for platz, run in enumerate(runs, start=1):
    name = run.info.run_name
    genauigkeit = run.data.metrics.get("genauigkeit", 0)
    f1_makro = run.data.metrics.get("f1_makro", 0)
    print(f"{platz}. {name:<15} Genauigkeit={genauigkeit:.2%}  F1={f1_makro:.2%}")
print("-" * 60)

bester = runs[0]
print("Bestes Modell:")
print(f"  Name:   {bester.info.run_name}")
print(f"  Run-ID: {bester.info.run_id}")
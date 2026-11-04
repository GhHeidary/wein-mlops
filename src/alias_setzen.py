# Datei: src/alias_setzen.py – in VS Code einfügen und speichern
# Wir setzen den Alias champion von Hand auf eine Version (zum Beispiel für einen Rollback).
# Aufruf: python src/alias_setzen.py <version>

import os
import sys

import mlflow
from mlflow import MlflowClient

# Die Adresse kommt aus MLFLOW_TRACKING_URI. Ohne Variable nutzen wir mlflow.db.
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(TRACKING_URI)
print(f"Tracking-URI: {TRACKING_URI}")
client = MlflowClient()

MODELL_NAME = "wein-klassifikator"

if len(sys.argv) != 2:
    print("Bitte eine Version angeben, zum Beispiel: python src/alias_setzen.py 1")
    sys.exit(1)

version = sys.argv[1]

# Wir prüfen, ob es die Version gibt. Sonst bricht MLflow mit einem Fehler ab.
client.get_model_version(MODELL_NAME, version)

client.set_registered_model_alias(MODELL_NAME, "champion", version)
print(f"@champion zeigt jetzt auf Version {version}.")

# Zur Kontrolle zeigen wir alle Versionen mit ihren Aliasen.
# Hinweis: search_model_versions liefert keine Aliase. Deshalb fragen wir jede Version einzeln ab.
treffer = client.search_model_versions(f"name='{MODELL_NAME}'")
for nummer in sorted(int(v.version) for v in treffer):
    v = client.get_model_version(MODELL_NAME, str(nummer))
    aliase = ", ".join(v.aliases) if v.aliases else "-"
    print(f"  Version {v.version}: Aliase = {aliase}")
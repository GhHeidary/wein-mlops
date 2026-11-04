# wein-mlops

Kursprojekt im Modul MLOps (IfaDW): Klassifikation von Weinen mit dem
Wine-Datensatz aus scikit-learn.

## Umgebung

    conda env create -f environment.yml
    conda activate mlops-kurs

## MLflow-Server starten (ab Tag 7)

    docker compose up -d --build
    export MLFLOW_TRACKING_URI=http://localhost:5000

Die MLflow UI läuft unter http://localhost:5000.

## Training

    python src/train.py

## Model Registry (ab Tag 8)

Bestes Modell registrieren und Champion wählen:

    python src/registrieren.py

Champion laden und testen:

    python src/modell_laden.py

Champion als API starten (Port 5001):

    mlflow models serve -m "models:/wein-klassifikator@champion" -p 5001 --env-manager local

API testen:

    curl -sS -X POST http://localhost:5001/invocations -H "Content-Type: application/json" -d @anfrage.json

Rollback auf eine ältere Version (danach die API neu starten):

    python src/alias_setzen.py <version>
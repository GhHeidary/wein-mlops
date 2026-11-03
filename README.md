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
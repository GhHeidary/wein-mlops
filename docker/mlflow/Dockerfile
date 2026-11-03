# Datei: docker/mlflow/Dockerfile – in VS Code einfügen und speichern
# Dockerfile für den MLflow-Server
# Wir bauen ein eigenes Image mit genau der MLflow-Version aus environment.yml.

FROM python:3.12-slim

# MLflow soll keine Nutzungsdaten an den Hersteller senden.
ENV MLFLOW_DISABLE_TELEMETRY=true \
    PIP_NO_CACHE_DIR=1

# Wir installieren MLflow mit fester Version.
RUN pip install mlflow==3.17.2

# Hier speichert der Server Datenbank und Artefakte.
# Diesen Ordner verbinden wir später mit einem Docker-Volume.
RUN mkdir -p /mlflow/artifacts
WORKDIR /mlflow

EXPOSE 5000

# Wir starten den Tracking-Server.
# --host 0.0.0.0: Der Server ist von außerhalb des Containers erreichbar.
# --backend-store-uri: Parameter, Metriken und Runs liegen in SQLite.
# --artifacts-destination: Modelle und Grafiken liegen im Ordner artifacts.
# --serve-artifacts: Der Server nimmt Artefakte an und liefert sie aus.
CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "sqlite:////mlflow/mlflow.db", \
     "--artifacts-destination", "/mlflow/artifacts", \
     "--serve-artifacts"]
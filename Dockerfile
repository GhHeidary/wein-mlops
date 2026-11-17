# Datei: Dockerfile – in VS Code einfügen und speichern
# Das Docker-Image für unsere Wein-API (ab Tag 17).
# Bauen:   docker build -t wein-api:0.3.0 .
# Starten: docker run -d --name wein-api -p 127.0.0.1:8000:8000 wein-api:0.3.0

# 1. Grundlage: ein schlankes Linux mit Python 3.12
FROM python:3.12-slim

# Python schreibt keine .pyc-Dateien und gibt Logs sofort aus.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Alle weiteren Befehle laufen im Ordner /app.
WORKDIR /app

# 2. Zuerst nur die Paketlisten kopieren und installieren.
#    Ändert sich nur der Code, nimmt Docker diese Schicht aus dem Cache.
COPY requirements-ci.txt requirements-api.txt ./
RUN pip install -r requirements-api.txt

# 3. Dann den Code und die Einstellungen kopieren.
COPY params.yaml metrics.json ./
COPY src/ src/
COPY api/ api/
COPY beispiele/ beispiele/

# 4. Das Modell beim Bauen trainieren – mit den Skripten der Pipeline (wie Render an Tag 16).
RUN python src/daten_exportieren.py \
 && python src/pipeline/validieren.py \
 && python src/pipeline/vorbereiten.py \
 && python src/pipeline/trainieren.py

# 5. Sicherheit: Die API läuft nicht als root, sondern als Benutzer "app".
RUN useradd --create-home app
USER app

# 6. Port: Standard 8000. Render setzt PORT selbst (z. B. 10000).
ENV PORT=8000
EXPOSE 8000

# 7. Docker prüft regelmäßig, ob die API antwortet.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen('http://localhost:' + os.environ['PORT'] + '/health')"

# 8. Start: "exec" ersetzt die Shell durch uvicorn. So kommt Strg + C bzw. "docker stop" direkt an.
CMD ["sh", "-c", "exec uvicorn api.app:app --host 0.0.0.0 --port ${PORT}"]
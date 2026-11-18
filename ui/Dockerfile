# Datei: ui/Dockerfile – in VS Code einfügen und speichern
# Image für die Benutzeroberfläche (Streamlit). Die UI spricht nur mit der API, sie hat kein Modell.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

WORKDIR /app

# Zuerst die Paketliste, dann der Code (Cache, siehe Tag 17).
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py .

# Wie bei der API: ein normaler Benutzer ohne root-Rechte.
RUN useradd --create-home app
USER app

EXPOSE 8501

# Streamlit hat einen eigenen Health-Endpunkt.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

# --server.address 0.0.0.0: von außerhalb des Containers erreichbar (wie bei der API).
# --browser.gatherUsageStats false: Streamlit sendet keine Nutzungsdaten.
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--browser.gatherUsageStats=false"]
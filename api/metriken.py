# Datei: api/metriken.py – in VS Code einfügen und speichern
# Ab Tag 19: Kennzahlen (Metriken) für das Monitoring mit Prometheus.
# Prometheus holt diese Zahlen regelmäßig über den Endpunkt /metrics.
#   Counter   – zählt nur nach oben (z. B. Anzahl Anfragen)
#   Histogram – sortiert Messwerte in Eimer (Buckets), z. B. Antwortzeiten.
#               Daraus berechnet Prometheus später p50 oder p95.

from prometheus_client import Counter, Histogram

# --- HTTP: die goldenen Signale (Verkehr, Fehler, Latenz) ---------------------
ANFRAGEN = Counter(
    "wein_http_anfragen_total",
    "Anzahl HTTP-Anfragen",
    ["pfad", "methode", "status"],
)
LATENZ = Histogram(
    "wein_http_latenz_sekunden",
    "Antwortzeit der API in Sekunden",
    ["pfad"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120],
)

# --- ML: Was sagt das Modell voraus, und wie sicher ist es? -------------------
VORHERSAGEN = Counter(
    "wein_vorhersagen_total",
    "Anzahl Vorhersagen pro Sorte",
    ["sorte"],
)
SICHERHEIT = Histogram(
    "wein_vorhersage_sicherheit",
    "Wahrscheinlichkeit der vorhergesagten Sorte",
    buckets=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
)

# --- LLM: Wie geht es dem Sprachmodell? --------------------------------------
LLM_ANFRAGEN = Counter(
    "wein_llm_anfragen_total",
    "Anfragen an das LLM nach Ergebnis",
    ["ergebnis"],  # ok, fallback
)
LLM_FEHLER = Counter(
    "wein_llm_fehler_total",
    "Fehler des LLM nach Art",
    ["art"],  # z. B. APIConnectionError, APITimeoutError, RateLimitError
)
LLM_LATENZ = Histogram(
    "wein_llm_latenz_sekunden",
    "Antwortzeit des LLM in Sekunden",
    buckets=[0.5, 1, 2, 5, 10, 20, 30, 60, 120],
)
LLM_TOKENS = Counter(
    "wein_llm_tokens_total",
    "Verbrauchte Tokens des LLM",
)
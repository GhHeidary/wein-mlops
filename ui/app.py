# Datei: ui/app.py – in VS Code einfügen und speichern
# Unsere Benutzeroberfläche mit Streamlit (ab Tag 18).
# Die UI rechnet nichts selbst. Sie schickt alles an die Wein-API und zeigt die Antwort.

import os

import pandas as pd
import requests
import streamlit as st

# Im Compose-Stack heißt die API einfach "api". Docker findet den Container über diesen Namen.
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Wein-Erkennung")
st.title("Wein-Erkennung")
st.caption(f"API: {API_URL}")


def api_fragen(pfad, wein=None, timeout=10):
    """Schickt eine Anfrage an die API. Gibt (Statuscode, Daten) zurück."""
    try:
        if wein is None:
            antwort = requests.get(f"{API_URL}{pfad}", timeout=timeout)
        else:
            antwort = requests.post(f"{API_URL}{pfad}", json=wein, timeout=timeout)
    except requests.RequestException as fehler:
        return None, f"Die API ist nicht erreichbar ({type(fehler).__name__})."
    return antwort.status_code, antwort.json()


# Beispielweine von der API holen (Datei beispiele/neue_weine.csv).
status, weine = api_fragen("/beispiele?anzahl=5")
if status != 200:
    st.error(weine)
    st.stop()

nummer = st.selectbox("Wein auswählen", range(len(weine)), format_func=lambda i: f"Wein {i + 1}")
wein = weine[nummer]
st.dataframe(pd.DataFrame([wein]).T.rename(columns={0: "Wert"}))

if st.button("Sorte vorhersagen"):
    status, daten = api_fragen("/vorhersage", wein)
    if status == 200:
        st.success(f"{daten['name']} – Sicherheit {daten['wahrscheinlichkeit']:.0%}")
    else:
        st.error(f"Fehler {status}: {daten}")

if st.button("Erklärung vom LLM"):
    # Ein lokales LLM braucht auf der CPU manchmal eine Minute. Deshalb warten wir länger.
    with st.spinner("Das LLM schreibt die Erklärung ..."):
        status, daten = api_fragen("/erklaerung", wein, timeout=180)
    if status == 200:
        st.success(f"{daten['vorhersage']['name']} – Sicherheit {daten['vorhersage']['wahrscheinlichkeit']:.0%}")
        st.write(daten["text"])
        st.caption(
            f"LLM: {daten['llm_modell']}, Prompt {daten['prompt_version']}, "
            f"{daten['tokens']} Tokens, {daten['sekunden']} s"
        )
    elif status is None:
        st.error(daten)
    else:
        st.error(f"Fehler {status}: {daten.get('detail', daten)}")
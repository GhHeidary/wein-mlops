# Datei: api/llm.py – in VS Code einfügen und speichern
# Ab Tag 18: Ein Sprachmodell (LLM) erklärt eine Vorhersage in einfachen Worten.
# Die Einstellungen kommen aus Umgebungsvariablen. So wechseln wir das LLM ohne Code-Änderung:
#   Ollama im Compose-Stack:  LLM_BASE_URL=http://ollama:11434/v1   LLM_MODELL=llama3.2:3b
#   Groq (Cloud, mit Key):    LLM_BASE_URL=https://api.groq.com/openai/v1   LLM_MODELL=llama-3.1-8b-instant
# Beide Dienste sprechen dieselbe Schnittstelle wie OpenAI. Deshalb nutzen wir das Paket openai.

import os
import time

from openai import OpenAI

# Wir merken uns, welcher Prompt eine Antwort erzeugt hat (wichtig für die Evaluation).
PROMPT_VERSION = "v2"

# v2 (Tag 19): v1 hat oft Rebsorten wie "Pinot Noir" erfunden (Evaluation Tag 18).
SYSTEM_PROMPT = (
    "Du bist ein Assistent in einem Weinlabor. "
    "Ein Machine-Learning-Modell ordnet Weine einer von drei Klassen zu: Sorte 1, Sorte 2 oder Sorte 3. "
    "Die echten Namen der Rebsorten sind unbekannt. Nenne deshalb nie eine Rebsorte wie Merlot, Riesling oder Pinot Noir. "
    "Erkläre die Vorhersage für Laien auf Deutsch in höchstens drei kurzen Sätzen. "
    "Nenne die vorhergesagte Sorte genau so, wie sie im Text steht, zum Beispiel 'Sorte 1'. "
    "Nutze nur die Werte aus dem Text und erfinde keine Messwerte."
)


def einstellungen() -> dict:
    """Liest die Einstellungen bei jedem Aufruf neu (so können Tests sie ändern)."""
    return {
        "base_url": os.environ.get("LLM_BASE_URL", ""),
        "modell": os.environ.get("LLM_MODELL", "llama3.2:3b"),
        # Ollama braucht keinen Key. Das Paket openai verlangt aber einen Text.
        "api_key": os.environ.get("LLM_API_KEY", "ollama"),
    }


def ist_eingerichtet() -> bool:
    return bool(einstellungen()["base_url"])


def prompt_bauen(sorte: str, wahrscheinlichkeit: float, merkmale: dict) -> str:
    zeilen = [f"- {name}: {wert}" for name, wert in merkmale.items()]
    return (
        f"Das Modell sagt voraus: {sorte} (Sicherheit {wahrscheinlichkeit:.0%}).\n"
        "Die wichtigsten Merkmale für das Modell und die Werte dieses Weins:\n"
        + "\n".join(zeilen)
        + "\nErkläre kurz, warum das Modell wahrscheinlich diese Sorte gewählt hat."
    )


def erklaeren(sorte: str, wahrscheinlichkeit: float, merkmale: dict) -> dict:
    """Fragt das LLM und gibt Text, Modell, Prompt-Version, Tokens und Dauer zurück."""
    e = einstellungen()
    # timeout: Ein lokales LLM auf der CPU braucht manchmal 30 Sekunden und mehr.
    client = OpenAI(base_url=e["base_url"], api_key=e["api_key"], timeout=120, max_retries=0)
    start = time.perf_counter()
    antwort = client.chat.completions.create(
        model=e["modell"],
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_bauen(sorte, wahrscheinlichkeit, merkmale)},
        ],
        # temperature=0: möglichst gleiche Antworten bei gleicher Frage (wichtig für Tests und Evaluation).
        temperature=0,
        max_tokens=200,
    )
    return {
        "text": antwort.choices[0].message.content.strip(),
        "llm_modell": e["modell"],
        "prompt_version": PROMPT_VERSION,
        "tokens": antwort.usage.total_tokens if antwort.usage else None,
        "sekunden": round(time.perf_counter() - start, 2),
    }
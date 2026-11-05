# Datei: src/pipeline/vorbereiten.py – in VS Code einfügen und speichern
# Stufe 2 der Pipeline: Wir teilen die Daten in Training und Test.

from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

# Die Parameter stehen in params.yaml, nicht im Code.
params = yaml.safe_load(Path("params.yaml").read_text(encoding="utf-8"))["vorbereiten"]

daten = pd.read_csv("data/raw/wein.csv")
train, test = train_test_split(
    daten,
    test_size=params["test_size"],
    random_state=params["random_state"],
    stratify=daten["target"],
)

ziel = Path("data/prepared")
ziel.mkdir(parents=True, exist_ok=True)
train.to_csv(ziel / "train.csv", index=False)
test.to_csv(ziel / "test.csv", index=False)

print(f"Training: {len(train)} Weine, Test: {len(test)} Weine")
# Datei: src/train.py – in VS Code einfügen und speichern
# Wir trainieren ein Modell für den Wein-Datensatz und speichern alles in MLflow.

import matplotlib
matplotlib.use("Agg")  # Wir zeichnen Grafiken ohne Fenster. Das reicht für MLflow.
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# 1. MLflow einrichten
# Wir speichern die Daten von MLflow in der Datei mlflow.db im Projektordner.
mlflow.set_tracking_uri("sqlite:///mlflow.db")
# Alle Läufe dieses Projekts landen im selben Experiment.
mlflow.set_experiment("wein-klassifikation")

# 2. Hyperparameter festlegen
# Diese Werte ändern wir später in den Übungen.
N_ESTIMATORS = 10
MAX_DEPTH = 3
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Diesen Typ braucht MLflow, um ein RandomForest-Modell sicher zu speichern.
# Wir vertrauen ihm, weil wir das Modell selbst trainieren.
VERTRAUTE_TYPEN = ["sklearn.tree._tree.Tree"]

# 3. Daten laden und aufteilen
daten = load_wine(as_frame=True)
X = daten.data
y = daten.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# 4. Training mit MLflow-Tracking
with mlflow.start_run(run_name=f"rf_n{N_ESTIMATORS}_d{MAX_DEPTH}") as run:
    # Hier speichern wir die Parameter.
    mlflow.log_param("modelltyp", "RandomForest")
    mlflow.log_param("n_estimators", N_ESTIMATORS)
    mlflow.log_param("max_depth", MAX_DEPTH)
    mlflow.log_param("test_size", TEST_SIZE)
    mlflow.log_param("random_state", RANDOM_STATE)

    # Jetzt trainieren wir das Modell.
    modell = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE,
    )
    modell.fit(X_train, y_train)

    # Wir prüfen das Modell mit den Testdaten.
    y_pred = modell.predict(X_test)
    genauigkeit = accuracy_score(y_test, y_pred)
    f1_makro = f1_score(y_test, y_pred, average="macro")

    # Hier speichern wir die Metriken.
    mlflow.log_metric("genauigkeit", genauigkeit)
    mlflow.log_metric("f1_makro", f1_makro)

    # Wir speichern eine Grafik als Artefakt: die Konfusionsmatrix.
    anzeige = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    anzeige.ax_.set_title("Konfusionsmatrix")
    mlflow.log_figure(anzeige.figure_, "konfusionsmatrix.png")
    plt.close(anzeige.figure_)

    # Zum Schluss speichern wir das Modell selbst.
    mlflow.sklearn.log_model(
        modell,
        name="modell",
        skops_trusted_types=VERTRAUTE_TYPEN,
    )

    print(f"Run-ID: {run.info.run_id}")
    print(f"Genauigkeit: {genauigkeit:.2%}")
    print(f"F1 (makro): {f1_makro:.2%}")
    print("Der Lauf ist in MLflow gespeichert.")
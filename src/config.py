"""Putanje i osnovna podesavanja projekta."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = ROOT_DIR / "data" / "data.csv"

OUTPUTS_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"
MODELS_DIR = OUTPUTS_DIR / "models"

TARGET = "Survived"
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# Kategorije (titule/palube) sa manje od ovoliko clanova presavijaju se u
# zbirnu kategoriju (Rare / U), da se izbegnu kategorije sa par primera.
MIN_CATEGORY = 5


def ensure_dirs():
    for d in (FIGURES_DIR, METRICS_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)

"""Configuration defaults and runtime paths for the manuscript code."""

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
FIGURES_DIR = ROOT_DIR / "figures"
OUTPUTS_DIR = ROOT_DIR / "outputs"
RESULTS_DIR = OUTPUTS_DIR / "results"
TEMP_GRIDS_DIR = OUTPUTS_DIR / "temp_grids"

# Core defaults shared by train/predict
DEFAULT_SEED = 42
DEFAULT_SPLIT_SEED = 11111
DEFAULT_TRAIN_RATIO = 0.6
DEFAULT_VAL_RATIO = 0.1
DEFAULT_MIN_SAMPLES = 1000
DEFAULT_K = 50
DEFAULT_SIGMA_KM = 2.0
DEFAULT_HIDDEN_DIM = 128
DEFAULT_LAYERS_COUNT = 2
DEFAULT_EPOCHS = 1000
DEFAULT_PATIENCE = 50
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_MODEL_NAME = "ours"
DEFAULT_CHUNK_SIZE = 100000
DEFAULT_GRID_SIZE = 0.5
DEFAULT_BUFFER_CELLS = 1
DEFAULT_UNKNOWN_THRESHOLD = 0.9
DEFAULT_UNKNOWN_ID = -1


def ensure_runtime_dirs() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_GRIDS_DIR.mkdir(parents=True, exist_ok=True)

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

# --------------------
# Data locations
# --------------------

DATA_ROOT = PROJECT_ROOT / "data" / "wlasl"

# Expected WLASL layout:
# DATA_ROOT /
#   WLASL_v0.3.json
#   videos/
#   keypoints/
#   metadata.csv

WLASL_JSON = DATA_ROOT / "WLASL_v0.3.json"
VIDEO_DIR = DATA_ROOT / "videos"
KEYPOINT_DIR = DATA_ROOT / "keypoints"
METADATA_PATH = DATA_ROOT / "metadata.csv"


# --------------------
# Dataset / subset config
# --------------------

# Use a small subset first for fast iteration
WLASL_SUBSET = "wlasl100"  # or "wlasl300", "wlasl1000"

# Number of glosses to use (will be inferred from metadata if set to None)
NUM_GLOSSES = None


# --------------------
# Model hyperparameters
# --------------------

INPUT_DIM = 126  # 21 landmarks * 2 hands * 3 coords
EMBED_DIM = 64
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.3
BIDIRECTIONAL = True


# --------------------
# Training hyperparameters
# --------------------

BATCH_SIZE = 16
LR = 1e-3
NUM_EPOCHS = 20
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 5.0
NUM_WORKERS = 4


# --------------------
# Misc
# --------------------

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pt"

DEVICE = "cuda"  # will fall back to cpu at runtime if needed


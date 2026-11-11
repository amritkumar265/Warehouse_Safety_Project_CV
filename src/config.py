# src/config.py
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
VIDEOS_DIR = PROJECT_ROOT / "videos"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Standardized Taxonomy for the entire pipeline
CLASS_MAPPING = {
    0: "person",
    1: "forklift",
    2: "pallet_truck"
}

# Detection parameters
YOLO_CONF_THRESH = 0.25
DINO_BOX_THRESH = 0.35
DINO_TEXT_THRESH = 0.25
NMS_IOU_THRESH = 0.50
DINO_PROMPT = "person. forklift. pallet truck. pallet jack. manual pallet jack."
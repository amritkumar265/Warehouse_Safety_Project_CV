import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CAMERAS_DIR = PROJECT_ROOT / "configs" / "cameras"

CAMERAS_DIR.mkdir(parents= True, exist_ok= True)

def get_camera_dir(camera_id : str) -> Path:

    camera_id = camera_id.strip()

    if not camera_id or Path(camera_id).name != camera_id:
        raise ValueError("Invalid camera ID")

    return CAMERAS_DIR/ camera_id

def load_profile(camera_id : str):
    profile_path = get_camera_dir(camera_id) / "profile.json"

    if not profile_path.exists():
        return None

    with open(profile_path , "r" , encoding="utf-8") as file:
        return json.load(file)


def save_profile(profile : dict):
    camera_id = profile.get("camera_id")

    if not camera_id:
        raise ValueError("Camera ID is missing from profile.")

    camera_dir = get_camera_dir(camera_id)
    camera_dir.mkdir(parents=True, exist_ok=True)

    profile_path = camera_dir / "profile.json"

    with open(profile_path, "w", encoding="utf-8") as file:
        json.dump(profile, file, indent=4)


def get_registered_cameras() -> list:
    """Return camera IDs having a saved profile."""

    if not CAMERAS_DIR.exists():
        return []

    return sorted(
        directory.name
        for directory in CAMERAS_DIR.iterdir()
        if directory.is_dir()
        and (directory / "profile.json").exists()
    )



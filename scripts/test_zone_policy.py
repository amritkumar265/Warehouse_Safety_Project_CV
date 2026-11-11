
import json
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.zones.zone_engine import ZoneEngine
from core.zones.zone_policy import ZonePolicyEngine


PROFILE_PATH = (
    PROJECT_ROOT
    / "configs"
    / "cameras"
    / "Camera_001"
    / "profile.json"
)

VIDEO_PATH = (
    PROJECT_ROOT
    / "videos"
    / "camera001"
    / "warehouse2.mp4"
)


def main():

    with open(PROFILE_PATH, "r") as file:
        profile = json.load(file)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    success, frame = cap.read()
    cap.release()

    if not success:
        raise RuntimeError("Could not read test frame.")

    zone_engine = ZoneEngine(profile)
    policy_engine = ZonePolicyEngine(
        suppression_threshold=0.70
    )

    test_detections = [
        {
            "class_name": "person",
            "bbox": [1250, 250, 1300, 450]
        },
        {
            "class_name": "forklift",
            "bbox": [1250, 400, 1450, 800]
        },
        {
            "class_name": "pallet_truck",
            "bbox": [100, 400, 300, 700]
        }
    ]

    print("\nZONE POLICY TEST RESULTS\n")

    for detection in test_detections:

        association = zone_engine.associate_detection(
            bbox=detection["bbox"],
            class_name=detection["class_name"],
            frame_shape=frame.shape
        )

        decision = policy_engine.evaluate(association)

        print("--------------------------------")
        print("Class:", detection["class_name"])
        print("Primary zone:", association["primary_zone"])
        print("Coverage:", association["zone_results"])
        print("Decision:", decision["decision"])
        print("Reason:", decision["reason"])


if __name__ == "__main__":
    main()

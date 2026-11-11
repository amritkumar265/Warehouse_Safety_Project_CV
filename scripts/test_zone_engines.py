import json
from pathlib import Path

import cv2
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.zones.zone_engine import ZoneEngine

PROFILE_PATH = (
    PROJECT_ROOT
    /"configs"
    /"cameras"
    /"camera_001"
    /"profile.json"
)

VIDEO_PATH = (PROJECT_ROOT / "videos" / "camera001" / "warehouse2.mp4")

def main():
    with open(PROFILE_PATH , "r") as file:
        profile = json.load(file)

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    success , frame = cap.read()

    cap.release()

    if not success:
        raise RuntimeError("Could not read test video frame. ")

    engine = ZoneEngine(profile)

    height , width = frame.shape[ : 2]

    print("Frame resolution:", width, "x", height)

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

    print("\nZONE ASSOCIATION RESULTS\n")

    for detection in test_detections:

        result = engine.associate_detection(
            bbox=detection["bbox"],
            class_name=detection["class_name"],
            frame_shape=frame.shape
        )

        print("--------------------------------")
        print("Class:", result["class_name"])
        print("BBox:", result["bbox"])
        print("Bottom centre:", result["bottom_center"])
        print("Primary zone:", result["primary_zone"])
        print("All containing zones:", result["containing_zones"])

        print("Coverage by zone:")

        for zone, values in result["zone_results"].items():

            print(
                f"  {zone}: "
                f"bottom_center={values['contains_bottom_center']}, "
                f"coverage={values['bbox_coverage']:.2%}"
            )


if __name__ == "__main__":
    main()





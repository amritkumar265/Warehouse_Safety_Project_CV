
import sys
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.detection_pipeline import DetectionPipeline
from core.camera.profile_manager import load_profile


VIDEO_PATH = PROJECT_ROOT / "videos" / "camera001" / "warehouse2.mp4"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pipeline_test"
OUTPUT_DIR.mkdir(parents = True , exist_ok=True)

CAMERA_ID = "camera_001"
MAX_FRAMES = 10

profile =load_profile(CAMERA_ID)

if profile is None:
    raise RuntimeError("Camera profile not found")

pipeline = DetectionPipeline(
    camera_profile=profile
)

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError(f"Cannot Open Video: {VIDEO_PATH}")

frame_idx = 0

try:
    while frame_idx < MAX_FRAMES:
        success , frame = cap.read()

        if not success:
            break


        results = pipeline.process_frame(frame)

        if frame_idx == 0:
            print("\n========== FRAME 1 ZONE AUDIT ==========")

        for record in results["audit_records"]:
            print(record)

        print("========================================\n")

        annotated = results["annotated_frame"]

        output_path = OUTPUT_DIR / f"frame_{frame_idx:03d}.jpg"

        cv2.imwrite(str(output_path) , annotated)

        print(
            f"Frame {frame_idx + 1}: "
            f"Raw={len(results['raw_detections'])}, "
            f"Retained={len(results['retained_detections'])}, "
            f"Suppressed={len(results['suppressed_candidates'])}, "
            f"Flags={len(results['safety_flags'])}"
        )

        frame_idx += 1

finally:
    cap.release()

print(f"\nProcessed frames: {frame_idx}")
print(f"Annotated frames saved in: {OUTPUT_DIR}")

print("Pipeline frame test completed.")

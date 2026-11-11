# src/yolo_detector.py
from ultralytics import YOLO
import supervision as sv
import numpy as np

class YoloDetector:
    def __init__(self, model_path: str, conf: float = 0.25, imgsz: int = 800):
        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz
        self.names = self.model.names  # e.g., {0: 'person', 1: 'forklift', ...}

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """Runs YOLO on a single frame and maps detections to unified classes."""
        results = self.model(frame, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        raw_det = sv.Detections.from_ultralytics(results)
        
        if len(raw_det) == 0:
            return sv.Detections.empty()

        # Map whatever class names this model uses to unified indices (0, 1, 2)
        name_to_unified = {"person": 0, "forklift": 1, "pallet_truck": 2}
        keep_indices = []
        unified_class_ids = []

        for idx, cid in enumerate(raw_det.class_id):
            label = self.names[int(cid)]
            if label in name_to_unified:
                keep_indices.append(idx)
                unified_class_ids.append(name_to_unified[label])

        if not keep_indices:
            return sv.Detections.empty()

        keep_indices = np.array(keep_indices)
        return sv.Detections(
            xyxy=raw_det.xyxy[keep_indices],
            confidence=raw_det.confidence[keep_indices],
            class_id=np.array(unified_class_ids)
        )
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import supervision as sv
import yaml
from ultralytics.trackers.byte_tracker import BYTETracker


class _ClassAwareBYTETracker(BYTETracker):
    def get_dists(self, tracks, detections):
        distances = super().get_dists(tracks, detections)
        for track_index, track in enumerate(tracks):
            for detection_index, detection in enumerate(detections):
                if track.cls != detection.cls:
                    distances[track_index, detection_index] = 1e6
        return distances


class _ByteTrackInput:
    def __init__(self, detections):
        self.conf = (
            np.asarray(detections.confidence, dtype=np.float32)
            if detections.confidence is not None
            else np.zeros(len(detections), dtype=np.float32)
        )
        self.cls = (
            np.asarray(detections.class_id, dtype=np.float32)
            if detections.class_id is not None
            else np.zeros(len(detections), dtype=np.float32)
        )

        if len(detections):
            x1, y1, x2, y2 = detections.xyxy.T
            self.xywh = np.column_stack((
                (x1 + x2) / 2,
                (y1 + y2) / 2,
                x2 - x1,
                y2 - y1
            )).astype(np.float32)
        else:
            self.xywh = np.empty((0, 4), dtype=np.float32)

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, indices):
        result = object.__new__(type(self))
        result.conf = self.conf[indices]
        result.cls = self.cls[indices]
        result.xywh = self.xywh[indices]
        return result


class ByteTrackAdapter:
    """Apply Ultralytics ByteTrack to boxes from any detector."""

    def __init__(self, config_path):
        config_path = Path(config_path)
        if not config_path.is_file():
            raise FileNotFoundError(
                f"ByteTrack configuration not found: {config_path}"
            )

        with config_path.open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
        if not isinstance(config, dict):
            raise ValueError(
                f"Invalid ByteTrack configuration: {config_path}"
            )

        self.tracker = _ClassAwareBYTETracker(SimpleNamespace(**config))

    def update(self, detections: sv.Detections):
        rows = self.tracker.update(_ByteTrackInput(detections))
        if rows.size == 0:
            return sv.Detections.empty()
        if rows.ndim != 2 or rows.shape[1] < 8:
            raise RuntimeError(
                "ByteTrack returned an unexpected tracked-box format."
            )

        detection_indices = rows[:, 7].astype(np.int64)
        if np.any(detection_indices < 0) or np.any(
            detection_indices >= len(detections)
        ):
            raise RuntimeError(
                "ByteTrack returned a box index outside the current detections."
            )

        tracked = detections[detection_indices]
        return sv.Detections(
            xyxy=tracked.xyxy,
            confidence=tracked.confidence,
            class_id=tracked.class_id,
            tracker_id=rows[:, 4].astype(np.int64),
            data=tracked.data
        )

    def reset(self):
        self.tracker.reset()

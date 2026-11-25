from pathlib import Path

import numpy as np
import pytest
import supervision as sv

from core.tracking.byte_tracker import ByteTrackAdapter


def _detections(boxes, class_ids, class_names, confidences=None):
    if confidences is None:
        confidences = [0.9] * len(boxes)
    return sv.Detections(
        xyxy=np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
        confidence=np.asarray(confidences, dtype=np.float32),
        class_id=np.asarray(class_ids, dtype=np.int64),
        data={"class_name": np.asarray(class_names, dtype=object)}
    )


def test_tracker_assigns_persistent_ids_to_generic_detector_boxes():
    config_path = (
        Path(__file__).resolve().parents[1] / "configs" / "bytetrack.yaml"
    )
    tracker = ByteTrackAdapter(config_path)
    first = _detections(
        [[10, 10, 50, 70], [100, 10, 140, 70]],
        [1, 0],
        ["forklift", "person"]
    )
    tracked_first = tracker.update(first)
    first_ids = {
        name: int(track_id)
        for name, track_id in zip(
            tracked_first.data["class_name"],
            tracked_first.tracker_id
        )
    }

    assert set(first_ids) == {"forklift", "person"}

    tracker.update(sv.Detections.empty())
    second = _detections(
        [[12, 10, 52, 70], [102, 10, 142, 70]],
        [1, 0],
        ["forklift", "person"]
    )
    tracked_second = tracker.update(second)
    second_ids = {
        name: int(track_id)
        for name, track_id in zip(
            tracked_second.data["class_name"],
            tracked_second.tracker_id
        )
    }

    assert second_ids == first_ids


def test_tracker_handles_empty_frames_and_preserves_detector_class_names():
    config_path = (
        Path(__file__).resolve().parents[1] / "configs" / "bytetrack.yaml"
    )
    tracker = ByteTrackAdapter(config_path)

    assert len(tracker.update(sv.Detections.empty())) == 0
    tracker.reset()
    tracked = tracker.update(_detections(
        [[0, 0, 20, 40]],
        [17],
        ["safety_cone"]
    ))

    assert tracked.data["class_name"].tolist() == ["safety_cone"]
    assert tracked.tracker_id is not None


def test_tracker_keeps_ids_associated_with_detector_classes():
    config_path = (
        Path(__file__).resolve().parents[1] / "configs" / "bytetrack.yaml"
    )
    tracker = ByteTrackAdapter(config_path)
    overlapping_box = [[10, 10, 50, 70], [10, 10, 50, 70]]
    first = tracker.update(_detections(
        overlapping_box,
        [0, 1],
        ["person", "forklift"]
    ))
    first_ids = dict(zip(
        first.data["class_name"].tolist(),
        first.tracker_id.tolist()
    ))

    second = tracker.update(_detections(
        overlapping_box,
        [1, 0],
        ["forklift", "person"]
    ))
    second_ids = dict(zip(
        second.data["class_name"].tolist(),
        second.tracker_id.tolist()
    ))

    assert second_ids == first_ids


def test_tracker_rejects_missing_config():
    with pytest.raises(FileNotFoundError, match="ByteTrack configuration not found"):
        ByteTrackAdapter("/missing/bytetrack.yaml")

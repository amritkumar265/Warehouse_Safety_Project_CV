import csv
import json

import numpy as np
import supervision as sv

from core.tracking.event_logger import TrackingEventLogger
from core.tracking.proximity import (
    RelativeMotionAnalyzer,
    calculate_bbox_distance,
    calculate_center_distance
)
from core.tracking.safety_events import SafetyEventManager
from core.tracking.trajectory import TrajectoryManager


def test_center_and_bounding_box_distances():
    assert calculate_center_distance((0, 0), (3, 4)) == 5
    assert calculate_bbox_distance(
        (0, 0, 10, 10),
        (13, 14, 20, 20)
    ) == 5
    assert calculate_bbox_distance(
        (0, 0, 10, 10),
        (8, 8, 20, 20)
    ) == 0


def test_motion_state_requires_multiple_consistent_observations():
    analyzer = RelativeMotionAnalyzer(
        smoothing_window=1,
        state_change_observations=3,
        delta_threshold_px=0.5
    )

    states = []
    for frame, distance in enumerate((100, 95, 90, 85)):
        metrics = analyzer.update(
            (1, 2),
            distance,
            frame_index=frame,
            timestamp_seconds=float(frame)
        )
        states.append(metrics["approach_state"])

    assert states == [
        "MAINTAINING",
        "MAINTAINING",
        "MAINTAINING",
        "APPROACHING"
    ]
    assert metrics["relative_approach_speed_px_s"] == 5


def test_motion_state_requires_consecutive_observations_after_gap():
    analyzer = RelativeMotionAnalyzer(
        smoothing_window=1,
        state_change_observations=3,
        delta_threshold_px=0.5
    )
    analyzer.update((1, 2), 100, 0, 0.0)
    analyzer.update((1, 2), 95, 1, 1.0)
    analyzer.update((1, 2), 90, 3, 3.0)

    result = analyzer.update((1, 2), 85, 4, 4.0)
    assert result["approach_state"] == "MAINTAINING"


def test_trajectory_history_is_bounded_and_inactive_tracks_are_counted():
    manager = TrajectoryManager(
        history_length=2,
        cleanup_after_frames=1
    )
    detections = sv.Detections(
        xyxy=np.array([[0, 0, 10, 20]], dtype=float),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
        tracker_id=np.array([7])
    )

    for frame in range(3):
        observations, history = manager.update(detections, frame, fps=10)

    assert len(history[7]) == 2
    assert observations[0].center == (5.0, 10.0)
    assert observations[0].bottom_center == (5.0, 20.0)
    empty = sv.Detections.empty()
    manager.update(empty, frame_index=5, fps=10)
    summary = manager.finalize(fps=10)
    assert summary["unique_tracked_people"] == 1
    assert 0 < summary["maximum_track_duration_seconds"] < 1


def test_alerts_are_transition_based_and_respect_cooldown():
    manager = SafetyEventManager(cooldown_seconds=5)

    def metrics(timestamp, level, approach):
        return {
            "timestamp_seconds": timestamp,
            "person_track_id": 1,
            "vehicle_track_id": 2,
            "proximity_level": level,
            "approach_state": approach
        }

    alerts = manager.update(metrics(0, "CAUTION", "APPROACHING"), 0)
    assert {event["alert_type"] for event in alerts} == {
        "CAUTION",
        "APPROACHING",
        "HIGH_RISK"
    }
    assert manager.update(metrics(1, "CAUTION", "APPROACHING"), 1) == []

    manager.update(metrics(2, "SAFE", "MAINTAINING"), 2)
    assert manager.update(metrics(3, "CAUTION", "APPROACHING"), 3) == []


def test_tracking_event_csv_and_summary_json(tmp_path):
    csv_path = tmp_path / "events.csv"
    summary_path = tmp_path / "summary.json"
    logger = TrackingEventLogger(csv_path)
    logger.write_events([{
        "timestamp": "00:00:01.000",
        "frame_number": 25,
        "person_track_id": 3,
        "vehicle_track_id": 4,
        "vehicle_class": "forklift",
        "center_distance_px": 100.0,
        "bbox_distance_px": 40.0,
        "relative_approach_speed_px_s": 2.0,
        "approach_state": "APPROACHING",
        "proximity_level": "DANGER",
        "alert_type": "HIGH_RISK"
    }])
    logger.close()
    TrackingEventLogger.write_summary(summary_path, {"frames": 25})

    with csv_path.open(newline="", encoding="utf-8") as event_file:
        rows = list(csv.DictReader(event_file))
    assert rows[0]["person_track_id"] == "3"
    assert rows[0]["alert_type"] == "HIGH_RISK"
    assert json.loads(summary_path.read_text()) == {"frames": 25}

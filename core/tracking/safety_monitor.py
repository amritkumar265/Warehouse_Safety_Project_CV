from core.tracking.proximity import (
    RelativeMotionAnalyzer,
    calculate_bbox_distance,
    calculate_center_distance
)
from core.tracking.safety_events import SafetyEventManager
from core.tracking.trajectory import TrajectoryManager


class SafetyMonitor:
    def __init__(self, config):
        proximity_config = config["proximity"]
        self.caution_distance_px = float(proximity_config["caution_px"])
        self.danger_distance_px = float(proximity_config["danger_px"])

        if (
            self.danger_distance_px <= 0
            or self.caution_distance_px <= self.danger_distance_px
        ):
            raise ValueError(
                "Proximity thresholds must satisfy 0 < danger_px < caution_px."
            )

        cleanup_after_frames = int(config["track_cleanup_after_frames"])
        self.trajectories = TrajectoryManager(
            history_length=int(config["trajectory_history_length"]),
            cleanup_after_frames=cleanup_after_frames
        )
        self.motion = RelativeMotionAnalyzer(
            smoothing_window=int(config["temporal_smoothing_window"]),
            state_change_observations=int(
                config["approach_state_observations"]
            ),
            delta_threshold_px=float(
                config["approach_delta_threshold_px"]
            ),
            cleanup_after_frames=cleanup_after_frames
        )
        self.events = SafetyEventManager(
            cooldown_seconds=float(config["alert_cooldown_seconds"]),
            cleanup_after_frames=cleanup_after_frames
        )

    def update(self, detections, frame_index: int, fps: float):
        observations, trajectory_history = self.trajectories.update(
            detections,
            frame_index,
            fps
        )
        people = [
            item for item in observations
            if item.object_class == "person"
        ]
        vehicles = [
            item for item in observations
            if item.object_class in {"forklift", "pallet_truck"}
        ]

        pair_metrics = []
        events = []

        for person in people:
            for vehicle in vehicles:
                pair_key = (person.track_id, vehicle.track_id)

                center_distance = calculate_center_distance(
                    person.center,
                    vehicle.center
                )
                bbox_distance = calculate_bbox_distance(
                    person.bbox,
                    vehicle.bbox
                )
                motion = self.motion.update(
                    pair_key=pair_key,
                    bbox_distance_px=bbox_distance,
                    frame_index=frame_index,
                    timestamp_seconds=person.timestamp_seconds
                )
                level_distance = motion["smoothed_bbox_distance_px"]
                if level_distance < self.danger_distance_px:
                    proximity_level = "DANGER"
                elif level_distance <= self.caution_distance_px:
                    proximity_level = "CAUTION"
                else:
                    proximity_level = "SAFE"

                high_risk = (
                    motion["approach_state"] == "APPROACHING"
                    and proximity_level in {"CAUTION", "DANGER"}
                )
                metrics = {
                    "timestamp_seconds": person.timestamp_seconds,
                    "frame_number": frame_index + 1,
                    "person_track_id": person.track_id,
                    "vehicle_track_id": vehicle.track_id,
                    "vehicle_class": vehicle.object_class,
                    "center_distance_px": center_distance,
                    "bbox_distance_px": bbox_distance,
                    "smoothed_bbox_distance_px": level_distance,
                    "distance_change_px": motion["distance_change_px"],
                    "relative_approach_speed_px_s": (
                        motion["relative_approach_speed_px_s"]
                    ),
                    "approach_state": motion["approach_state"],
                    "approach_state_changed": (
                        motion["approach_state_changed"]
                    ),
                    "proximity_level": proximity_level,
                    "high_risk": high_risk,
                    "person_center": person.center,
                    "vehicle_center": vehicle.center
                }
                pair_metrics.append(metrics)
                events.extend(
                    self.events.update(metrics, frame_index)
                )

        self.motion.cleanup(frame_index)
        self.events.cleanup(frame_index)

        return {
            "observations": observations,
            "trajectories": trajectory_history,
            "pair_metrics": pair_metrics,
            "events": events
        }

    def finalize(self, fps: float, average_inference_fps: float):
        summary = self.trajectories.finalize(fps)
        alert_counts = self.events.alert_counts
        summary.update({
            "total_person_vehicle_proximity_events": (
                alert_counts["CAUTION"] + alert_counts["DANGER"]
            ),
            "caution_events": alert_counts["CAUTION"],
            "danger_events": alert_counts["DANGER"],
            "approaching_events": alert_counts["APPROACHING"],
            "high_risk_events": alert_counts["HIGH_RISK"],
            "average_inference_fps": average_inference_fps,
            "pixel_thresholds_are_experimental": True,
            "distance_units": "image pixels"
        })
        return summary

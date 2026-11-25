from collections import Counter, deque
from dataclasses import dataclass, field

import supervision as sv

from src.config import CLASS_MAPPING


@dataclass(frozen=True)
class TrackObservation:
    track_id: int
    object_class: str
    frame_number: int
    timestamp_seconds: float
    bbox: tuple[float, float, float, float]
    center_x: float
    center_y: float
    bottom_center_x: float
    bottom_center_y: float
    width: float
    height: float
    confidence: float

    @property
    def center(self):
        return self.center_x, self.center_y

    @property
    def bottom_center(self):
        return self.bottom_center_x, self.bottom_center_y


def _normalize_class_name(name):
    normalized = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"pallet_jack", "manual_pallet_jack"}:
        return "pallet_truck"
    return normalized


@dataclass
class _TrackState:
    history: deque
    class_votes: Counter = field(default_factory=Counter)
    last_frame: int = 0
    first_timestamp: float = 0.0
    last_timestamp: float = 0.0


class TrajectoryManager:
    def __init__(
        self,
        history_length: int = 30,
        cleanup_after_frames: int = 60
    ):
        if history_length < 1:
            raise ValueError("Trajectory history length must be at least one.")
        if cleanup_after_frames < 1:
            raise ValueError("Track cleanup period must be at least one frame.")

        self.history_length = history_length
        self.cleanup_after_frames = cleanup_after_frames
        self.active_tracks: dict[int, _TrackState] = {}
        self.unique_track_counts = Counter()
        self.duration_sum_seconds = 0.0
        self.duration_count = 0
        self.maximum_duration_seconds = 0.0
        self._finalized = False

    def update(self, detections: sv.Detections, frame_index: int, fps: float):
        if self._finalized:
            raise RuntimeError("Cannot update trajectories after finalization.")

        fps = fps if fps > 0 else 30.0
        timestamp = frame_index / fps
        observations = []
        track_ids = detections.tracker_id
        class_names = detections.data.get("class_name")

        if track_ids is not None:
            for index, raw_track_id in enumerate(track_ids):
                track_id = int(raw_track_id)
                if track_id < 0:
                    continue

                class_id = (
                    int(detections.class_id[index])
                    if detections.class_id is not None
                    else -1
                )
                object_class = (
                    _normalize_class_name(class_names[index])
                    if class_names is not None
                    else CLASS_MAPPING.get(class_id, f"class_{class_id}")
                )

                x1, y1, x2, y2 = (
                    float(value) for value in detections.xyxy[index]
                )
                observation = TrackObservation(
                    track_id=track_id,
                    object_class=object_class,
                    frame_number=frame_index + 1,
                    timestamp_seconds=timestamp,
                    bbox=(x1, y1, x2, y2),
                    center_x=(x1 + x2) / 2,
                    center_y=(y1 + y2) / 2,
                    bottom_center_x=(x1 + x2) / 2,
                    bottom_center_y=y2,
                    width=max(0.0, x2 - x1),
                    height=max(0.0, y2 - y1),
                    confidence=(
                        float(detections.confidence[index])
                        if detections.confidence is not None
                        else 0.0
                    )
                )

                state = self.active_tracks.get(track_id)
                if state is None:
                    state = _TrackState(
                        history=deque(maxlen=self.history_length),
                        first_timestamp=timestamp
                    )
                    self.active_tracks[track_id] = state

                state.history.append(observation)
                state.class_votes[object_class] += 1
                state.last_frame = frame_index
                state.last_timestamp = timestamp
                observations.append(observation)

        self._cleanup(frame_index, fps)
        trajectories = {
            track_id: tuple(state.history)
            for track_id, state in self.active_tracks.items()
        }
        return observations, trajectories

    def _cleanup(self, frame_index: int, fps: float):
        expired_ids = [
            track_id
            for track_id, state in self.active_tracks.items()
            if frame_index - state.last_frame > self.cleanup_after_frames
        ]
        for track_id in expired_ids:
            self._finalize_track(self.active_tracks.pop(track_id), fps)

    def _finalize_track(self, state: _TrackState, fps: float):
        if not state.class_votes:
            return

        object_class = state.class_votes.most_common(1)[0][0]
        self.unique_track_counts[object_class] += 1
        duration = max(
            0.0,
            state.last_timestamp - state.first_timestamp
        ) + 1.0 / fps
        self.duration_sum_seconds += duration
        self.duration_count += 1
        self.maximum_duration_seconds = max(
            self.maximum_duration_seconds,
            duration
        )

    def finalize(self, fps: float):
        fps = fps if fps > 0 else 30.0
        if not self._finalized:
            for state in self.active_tracks.values():
                self._finalize_track(state, fps)
            self.active_tracks.clear()
            self._finalized = True

        return {
            "unique_tracked_people": self.unique_track_counts["person"],
            "unique_tracked_forklifts": self.unique_track_counts["forklift"],
            "unique_tracked_pallet_trucks": self.unique_track_counts[
                "pallet_truck"
            ],
            "average_track_duration_seconds": (
                self.duration_sum_seconds / self.duration_count
                if self.duration_count
                else 0.0
            ),
            "maximum_track_duration_seconds": self.maximum_duration_seconds
        }

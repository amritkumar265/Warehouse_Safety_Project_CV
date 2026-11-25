from collections import deque
from dataclasses import dataclass, field
import math


def calculate_center_distance(center_a, center_b):
    return math.hypot(
        center_a[0] - center_b[0],
        center_a[1] - center_b[1]
    )


def calculate_bbox_distance(bbox_a, bbox_b):
    horizontal_gap = max(
        bbox_a[0] - bbox_b[2],
        bbox_b[0] - bbox_a[2],
        0.0
    )
    vertical_gap = max(
        bbox_a[1] - bbox_b[3],
        bbox_b[1] - bbox_a[3],
        0.0
    )
    return math.hypot(horizontal_gap, vertical_gap)


@dataclass
class _PairMotionState:
    distances: deque
    last_smoothed_distance: float | None = None
    last_timestamp: float | None = None
    last_frame: int | None = None
    candidate_state: str | None = None
    candidate_count: int = 0
    approach_state: str = "MAINTAINING"


class RelativeMotionAnalyzer:
    def __init__(
        self,
        smoothing_window: int = 5,
        state_change_observations: int = 3,
        delta_threshold_px: float = 1.0,
        cleanup_after_frames: int = 60
    ):
        if smoothing_window < 1:
            raise ValueError("Smoothing window must be at least one.")
        if state_change_observations < 1:
            raise ValueError("State transition observations must be at least one.")
        if delta_threshold_px < 0:
            raise ValueError("Approach delta threshold cannot be negative.")
        if cleanup_after_frames < 1:
            raise ValueError("Pair cleanup period must be at least one frame.")

        self.smoothing_window = smoothing_window
        self.state_change_observations = state_change_observations
        self.delta_threshold_px = delta_threshold_px
        self.cleanup_after_frames = cleanup_after_frames
        self.pair_states: dict[tuple[int, int], _PairMotionState] = {}

    def update(
        self,
        pair_key: tuple[int, int],
        bbox_distance_px: float,
        frame_index: int,
        timestamp_seconds: float
    ):
        state = self.pair_states.get(pair_key)
        if state is None:
            state = _PairMotionState(
                distances=deque(maxlen=self.smoothing_window)
            )
            self.pair_states[pair_key] = state

        if (
            state.last_frame is not None
            and frame_index != state.last_frame + 1
        ):
            state.candidate_state = None
            state.candidate_count = 0

        state.distances.append(float(bbox_distance_px))
        smoothed_distance = sum(state.distances) / len(state.distances)

        distance_change = None
        approach_speed = 0.0
        state_changed = False

        if state.last_smoothed_distance is not None:
            distance_change = smoothed_distance - state.last_smoothed_distance
            elapsed = timestamp_seconds - state.last_timestamp
            if elapsed > 0:
                approach_speed = -distance_change / elapsed

            if distance_change < -self.delta_threshold_px:
                candidate = "APPROACHING"
            elif distance_change > self.delta_threshold_px:
                candidate = "SEPARATING"
            else:
                candidate = "MAINTAINING"

            if state.candidate_state == candidate:
                state.candidate_count += 1
            else:
                state.candidate_state = candidate
                state.candidate_count = 1

            if (
                state.candidate_count >= self.state_change_observations
                and state.approach_state != candidate
            ):
                state.approach_state = candidate
                state_changed = True

        state.last_smoothed_distance = smoothed_distance
        state.last_timestamp = timestamp_seconds
        state.last_frame = frame_index

        return {
            "smoothed_bbox_distance_px": smoothed_distance,
            "distance_change_px": distance_change,
            "relative_approach_speed_px_s": approach_speed,
            "approach_state": state.approach_state,
            "approach_state_changed": state_changed
        }

    def cleanup(self, frame_index: int):
        expired_pairs = [
            pair_key
            for pair_key, state in self.pair_states.items()
            if (
                state.last_frame is not None
                and frame_index - state.last_frame > self.cleanup_after_frames
            )
        ]
        for pair_key in expired_pairs:
            del self.pair_states[pair_key]

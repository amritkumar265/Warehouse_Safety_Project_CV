from collections import Counter
from dataclasses import dataclass, field


def format_video_timestamp(timestamp_seconds: float) -> str:
    hours = int(timestamp_seconds // 3600)
    minutes = int((timestamp_seconds % 3600) // 60)
    seconds = timestamp_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


@dataclass
class _PairAlertState:
    proximity_level: str = "SAFE"
    approach_state: str = "MAINTAINING"
    high_risk: bool = False
    last_frame: int = 0
    last_alert_times: dict[str, float] = field(default_factory=dict)


class SafetyEventManager:
    def __init__(
        self,
        cooldown_seconds: float = 5.0,
        cleanup_after_frames: int = 60
    ):
        if cooldown_seconds < 0:
            raise ValueError("Alert cooldown cannot be negative.")
        if cleanup_after_frames < 1:
            raise ValueError("Event cleanup period must be at least one frame.")

        self.cooldown_seconds = cooldown_seconds
        self.cleanup_after_frames = cleanup_after_frames
        self.pair_states: dict[tuple[int, int], _PairAlertState] = {}
        self.alert_counts = Counter()

    def update(self, metrics, frame_index: int):
        pair_key = (
            int(metrics["person_track_id"]),
            int(metrics["vehicle_track_id"])
        )
        state = self.pair_states.setdefault(pair_key, _PairAlertState())
        level = metrics["proximity_level"]
        approach_state = metrics["approach_state"]
        high_risk = (
            approach_state == "APPROACHING"
            and level in {"CAUTION", "DANGER"}
        )

        alert_types = []
        if level == "CAUTION" and state.proximity_level == "SAFE":
            alert_types.append("CAUTION")
        elif level == "DANGER" and state.proximity_level != "DANGER":
            alert_types.append("DANGER")

        if (
            approach_state == "APPROACHING"
            and state.approach_state != "APPROACHING"
        ):
            alert_types.append("APPROACHING")

        if high_risk and not state.high_risk:
            alert_types.append("HIGH_RISK")

        state.proximity_level = level
        state.approach_state = approach_state
        state.high_risk = high_risk
        state.last_frame = frame_index

        events = []
        timestamp = float(metrics["timestamp_seconds"])
        for alert_type in alert_types:
            last_alert = state.last_alert_times.get(alert_type)
            if (
                last_alert is not None
                and timestamp - last_alert < self.cooldown_seconds
            ):
                continue

            state.last_alert_times[alert_type] = timestamp
            self.alert_counts[alert_type] += 1
            event = dict(metrics)
            event["timestamp"] = format_video_timestamp(timestamp)
            event["frame_number"] = frame_index + 1
            event["alert_type"] = alert_type
            events.append(event)

        return events

    def cleanup(self, frame_index: int):
        expired_pairs = [
            pair_key
            for pair_key, state in self.pair_states.items()
            if frame_index - state.last_frame > self.cleanup_after_frames
        ]
        for pair_key in expired_pairs:
            del self.pair_states[pair_key]

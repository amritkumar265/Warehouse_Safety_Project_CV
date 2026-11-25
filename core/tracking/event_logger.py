import csv
import json
from pathlib import Path


EVENT_FIELDS = (
    "timestamp",
    "frame_number",
    "person_track_id",
    "vehicle_track_id",
    "vehicle_class",
    "center_distance_px",
    "bbox_distance_px",
    "relative_approach_speed_px_s",
    "approach_state",
    "proximity_level",
    "alert_type"
)


class TrackingEventLogger:
    def __init__(self, csv_path):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.csv_path.open(
            "w",
            newline="",
            encoding="utf-8"
        )
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=EVENT_FIELDS,
            extrasaction="ignore"
        )
        self._writer.writeheader()
        self._closed = False

    def write_events(self, events):
        if self._closed:
            raise RuntimeError("Cannot write to a closed tracking event log.")
        for event in events:
            self._writer.writerow(event)
        if events:
            self._file.flush()

    @staticmethod
    def write_summary(json_path, summary):
        path = Path(json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as summary_file:
            json.dump(summary, summary_file, indent=2)

    def close(self):
        if not self._closed:
            self._file.close()
            self._closed = True

import json
import time
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import yaml

from core.tracking.byte_tracker import ByteTrackAdapter
from core.tracking.event_logger import TrackingEventLogger
from core.tracking.safety_monitor import SafetyMonitor
from src.config import (
    BYTE_TRACKER_CONFIG_PATH,
    CLASS_MAPPING,
    TRACKING_SAFETY_CONFIG_PATH
)


class TrackingPipeline:
    def __init__(
        self,
        video_path,
        detection_data_path,
        progress_callback=None,
        tracker_config_path=BYTE_TRACKER_CONFIG_PATH,
        tracking_safety_config_path=TRACKING_SAFETY_CONFIG_PATH,
        detection_inference_fps=0.0
    ):
        self.video_path = Path(video_path)
        self.detection_data_path = Path(detection_data_path)
        if not self.video_path.is_file():
            raise FileNotFoundError(f"Video not found: {self.video_path}")
        if not self.detection_data_path.is_file():
            raise FileNotFoundError(
                f"Detection data not found: {self.detection_data_path}"
            )

        self.tracker_config_path = Path(tracker_config_path)
        self.tracking_safety_config_path = Path(
            tracking_safety_config_path
        )
        self.progress_callback = progress_callback
        self.detection_inference_fps = float(detection_inference_fps)
        self.tracker = ByteTrackAdapter(self.tracker_config_path)
        self.tracking_config = self._load_tracking_config()
        self.safety_monitor = SafetyMonitor(self.tracking_config)
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(
            text_scale=0.5,
            text_thickness=1
        )

    def _load_tracking_config(self):
        if not self.tracking_safety_config_path.is_file():
            raise FileNotFoundError(
                "Tracking safety configuration not found: "
                f"{self.tracking_safety_config_path}"
            )
        with self.tracking_safety_config_path.open(
            "r",
            encoding="utf-8"
        ) as config_file:
            config = yaml.safe_load(config_file)
        if not isinstance(config, dict):
            raise ValueError(
                "Invalid tracking safety configuration: "
                f"{self.tracking_safety_config_path}"
            )
        return config

    @staticmethod
    def _detections_from_record(record, frame_index):
        if not isinstance(record, dict):
            raise ValueError(
                f"Invalid detection record for frame {frame_index}."
            )
        if record.get("frame_index") != frame_index:
            raise ValueError(
                "Detection data frame index mismatch: expected "
                f"{frame_index}, found {record.get('frame_index')}."
            )

        serialized = record.get("detections")
        if not isinstance(serialized, list):
            raise ValueError(
                f"Invalid detections entry for frame {frame_index}."
            )
        if not serialized:
            return sv.Detections.empty()

        return sv.Detections(
            xyxy=np.asarray(
                [item["bbox"] for item in serialized],
                dtype=np.float32
            ).reshape(-1, 4),
            confidence=np.asarray(
                [item.get("confidence", 0.0) for item in serialized],
                dtype=np.float32
            ),
            class_id=np.asarray(
                [
                    item["class_id"] if item.get("class_id") is not None else -1
                    for item in serialized
                ],
                dtype=np.int64
            ),
            data={
                "class_name": np.asarray(
                    [str(item["class_name"]) for item in serialized],
                    dtype=object
                )
            }
        )

    def _annotate_detections(self, frame, detections):
        labels = []
        if len(detections):
            class_names = detections.data.get("class_name")
            for index in range(len(detections)):
                class_id = (
                    int(detections.class_id[index])
                    if detections.class_id is not None
                    else -1
                )
                class_name = (
                    str(class_names[index])
                    if class_names is not None
                    else CLASS_MAPPING.get(class_id, f"class_{class_id}")
                )
                confidence = (
                    float(detections.confidence[index])
                    if detections.confidence is not None
                    else 0.0
                )
                track_id = (
                    int(detections.tracker_id[index])
                    if detections.tracker_id is not None
                    else -1
                )
                labels.append(
                    f"{class_name} {confidence:.2f} ID:{track_id}"
                )

        annotated = self.box_annotator.annotate(
            scene=frame.copy(),
            detections=detections
        )
        return self.label_annotator.annotate(
            scene=annotated,
            detections=detections,
            labels=labels
        )

    @staticmethod
    def _draw_tracking_visualization(frame, tracking_data):
        class_colors = {
            "person": (255, 180, 0),
            "forklift": (0, 180, 255),
            "pallet_truck": (180, 80, 255)
        }
        for history in tracking_data["trajectories"].values():
            points = np.asarray(
                [
                    (int(item.bottom_center_x), int(item.bottom_center_y))
                    for item in history
                ],
                dtype=np.int32
            )
            if len(points) > 1:
                cv2.polylines(
                    frame,
                    [points],
                    False,
                    class_colors.get(
                        history[-1].object_class,
                        (220, 220, 220)
                    ),
                    2,
                    cv2.LINE_AA
                )
            if len(points):
                color = class_colors.get(
                    history[-1].object_class,
                    (220, 220, 220)
                )
                cv2.circle(
                    frame,
                    tuple(points[-1]),
                    4,
                    color,
                    -1,
                    cv2.LINE_AA
                )

        status_colors = {
            "SAFE": (60, 190, 60),
            "CAUTION": (0, 200, 255),
            "DANGER": (0, 0, 255)
        }
        for metrics in tracking_data["pair_metrics"]:
            person_center = tuple(
                int(value) for value in metrics["person_center"]
            )
            vehicle_center = tuple(
                int(value) for value in metrics["vehicle_center"]
            )
            level = metrics["proximity_level"]
            color = status_colors[level]
            if metrics["high_risk"]:
                color = (255, 0, 255)

            cv2.line(
                frame,
                person_center,
                vehicle_center,
                color,
                2,
                cv2.LINE_AA
            )
            midpoint = (
                (person_center[0] + vehicle_center[0]) // 2,
                (person_center[1] + vehicle_center[1]) // 2
            )
            label = (
                f'{metrics["bbox_distance_px"]:.0f}px '
                f'{metrics["approach_state"]} {level}'
            )
            if metrics["high_risk"]:
                label += " HIGH RISK"
            cv2.putText(
                frame,
                label,
                (midpoint[0] + 4, midpoint[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                2,
                cv2.LINE_AA
            )

        cv2.putText(
            frame,
            "Experimental image-space thresholds (pixels)",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        return frame

    def process_video(self, output_path=None):
        if output_path is None:
            output_path = self.video_path.with_name(
                f"{self.video_path.stem}_tracked.mp4"
            )
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        events_path = output_path.with_name(
            f"{output_path.stem}_events.csv"
        )
        summary_path = output_path.with_name(
            f"{output_path.stem}_summary.json"
        )

        video_info = sv.VideoInfo.from_video_path(str(self.video_path))
        total_frames = video_info.total_frames
        frame_count = 0
        start_time = time.perf_counter()
        event_logger = TrackingEventLogger(events_path)
        self.tracker.reset()
        self.safety_monitor = SafetyMonitor(self.tracking_config)

        try:
            with self.detection_data_path.open(
                "r",
                encoding="utf-8"
            ) as detections_file:
                with sv.VideoSink(
                    target_path=str(output_path),
                    video_info=video_info,
                    codec="avc1"
                ) as sink:
                    for frame_index, frame in enumerate(
                        sv.get_video_frames_generator(
                            source_path=str(self.video_path)
                        )
                    ):
                        line = detections_file.readline()
                        if not line:
                            raise ValueError(
                                "Detection data ended before video frame "
                                f"{frame_index}."
                            )
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError as error:
                            raise ValueError(
                                "Invalid detection JSON on frame "
                                f"{frame_index}."
                            ) from error

                        detections = self._detections_from_record(
                            record,
                            frame_index
                        )
                        tracked = self.tracker.update(detections)
                        tracking_data = self.safety_monitor.update(
                            tracked,
                            frame_index=frame_index,
                            fps=video_info.fps
                        )
                        annotated = self._annotate_detections(
                            frame,
                            tracked
                        )
                        annotated = self._draw_tracking_visualization(
                            annotated,
                            tracking_data
                        )
                        sink.write_frame(annotated)
                        event_logger.write_events(tracking_data["events"])
                        frame_count += 1

                        if self.progress_callback:
                            self.progress_callback(frame_count, total_frames)

                if any(line.strip() for line in detections_file):
                    raise ValueError(
                        "Detection data contains more frames than the video."
                    )
        finally:
            event_logger.close()

        elapsed = time.perf_counter() - start_time
        summary = self.safety_monitor.finalize(
            fps=video_info.fps,
            average_inference_fps=self.detection_inference_fps
        )
        TrackingEventLogger.write_summary(summary_path, summary)
        return {
            "output_video": str(output_path),
            "events_csv": str(events_path),
            "summary_json": str(summary_path),
            "frames_processed": frame_count,
            "processing_time": round(elapsed, 2),
            "average_fps": round(
                frame_count / elapsed if elapsed > 0 else 0.0,
                2
            ),
            "tracking_summary": summary
        }

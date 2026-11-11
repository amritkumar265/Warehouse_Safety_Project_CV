import time
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import supervision as sv

from src.config import (
    MODELS_DIR,
    CLASS_MAPPING,
    DINO_PROMPT,
    YOLO_CONF_THRESH,
    DINO_BOX_THRESH,
    DINO_TEXT_THRESH,
    NMS_IOU_THRESH
)

from src.yolo_detector import YoloDetector
from src.dino_detector import DinoDetector
from src.fusion import BoxFusion

from core.zones.detection_filter import ZoneDetectionFilter

class DetectionPipeline:

    def __init__(self , camera_profile , yolo_model_path = None , progress_callback = None):

        self.camera_profile = camera_profile
        self.progress_callback = progress_callback

        if yolo_model_path is None:
            yolo_model_path = MODELS_DIR / "yolo11s_warehouse_best.pt"

        if not Path(yolo_model_path).exists():
            raise FileNotFoundError(
                f"YOLO model not found : {yolo_model_path}"
            )

        print("Loading YOLO.....")

        self.yolo = YoloDetector(
            str(yolo_model_path),
            conf = YOLO_CONF_THRESH
        )

        print("Loading Grounding DINO.....")

        self.dino = DinoDetector(
            box_thresh=DINO_BOX_THRESH,
            text_thresh=DINO_TEXT_THRESH
        )

        print("Initializing Box Fusion.....")

        self.fusion = BoxFusion(
            iou_threshold=NMS_IOU_THRESH
        )

        print("Initializing Zone Detection Filter.....")
        self.zone_filter = ZoneDetectionFilter(
            camera_profile=camera_profile
        )

        self.box_annotator = sv.BoxAnnotator(thickness=2)

        self.label_annotator = sv.LabelAnnotator(
            text_scale = 0.5,
            text_thickness=1
        )


    def _annotate_detections(self, frame, detections):
        labels = []

        if len(detections) > 0:
            for cid, conf in zip(
                detections.class_id,
                detections.confidence
            ):
                class_name = CLASS_MAPPING.get(
                    int(cid),
                    "unknown"
                )
                labels.append(f"{class_name} {float(conf):.2f}")

        annotated = self.box_annotator.annotate(
            scene=frame.copy(),
            detections=detections
        )

        annotated = self.label_annotator.annotate(
            scene=annotated,
            detections=detections,
            labels=labels
        )

        return annotated


    def process_frame(self, frame):
    # 1. Run detection
        yolo_det = self.yolo.detect(frame)

        dino_det = self.dino.detect(
            frame,
            prompt=DINO_PROMPT
        )

    # 2. Fuse detections
        fused_det = self.fusion.merge_and_suppress(
            yolo_det,
            dino_det
        )

    # 3. Baseline: annotate all fused detections
        baseline_annotated = self._annotate_detections(
            frame,
            fused_det
        )

    # 4. Apply zone filtering
        zone_results = self.zone_filter.process(
            fused_det,
            frame.shape
        )

        retained = zone_results["retained_detections"]

    # 5. Zone-filtered output
        filtered_annotated = self._annotate_detections(
            frame,
            retained
        )

        return {
            "baseline_annotated_frame": baseline_annotated,
            "annotated_frame": filtered_annotated,
            "raw_detections": fused_det,
            "retained_detections": retained,
            "suppressed_candidates": zone_results[
                "suppressed_candidates"
            ],
            "safety_flags": zone_results["safety_flags"],
            "audit_records": zone_results["audit_records"]
        }


    def process_video(self, video_path , output_path = None):
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(
                f"Video not found: {video_path}"
            )

        if output_path is None:
            output_dir = Path("outputs/processed")
            output_dir.mkdir(parents=True, exist_ok=True)

            output_path = (
                output_dir / f"{video_path.stem}_processed.mp4"
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)


        video_info = sv.VideoInfo.from_video_path(
            str(video_path)
        )

        total_frames = video_info.total_frames

        start_time = time.perf_counter()

        total_raw = 0
        total_retained = 0
        total_suppressed = 0
        total_flags = 0


        frame_count = 0


        with sv.VideoSink(
            target_path = str(output_path),
            video_info=video_info,
            codec = "mp4v"
        ) as sink:

            for frame_idx, frame in enumerate(
                sv.get_video_frames_generator(
                    source_path=str(video_path)
                )
            ):

                results = self.process_frame(frame)

                sink.write_frame(
                    results["annotated_frame"]
                )

                total_raw += len(results["raw_detections"])
                total_retained += len(
                    results["retained_detections"]
                )
                total_suppressed += len(
                    results["suppressed_candidates"]
                )
                total_flags += len(results["safety_flags"])

                frame_count += 1

                if self.progress_callback:
                    self.progress_callback(
                        frame_count,
                        total_frames
                    )

        elapsed = time.perf_counter() - start_time

        average_fps = (
            frame_count / elapsed if elapsed > 0 else 0
        )

        return {
            "output_video": str(output_path),
            "frames_processed": frame_count,
            "raw_detections": total_raw,
            "retained_detections": total_retained,
            "suppressed_candidates": total_suppressed,
            "safety_flags": total_flags,
            "processing_time": round(elapsed, 2),
            "average_fps": round(average_fps, 2)
        }

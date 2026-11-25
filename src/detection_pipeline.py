import time
from pathlib import Path
import sys

import json

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
    NMS_IOU_THRESH,
    FUSION_VEHICLE_MIN_CONFIDENCE,
    FUSION_VEHICLE_AGREEMENT_IOU
)

from src.yolo_detector import YoloDetector
from src.dino_detector import DinoDetector
from src.fusion import BoxFusion
from src.rfdetr_detector import RFDETRDetector

from core.zones.detection_filter import ZoneDetectionFilter

class DetectionPipeline:

    MODEL_CHOICES = {
        "YOLO",
        "Grounding DINO",
        "Fusion",
        "RF-DETR",
    }

    def __init__(
        self,
        camera_profile,
        yolo_model_path=None,
        progress_callback=None,
        model_choice="Fusion",
        rfdetr_checkpoint_path=None
    ):

        self.camera_profile = camera_profile
        self.progress_callback = progress_callback
        self.model_choice = model_choice

        if model_choice not in self.MODEL_CHOICES:
            raise ValueError(f"Unsupported detection model: {model_choice}")

        if model_choice in {"YOLO", "Fusion"}:
            if yolo_model_path is None:
                yolo_model_path = MODELS_DIR / "yolo11s_warehouse_best.pt"

            if not Path(yolo_model_path).exists():
                raise FileNotFoundError(
                    f"YOLO model not found: {yolo_model_path}"
                )

            print("Loading YOLO.....")
            self.yolo = YoloDetector(
                str(yolo_model_path),
                conf=YOLO_CONF_THRESH
            )

        if model_choice in {"Grounding DINO", "Fusion"}:
            print("Loading Grounding DINO.....")
            self.dino = DinoDetector(
                box_thresh=DINO_BOX_THRESH,
                text_thresh=DINO_TEXT_THRESH
            )

        if model_choice == "Fusion":
            print("Initializing Box Fusion.....")
            self.fusion = BoxFusion(
                iou_threshold=NMS_IOU_THRESH,
                vehicle_min_confidence=FUSION_VEHICLE_MIN_CONFIDENCE,
                vehicle_agreement_iou=FUSION_VEHICLE_AGREEMENT_IOU
            )

        if model_choice == "RF-DETR":
            print("Loading RF-DETR.....")
            self.rfdetr = RFDETRDetector(
                checkpoint_path=rfdetr_checkpoint_path,
                conf=YOLO_CONF_THRESH
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
            detected_names = detections.data.get("class_name")
            for index, (cid, conf) in enumerate(zip(
                detections.class_id,
                detections.confidence
            )):
                class_name = (
                    str(detected_names[index])
                    if detected_names is not None
                    else CLASS_MAPPING.get(int(cid), "unknown")
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
        inference_start = time.perf_counter()
        if self.model_choice == "YOLO":
            detections = self.yolo.detect(frame)
        elif self.model_choice == "Grounding DINO":
            detections = self.dino.detect(frame, prompt=DINO_PROMPT)
        elif self.model_choice == "RF-DETR":
            detections = self.rfdetr.detect(frame)
        else:
            yolo_detections = self.yolo.detect(frame)
            dino_detections = self.dino.detect(
                frame,
                prompt=DINO_PROMPT
            )
            detections = self.fusion.merge_and_suppress(
                yolo_detections,
                dino_detections
            )
        inference_seconds = time.perf_counter() - inference_start

        baseline_annotated = self._annotate_detections(
            frame,
            detections
        )

        zone_results = self.zone_filter.process(
            detections,
            frame.shape
        )

        retained = zone_results["retained_detections"]

        filtered_annotated = self._annotate_detections(
            frame,
            retained
        )

        return {
            "baseline_annotated_frame": baseline_annotated,
            "annotated_frame": filtered_annotated,
            "raw_detections": detections,
            "retained_detections": retained,
            "suppressed_candidates": zone_results[
                "suppressed_candidates"
            ],
            "safety_flags": zone_results["safety_flags"],
            "audit_records": zone_results["audit_records"],
            "inference_seconds": inference_seconds
        }


    @staticmethod
    def serialize_detections(detections):
        class_names = detections.data.get("class_name")
        serialized = []

        for index, box in enumerate(detections.xyxy):
            class_id = (
                int(detections.class_id[index])
                if detections.class_id is not None
                else None
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
            serialized.append({
                "bbox": [float(value) for value in box],
                "confidence": confidence,
                "class_id": class_id,
                "class_name": class_name
            })

        return serialized

    def process_video(
        self,
        video_path,
        output_path=None,
        detection_data_path=None
    ):
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
        if detection_data_path is None:
            detection_data_path = output_path.with_name(
                f"{output_path.stem}_detections.jsonl"
            )
        detection_data_path = Path(detection_data_path)
        detection_data_path.parent.mkdir(parents=True, exist_ok=True)


        video_info = sv.VideoInfo.from_video_path(str(video_path))
        total_frames = video_info.total_frames
        start_time = time.perf_counter()
        total_raw = 0
        total_retained = 0
        total_suppressed = 0
        total_flags = 0
        frame_count = 0
        total_inference_seconds = 0.0

        with detection_data_path.open("w", encoding="utf-8") as data_file:
            with sv.VideoSink(
                target_path=str(output_path),
                video_info=video_info,
                codec="avc1"
            ) as sink:
                for frame_idx, frame in enumerate(
                    sv.get_video_frames_generator(
                        source_path=str(video_path)
                    )
                ):
                    results = self.process_frame(
                        frame
                    )
                    sink.write_frame(results["annotated_frame"])
                    data_file.write(json.dumps({
                        "frame_index": frame_idx,
                        "detections": self.serialize_detections(
                            results["retained_detections"]
                        )
                    }) + "\n")

                    total_raw += len(results["raw_detections"])
                    total_retained += len(results["retained_detections"])
                    total_suppressed += len(
                        results["suppressed_candidates"]
                    )
                    total_flags += len(results["safety_flags"])
                    total_inference_seconds += results["inference_seconds"]
                    frame_count += 1

                    if self.progress_callback:
                        self.progress_callback(frame_count, total_frames)

        elapsed = time.perf_counter() - start_time
        average_fps = frame_count / elapsed if elapsed > 0 else 0

        return {
            "output_video": str(output_path),
            "detection_data_path": str(detection_data_path),
            "frames_processed": frame_count,
            "average_inference_fps": round(
                frame_count / total_inference_seconds
                if total_inference_seconds > 0
                else 0.0,
                2
            ),
            "raw_detections": total_raw,
            "retained_detections": total_retained,
            "suppressed_candidates": total_suppressed,
            "safety_flags": total_flags,
            "processing_time": round(elapsed, 2),
            "average_fps": round(average_fps, 2)
        }

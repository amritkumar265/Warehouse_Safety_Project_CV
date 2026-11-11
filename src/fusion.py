
import supervision as sv
import numpy as np


class BoxFusion:
    def __init__(self, iou_threshold: float = 0.50):
        self.iou_threshold = iou_threshold

    def _sanitize(self, det: sv.Detections) -> sv.Detections:
        if det is None or len(det) == 0:
            return sv.Detections.empty()

        xyxy = np.ascontiguousarray(det.xyxy, dtype=np.float32)
        conf = np.ascontiguousarray(det.confidence, dtype=np.float32)
        class_id = np.ascontiguousarray(det.class_id, dtype=np.int64)

        return sv.Detections(
            xyxy=xyxy,
            confidence=conf,
            class_id=class_id
        )

    def merge_and_suppress(
        self,
        det_a: sv.Detections,
        det_b: sv.Detections
    ) -> sv.Detections:

        clean_a = self._sanitize(det_a)
        clean_b = self._sanitize(det_b)

        has_a = len(clean_a) > 0
        has_b = len(clean_b) > 0

        if not has_a and not has_b:
            return sv.Detections.empty()

        if has_a and not has_b:
            return clean_a

        if not has_a and has_b:
            return clean_b

        combined = sv.Detections.merge([clean_a, clean_b])

        return combined.with_nms(
            threshold=self.iou_threshold,
            class_agnostic=False
        )

    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        yolo_detector,
        dino_cache_dict: dict
    ) -> sv.Detections:

        yolo_det = yolo_detector.detect(frame)
        dino_det = dino_cache_dict.get(
            frame_idx,
            sv.Detections.empty()
        )

        return self.merge_and_suppress(yolo_det, dino_det)

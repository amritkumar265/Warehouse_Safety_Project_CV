
import supervision as sv
import numpy as np


class BoxFusion:
    def __init__(
        self,
        iou_threshold: float = 0.50,
        vehicle_min_confidence: float | None = None,
        vehicle_agreement_iou: float = 0.50
    ):
        self.iou_threshold = iou_threshold
        self.vehicle_min_confidence = vehicle_min_confidence
        self.vehicle_agreement_iou = vehicle_agreement_iou

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

    def _filter_unconfirmed_vehicles(
        self,
        detections: sv.Detections,
        other_model_detections: sv.Detections
    ) -> sv.Detections:
        if self.vehicle_min_confidence is None or len(detections) == 0:
            return detections

        keep_indices = []

        for index, (class_id, confidence) in enumerate(zip(
            detections.class_id,
            detections.confidence
        )):
            if class_id not in (1, 2) or confidence >= self.vehicle_min_confidence:
                keep_indices.append(index)
                continue

            other_indices = np.flatnonzero(
                other_model_detections.class_id == class_id
            )
            if len(other_indices) == 0:
                continue

            box = detections.xyxy[index]
            other_boxes = other_model_detections.xyxy[other_indices]
            top_left = np.maximum(box[:2], other_boxes[:, :2])
            bottom_right = np.minimum(box[2:], other_boxes[:, 2:])
            intersection_size = np.maximum(bottom_right - top_left, 0)
            intersection_area = (
                intersection_size[:, 0] * intersection_size[:, 1]
            )
            box_area = (
                max(box[2] - box[0], 0) * max(box[3] - box[1], 0)
            )
            other_areas = (
                np.maximum(other_boxes[:, 2] - other_boxes[:, 0], 0)
                * np.maximum(other_boxes[:, 3] - other_boxes[:, 1], 0)
            )
            union = box_area + other_areas - intersection_area
            iou = np.divide(
                intersection_area,
                union,
                out=np.zeros_like(intersection_area, dtype=float),
                where=union > 0
            )

            if np.any(iou >= self.vehicle_agreement_iou):
                keep_indices.append(index)

        if not keep_indices:
            return sv.Detections.empty()

        return detections[np.asarray(keep_indices, dtype=int)]

    def merge_and_suppress(
        self,
        det_a: sv.Detections,
        det_b: sv.Detections
    ) -> sv.Detections:

        clean_a = self._sanitize(det_a)
        clean_b = self._sanitize(det_b)
        clean_a = self._filter_unconfirmed_vehicles(clean_a, clean_b)
        clean_b = self._filter_unconfirmed_vehicles(clean_b, clean_a)

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

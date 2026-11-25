import cv2
import numpy as np
import supervision as sv


class RFDETRDetector:
    COMMON_CLASS_IDS = {
        "person": 0,
        "forklift": 1,
        "pallet truck": 2,
        "pallet jack": 2,
        "manual pallet jack": 2,
    }

    def __init__(self, checkpoint_path=None, conf=0.25):
        try:
            from rfdetr import RFDETR, RFDETRSmall
        except ImportError as error:
            raise ImportError(
                "RF-DETR requires the rfdetr package. Install project requirements."
            ) from error

        if checkpoint_path:
            model = RFDETR.from_checkpoint(str(checkpoint_path))
        else:
            model = RFDETRSmall()

        self.predictor = model.inference(compile=False)
        self.conf = conf
        self.class_names = self._normalize_class_names(
            self.predictor.class_names
        )
        self.class_ids = self._build_class_ids()

    @staticmethod
    def _normalize_class_names(class_names):
        if isinstance(class_names, dict):
            return {
                int(class_id): str(class_name)
                for class_id, class_name in class_names.items()
            }

        if isinstance(class_names, (list, tuple)):
            return {
                class_id: str(class_name)
                for class_id, class_name in enumerate(class_names)
            }

        raise TypeError("RF-DETR did not provide a supported class-name mapping.")

    @staticmethod
    def _normalize_label(label):
        return " ".join(
            str(label).strip().lower().replace("_", " ").replace("-", " ").split()
        )

    def _build_class_ids(self):
        class_ids = {}
        next_class_id = 3

        for source_id, label in self.class_names.items():
            normalized_label = self._normalize_label(label)

            if normalized_label in self.COMMON_CLASS_IDS:
                class_ids[source_id] = self.COMMON_CLASS_IDS[normalized_label]
            elif "forklift" in normalized_label:
                class_ids[source_id] = self.COMMON_CLASS_IDS["forklift"]
            elif "pallet truck" in normalized_label or "pallet jack" in normalized_label:
                class_ids[source_id] = self.COMMON_CLASS_IDS["pallet truck"]
            else:
                class_ids[source_id] = next_class_id
                next_class_id += 1

        return class_ids

    def detect(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = self.predictor.predict(
            rgb_frame,
            threshold=self.conf
        )

        if len(detections) == 0:
            return sv.Detections.empty()

        source_class_ids = detections.class_id
        if source_class_ids is None:
            return sv.Detections.empty()

        class_names = [
            self.class_names.get(int(class_id), f"class_{int(class_id)}")
            for class_id in source_class_ids
        ]
        unified_class_ids = [
            self.class_ids.get(int(class_id), 3)
            for class_id in source_class_ids
        ]

        return sv.Detections(
            xyxy=np.ascontiguousarray(detections.xyxy, dtype=np.float32),
            confidence=np.ascontiguousarray(
                detections.confidence,
                dtype=np.float32
            ),
            class_id=np.ascontiguousarray(
                unified_class_ids,
                dtype=np.int64
            ),
            data={"class_name": np.asarray(class_names, dtype=object)}
        )

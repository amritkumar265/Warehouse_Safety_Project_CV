import sys
from types import ModuleType

import numpy as np
import supervision as sv

from src.rfdetr_detector import RFDETRDetector


def test_rfdetr_predictions_are_normalized_and_keep_all_classes(monkeypatch):
    class FakePredictor:
        class_names = {
            0: "person",
            1: "forklift",
            2: "pallet truck",
            3: "chair"
        }

        def predict(self, image, threshold):
            assert image[0, 0].tolist() == [3, 2, 1]
            assert threshold == 0.4
            return sv.Detections(
                xyxy=np.asarray([[0, 0, 1, 1]] * 4, dtype=np.float32),
                confidence=np.asarray([0.9] * 4, dtype=np.float32),
                class_id=np.asarray([0, 1, 2, 3], dtype=np.int64)
            )

    class FakeModel:
        def inference(self, compile):
            assert compile is False
            return FakePredictor()

    fake_rfdetr = ModuleType("rfdetr")
    fake_rfdetr.RFDETR = FakeModel
    fake_rfdetr.RFDETRSmall = FakeModel
    monkeypatch.setitem(sys.modules, "rfdetr", fake_rfdetr)

    detector = RFDETRDetector(conf=0.4)
    detections = detector.detect(
        np.asarray([[[1, 2, 3]]], dtype=np.uint8)
    )

    assert detections.class_id.tolist() == [0, 1, 2, 3]
    assert detections.data["class_name"].tolist() == [
        "person",
        "forklift",
        "pallet truck",
        "chair"
    ]

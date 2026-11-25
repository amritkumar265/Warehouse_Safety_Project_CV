import numpy as np
import pytest
import supervision as sv

from src.detection_pipeline import DetectionPipeline


def test_detection_export_preserves_generic_detector_labels():
    detections = sv.Detections(
        xyxy=np.asarray([[1, 2, 11, 22]], dtype=np.float32),
        confidence=np.asarray([0.85], dtype=np.float32),
        class_id=np.asarray([17], dtype=np.int64),
        data={"class_name": np.asarray(["safety_cone"], dtype=object)}
    )

    exported = DetectionPipeline.serialize_detections(detections)

    assert exported[0]["bbox"] == [1.0, 2.0, 11.0, 22.0]
    assert exported[0]["confidence"] == pytest.approx(0.85)
    assert exported[0]["class_id"] == 17
    assert exported[0]["class_name"] == "safety_cone"

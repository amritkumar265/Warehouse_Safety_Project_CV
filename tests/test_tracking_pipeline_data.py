import pytest

from src.tracking_pipeline import TrackingPipeline


def test_tracking_pipeline_loads_boxes_and_labels_from_any_detector():
    detections = TrackingPipeline._detections_from_record(
        {
            "frame_index": 3,
            "detections": [{
                "bbox": [1.0, 2.0, 11.0, 22.0],
                "confidence": 0.85,
                "class_id": 17,
                "class_name": "safety_cone"
            }]
        },
        frame_index=3
    )

    assert detections.xyxy.tolist() == [[1.0, 2.0, 11.0, 22.0]]
    assert detections.data["class_name"].tolist() == ["safety_cone"]
    assert detections.class_id.tolist() == [17]


def test_tracking_pipeline_rejects_misaligned_frame_data():
    with pytest.raises(ValueError, match="frame index mismatch"):
        TrackingPipeline._detections_from_record(
            {"frame_index": 2, "detections": []},
            frame_index=3
        )

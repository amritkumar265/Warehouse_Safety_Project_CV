from unittest.mock import Mock

import pytest

import src.detection_pipeline as detection_pipeline


@pytest.mark.parametrize(
    ("model_choice", "expected_models"),
    [
        ("YOLO", {"yolo"}),
        ("Grounding DINO", {"dino"}),
        ("Fusion", {"yolo", "dino", "fusion"}),
        ("RF-DETR", {"rfdetr"})
    ],
)
def test_pipeline_loads_only_selected_model(
    monkeypatch,
    tmp_path,
    model_choice,
    expected_models
):
    yolo = Mock()
    dino = Mock()
    rfdetr = Mock()
    fusion = Mock()
    monkeypatch.setattr(detection_pipeline, "YoloDetector", yolo)
    monkeypatch.setattr(detection_pipeline, "DinoDetector", dino)
    monkeypatch.setattr(detection_pipeline, "RFDETRDetector", rfdetr)
    monkeypatch.setattr(detection_pipeline, "BoxFusion", fusion)

    yolo_checkpoint = tmp_path / "yolo.pt"
    yolo_checkpoint.touch()

    pipeline = detection_pipeline.DetectionPipeline(
        camera_profile={"camera_id": "test", "zones": {}},
        yolo_model_path=yolo_checkpoint,
        model_choice=model_choice
    )

    assert hasattr(pipeline, "yolo") == ("yolo" in expected_models)
    assert hasattr(pipeline, "dino") == ("dino" in expected_models)
    assert hasattr(pipeline, "fusion") == ("fusion" in expected_models)
    assert hasattr(pipeline, "rfdetr") == ("rfdetr" in expected_models)


def test_detection_pipeline_does_not_combine_tracking_with_detection():
    assert "YOLO + ByteTrack Safety" not in (
        detection_pipeline.DetectionPipeline.MODEL_CHOICES
    )

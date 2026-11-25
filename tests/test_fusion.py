import numpy as np
import supervision as sv

from src.fusion import BoxFusion


def detections(boxes, confidence, class_ids):
    return sv.Detections(
        xyxy=np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
        confidence=np.asarray(confidence, dtype=np.float32),
        class_id=np.asarray(class_ids, dtype=np.int64)
    )


def test_low_confidence_vehicle_requires_cross_model_agreement():
    fusion = BoxFusion(
        vehicle_min_confidence=0.50,
        vehicle_agreement_iou=0.50
    )
    one_model = detections([[10, 10, 110, 110]], [0.42], [1])

    result = fusion.merge_and_suppress(one_model, sv.Detections.empty())

    assert len(result) == 0


def test_low_confidence_vehicle_is_kept_when_models_agree():
    fusion = BoxFusion(
        vehicle_min_confidence=0.50,
        vehicle_agreement_iou=0.50
    )
    yolo = detections([[10, 10, 110, 110]], [0.42], [1])
    dino = detections([[15, 15, 115, 115]], [0.44], [1])

    result = fusion.merge_and_suppress(yolo, dino)

    assert len(result) == 1
    assert int(result.class_id[0]) == 1


def test_high_confidence_vehicle_and_person_are_not_gated():
    fusion = BoxFusion(
        vehicle_min_confidence=0.50,
        vehicle_agreement_iou=0.50
    )
    model_detections = detections(
        [[10, 10, 110, 110], [200, 200, 250, 300]],
        [0.75, 0.42],
        [1, 0]
    )

    result = fusion.merge_and_suppress(
        model_detections,
        sv.Detections.empty()
    )

    assert result.class_id.tolist() == [1, 0]


def test_default_box_fusion_keeps_existing_behavior():
    fusion = BoxFusion()
    one_model = detections([[10, 10, 110, 110]], [0.42], [1])

    result = fusion.merge_and_suppress(one_model, sv.Detections.empty())

    assert len(result) == 1

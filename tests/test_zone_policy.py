import pytest

from core.zones.zone_policy import ZonePolicyEngine
from core.zones.zone_engine import ZoneEngine


@pytest.mark.parametrize(
    ("class_name", "coverage", "expected"),
    [
        ("person", 0.50, "retain"),
        ("person", 0.5001, "suppress_candidate"),
        ("forklift", 0.70, "retain"),
        ("forklift", 0.7001, "suppress_candidate"),
        ("pallet_truck", 0.7001, "suppress_candidate"),
    ],
)
def test_machinery_suppression_uses_class_specific_overlap(
    class_name,
    coverage,
    expected
):
    policy = ZonePolicyEngine()
    result = policy.evaluate({
        "class_name": class_name,
        "zone_results": {
            "machinery": {
                "contains_bottom_center": True,
                "bbox_coverage": coverage
            }
        }
    })

    assert result["decision"] == expected


@pytest.mark.parametrize("class_name", ["person", "forklift", "chair"])
def test_miscellaneous_suppresses_any_positive_overlap(class_name):
    policy = ZonePolicyEngine()
    result = policy.evaluate({
        "class_name": class_name,
        "zone_results": {
            "miscellaneous": {
                "contains_bottom_center": False,
                "bbox_coverage": 0.0001
            }
        }
    })

    assert result["decision"] == "suppress_candidate"


def test_miscellaneous_does_not_suppress_without_overlap():
    policy = ZonePolicyEngine()
    result = policy.evaluate({
        "class_name": "person",
        "zone_results": {
            "miscellaneous": {
                "contains_bottom_center": True,
                "bbox_coverage": 0
            }
        }
    })

    assert result["decision"] == "retain"


def test_tiny_miscellaneous_overlap_is_not_rounded_away():
    profile = {
        "camera_id": "test",
        "zones": {
            "miscellaneous": {
                "polygons": [[
                    [0.100, 0.100],
                    [0.101, 0.100],
                    [0.101, 0.101],
                    [0.100, 0.101]
                ]]
            }
        }
    }
    association = ZoneEngine(profile).associate_detection(
        bbox=[0, 0, 1000, 1000],
        class_name="person",
        frame_shape=(1000, 1000, 3)
    )

    assert association["zone_results"]["miscellaneous"]["bbox_coverage"] > 0
    assert ZonePolicyEngine().evaluate(association)["decision"] == "suppress_candidate"


@pytest.mark.parametrize("zone", ["shared_movement", "pedestrian_walkway"])
def test_non_suppression_zones_keep_detections(zone):
    policy = ZonePolicyEngine()
    result = policy.evaluate({
        "class_name": "forklift",
        "zone_results": {
            zone: {
                "contains_bottom_center": True,
                "bbox_coverage": 0.9
            }
        }
    })

    assert result["decision"] != "suppress_candidate"

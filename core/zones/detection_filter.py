
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from zones.zone_engine import ZoneEngine
from zones.zone_policy import ZonePolicyEngine


CLASS_MAPPING = {
    0: "person",
    1: "forklift",
    2: "pallet_truck"
}


class ZoneDetectionFilter:

    def __init__(self, camera_profile, suppression_threshold=0.70):

        self.zone_engine = ZoneEngine(camera_profile)

        self.policy_engine = ZonePolicyEngine(
            suppression_threshold=suppression_threshold
        )

    def process(self, detections, frame_shape):

        retained_indices = []
        suppressed_indices = []

        audit_records = []
        safety_flags = []

        # Handle empty detections consistently
        if detections is None or len(detections) == 0:

            return {
                "raw_detections": detections,
                "retained_detections": detections,
                "suppressed_candidates": detections,
                "safety_flags": safety_flags,
                "audit_records": audit_records
            }

        for index in range(len(detections)):

            bbox = detections.xyxy[index].tolist()

            class_id = (
                int(detections.class_id[index])
                if detections.class_id is not None
                else None
            )

            class_name = CLASS_MAPPING.get(
                class_id,
                "unknown"
            )

            confidence = (
                float(detections.confidence[index])
                if detections.confidence is not None
                else None
            )

            # Unknown classes are retained by default
            if class_name == "unknown":

                retained_indices.append(index)

                audit_records.append({
                    "detection_index": index,
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "decision": "retain",
                    "reason": "unknown_class_default_retain",
                    "zone": None
                })

                continue

            # Zone association
            association = self.zone_engine.associate_detection(
                bbox=bbox,
                class_name=class_name,
                frame_shape=frame_shape
            )

            # Evaluate zone policy
            decision = self.policy_engine.evaluate(association)

            action = decision["decision"]

            record = {
                "detection_index": index,
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "bbox": bbox,
                "zone": decision["zone"],
                "containing_zones": association["containing_zones"],
                "zone_results": association["zone_results"],
                "decision": action,
                "reason": decision["reason"]
            }

            audit_records.append(record)

            # Apply decision
            if action == "suppress_candidate":

                suppressed_indices.append(index)

            else:

                # Both retain and flag detections remain available
                # to downstream tracking and safety analysis.
                retained_indices.append(index)

            if action == "flag":
                safety_flags.append(record)

        # Convert selected indices into Supervision detections
        retained_detections = detections[
            np.asarray(retained_indices, dtype=int)
        ]

        suppressed_candidates = detections[
            np.asarray(suppressed_indices, dtype=int)
        ]

        return {
            "raw_detections": detections,
            "retained_detections": retained_detections,
            "suppressed_candidates": suppressed_candidates,
            "safety_flags": safety_flags,
            "audit_records": audit_records
        }

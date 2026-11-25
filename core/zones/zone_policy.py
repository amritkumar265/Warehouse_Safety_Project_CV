
class ZonePolicyEngine:

    def __init__(self, suppression_threshold=0.70):
        self.suppression_threshold = suppression_threshold

        self.policies = {
            "machinery": {
                "person": "suppress_candidate",
                "forklift": "suppress_candidate",
                "pallet_truck": "suppress_candidate"
            },

            "shared_movement": {
                "person": "retain",
                "forklift": "retain",
                "pallet_truck": "retain"
            },

            "pedestrian_walkway": {
                "person": "retain",
                "forklift": "flag",
                "pallet_truck": "flag"
            },

            "miscellaneous": {
                "person": "suppress_candidate",
                "forklift": "suppress_candidate",
                "pallet_truck": "suppress_candidate"
            }
        }

        self.suppression_thresholds = {
            "machinery": {
                "person": 0.50,
                "forklift": 0.70,
                "pallet_truck": 0.70
            }
        }

    def evaluate(self, association_result):

        class_name = association_result["class_name"].lower()
        zone_results = association_result["zone_results"]

        decision = "retain"
        reason = "outside_configured_zone"
        selected_zone = None

        candidate_decisions = []

        for zone_key, zone_data in zone_results.items():
            zone_policy = self.policies.get(zone_key, {})
            if zone_key == "miscellaneous":
                zone_decision = "suppress_candidate"
            else:
                zone_decision = zone_policy.get(class_name, "retain")

            contains_bottom_center = zone_data["contains_bottom_center"]
            coverage = zone_data["bbox_coverage"]
            suppression_threshold = self.suppression_threshold

            if zone_key == "miscellaneous":
                is_associated = coverage > 0
                suppression_threshold = 0
            elif zone_decision == "suppress_candidate":
                suppression_threshold = self.suppression_thresholds.get(
                    zone_key, {}
                ).get(class_name, self.suppression_threshold)
                is_associated = coverage > suppression_threshold
            else:
                is_associated = (
                    contains_bottom_center
                    or coverage >= self.suppression_threshold
                )

            if not is_associated:
                continue

            priority = {
                "suppress_candidate": 3,
                "flag": 2,
                "retain": 1
            }[zone_decision]
            candidate_decisions.append(
                (
                    priority,
                    zone_decision,
                    zone_key,
                    coverage,
                    suppression_threshold
                )
            )

        if candidate_decisions:

            (
                _,
                proposed_decision,
                selected_zone,
                coverage,
                suppression_threshold
            ) = max(
                candidate_decisions,
                key=lambda item: item[0]
            )

            if proposed_decision == "suppress_candidate":
                decision = "suppress_candidate"
                if selected_zone == "miscellaneous":
                    reason = "miscellaneous_zone_overlap"
                else:
                    reason = (
                        f"{selected_zone}_overlap_above_"
                        f"{suppression_threshold:.0%}"
                    )

            elif proposed_decision == "flag":
                decision = "flag"
                reason = "vehicle_in_pedestrian_walkway"

            else:
                decision = "retain"
                reason = "class_allowed_in_zone"

        return {
            "class_name": class_name,
            "zone": selected_zone,
            "decision": decision,
            "reason": reason
        }


class ZonePolicyEngine:

    def __init__(self, suppression_threshold=0.70):
        self.suppression_threshold = suppression_threshold

        self.policies = {
            "machinery": {
                "person": "retain",
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

    def evaluate(self, association_result):

        class_name = association_result["class_name"].lower()
        zone_results = association_result["zone_results"]

        decision = "retain"
        reason = "outside_configured_zone"
        selected_zone = None

        candidate_decisions = []

        # Evaluate every configured zone.
        for zone_key, zone_data in zone_results.items():

            zone_policy = self.policies.get(zone_key, {})
            if zone_key == "miscellaneous":
                zone_decision = "suppress_candidate"

            else:
                zone_decision = zone_policy.get(class_name , "retain")

            contains_bottom_center = zone_data["contains_bottom_center"]
            coverage = zone_data["bbox_coverage"]

            # Zone association:
            # Bottom-centre inside OR sufficient bbox coverage.
            if zone_key == "miscellaneous":
                is_associated = (
                    contains_bottom_center or coverage >= 0.50
                )

            else:
                is_associated = (
                    contains_bottom_center or coverage >= self.suppression_threshold
                )
                
            if not is_associated:
                continue

            if zone_decision == "flag":
                candidate_decisions.append(
                    (3, "flag", zone_key, coverage)
                )

            elif zone_decision == "suppress_candidate":
                candidate_decisions.append(
                    (2, "suppress_candidate", zone_key, coverage)
                )

            else:
                candidate_decisions.append(
                    (1, "retain", zone_key, coverage)
                )

        # Resolve conflicting zone policies using priority.
        if candidate_decisions:

            priority, proposed_decision, selected_zone, coverage = max(
                candidate_decisions,
                key=lambda item: item[0]
            )

            if proposed_decision == "suppress_candidate":

                if coverage >= self.suppression_threshold or selected_zone == "miscellaneous":
                    decision = "suppress_candidate"
                    reason = f"{selected_zone}_high_coverage"

                else:
                    decision = "retain"
                    reason = "insufficient_zone_coverage"

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

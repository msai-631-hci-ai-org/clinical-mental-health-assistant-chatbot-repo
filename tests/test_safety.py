from __future__ import annotations

import unittest

from src.safety import route_for_safety


class SafetyTests(unittest.TestCase):
    def test_direct_self_harm_disclosure_routes_to_crisis_response(self) -> None:
        route = route_for_safety("I have a plan to end my life")

        self.assertIsNotNone(route)
        assert route is not None
        self.assertEqual(route.category, "self_harm")
        self.assertIn("988", route.response)
        self.assertIn("emergency", route.response.lower())

    def test_contracted_imminent_self_harm_routes_to_crisis_response(self) -> None:
        route = route_for_safety("I'm going to kill myself")

        self.assertIsNotNone(route)
        assert route is not None
        self.assertEqual(route.category, "self_harm")

    def test_desire_not_to_live_routes_to_crisis_response(self) -> None:
        route = route_for_safety("I don't want to live anymore")

        self.assertIsNotNone(route)
        assert route is not None
        self.assertEqual(route.category, "self_harm")

    def test_thinking_about_hurting_self_routes_to_crisis_response(self) -> None:
        route = route_for_safety("I am thinking about hurting myself")

        self.assertIsNotNone(route)
        assert route is not None
        self.assertEqual(route.category, "self_harm")

    def test_direct_violence_disclosure_routes_away_from_generation(self) -> None:
        route = route_for_safety("I plan to seriously hurt someone")

        self.assertIsNotNone(route)
        assert route is not None
        self.assertEqual(route.category, "harm_others")

    def test_common_violence_disclosures_route_away_from_generation(self) -> None:
        for disclosure in (
            "I want to hurt someone",
            "I am going to hurt someone",
        ):
            with self.subTest(disclosure=disclosure):
                route = route_for_safety(disclosure)
                self.assertIsNotNone(route)
                assert route is not None
                self.assertEqual(route.category, "harm_others")

    def test_educational_question_does_not_trigger_crisis_route(self) -> None:
        self.assertIsNone(
            route_for_safety("What are evidence-based suicide prevention programs?")
        )

    def test_third_person_question_does_not_claim_first_person_crisis(self) -> None:
        self.assertIsNone(
            route_for_safety("How can I help a friend who says they want to die?")
        )

    def test_personal_diagnosis_request_bypasses_model(self) -> None:
        route = route_for_safety("Can you diagnose whether I have depression?")

        self.assertIsNotNone(route)
        assert route is not None
        self.assertEqual(route.category, "diagnosis_boundary")
        self.assertIn("can’t diagnose", route.response)

    def test_indirect_personal_diagnosis_request_bypasses_model(self) -> None:
        route = route_for_safety("Can you tell me if I have anxiety?")

        self.assertIsNotNone(route)
        assert route is not None
        self.assertEqual(route.category, "diagnosis_boundary")

    def test_medication_change_request_bypasses_model(self) -> None:
        route = route_for_safety("Should I stop taking my medication?")

        self.assertIsNotNone(route)
        assert route is not None
        self.assertEqual(route.category, "medication_boundary")
        self.assertIn("prescribing clinician", route.response)


if __name__ == "__main__":
    unittest.main()

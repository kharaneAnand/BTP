"""Regression tests for sequential controller metric alignment."""

from __future__ import annotations

import unittest

import pandas as pd

from evaluations.supervisor.evaluate_reliability_controller import (
    select_current_frame_results,
)


class ControllerEvaluationAlignmentTests(unittest.TestCase):
    def test_current_frame_is_scored_with_receiver_used_before_decision(self) -> None:
        decisions = pd.DataFrame(
            [
                {
                    "sample_index": 0,
                    "active_model_before": "oamp_dl",
                    "active_model_after": "original_gnn",
                    "receiver_used_for_frame": "oamp_dl",
                }
            ]
        )
        labels = pd.DataFrame(
            [
                {"sample_index": 0, "receiver": "oamp_dl", "ber": 0.01},
                {"sample_index": 0, "receiver": "original_gnn", "ber": 0.20},
            ]
        )
        best = pd.DataFrame(
            [{"sample_index": 0, "best_ber_receiver": "oamp_dl", "best_ber": 0.01}]
        )

        selected = select_current_frame_results(decisions, labels, best)

        self.assertEqual(selected.loc[0, "receiver"], "oamp_dl")
        self.assertAlmostEqual(selected.loc[0, "ber"], 0.01)
        self.assertTrue(bool(selected.loc[0, "selected_is_best_ber"]))

    def test_legacy_decision_frame_without_explicit_used_column_is_supported(self) -> None:
        decisions = pd.DataFrame(
            [
                {
                    "sample_index": 1,
                    "active_model_before": "mmse",
                    "active_model_after": "oamp_dl",
                }
            ]
        )
        labels = pd.DataFrame(
            [
                {"sample_index": 1, "receiver": "mmse", "ber": 0.05},
                {"sample_index": 1, "receiver": "oamp_dl", "ber": 0.01},
            ]
        )
        best = pd.DataFrame(
            [{"sample_index": 1, "best_ber_receiver": "oamp_dl", "best_ber": 0.01}]
        )

        selected = select_current_frame_results(decisions, labels, best)

        self.assertEqual(selected.loc[0, "receiver"], "mmse")
        self.assertAlmostEqual(selected.loc[0, "ber"], 0.05)
        self.assertFalse(bool(selected.loc[0, "selected_is_best_ber"]))


if __name__ == "__main__":
    unittest.main()

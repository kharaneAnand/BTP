"""Tests for paired, condition-stratified pipeline uncertainty estimates."""

from __future__ import annotations

import unittest

import numpy as np

from run_pipeline import stratified_bootstrap_ci


class StratifiedBootstrapTests(unittest.TestCase):
    def test_constant_paired_differences_have_degenerate_interval(self) -> None:
        values = np.full(18, -0.01)
        conditions = [(10, 30)] * 6 + [(15, 120)] * 6 + [(20, 500)] * 6

        low, high = stratified_bootstrap_ci(values, conditions, replicates=1000)

        self.assertAlmostEqual(low, -0.01)
        self.assertAlmostEqual(high, -0.01)

    def test_bootstrap_is_deterministic_and_preserves_cell_balance(self) -> None:
        values = np.asarray(
            [0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3],
            dtype=float,
        )
        conditions = [(10, 30)] * 4 + [(20, 500)] * 4

        first = stratified_bootstrap_ci(values, conditions, replicates=1000, seed=7)
        second = stratified_bootstrap_ci(values, conditions, replicates=1000, seed=7)

        self.assertEqual(first, second)
        self.assertLess(first[0], float(values.mean()))
        self.assertGreater(first[1], float(values.mean()))

    def test_rejects_unbalanced_singleton_condition(self) -> None:
        with self.assertRaises(ValueError):
            stratified_bootstrap_ci(
                np.asarray([0.1, 0.2, 0.3]),
                [(10, 30), (10, 30), (20, 500)],
                replicates=1000,
            )


if __name__ == "__main__":
    unittest.main()

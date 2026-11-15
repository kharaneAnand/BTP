"""Tests for the target-free pilot-guard noise estimator."""
from __future__ import annotations

import types
import unittest

import numpy as np

from models.supervisor.pilot_noise_estimator import (
    estimate_noise_power_from_rx_dd,
    pilot_guard_noise_samples,
)


def config():
    return types.SimpleNamespace(
        otfs=types.SimpleNamespace(M=32, N=16),
        pilot=types.SimpleNamespace(pilot_delay_length=1, pilot_doppler_length=1),
        channel=types.SimpleNamespace(max_delay=4),
    )


class PilotGuardNoiseEstimatorTests(unittest.TestCase):
    def test_reads_only_central_pilot_delay_column(self):
        rng = np.random.default_rng(2026)
        noise_power = 0.04
        rx = np.full((16, 32), 1e4 + 1e4j, dtype=np.complex128)
        noise = np.sqrt(noise_power / 2) * (
            rng.standard_normal(16) + 1j * rng.standard_normal(16)
        )
        rx[:, 15] = noise
        np.testing.assert_array_equal(pilot_guard_noise_samples(rx, config()), noise)
        estimate = estimate_noise_power_from_rx_dd(rx, config())
        self.assertAlmostEqual(estimate, float(np.mean(np.abs(noise) ** 2)))

    def test_estimate_is_unbiased_in_expectation_over_many_frames(self):
        rng = np.random.default_rng(53)
        noise_power = 0.07
        estimates = []
        for _ in range(2000):
            rx = np.zeros((16, 32), dtype=np.complex128)
            noise = np.sqrt(noise_power / 2) * (
                rng.standard_normal(16) + 1j * rng.standard_normal(16)
            )
            rx[:, 15] = noise
            estimates.append(estimate_noise_power_from_rx_dd(rx, config()))
        self.assertAlmostEqual(float(np.mean(estimates)), noise_power, delta=0.002)

    def test_rejects_incompatible_multi_pilot_configuration(self):
        cfg = config()
        cfg.pilot.pilot_delay_length = 2
        with self.assertRaises(ValueError):
            estimate_noise_power_from_rx_dd(np.zeros((16, 32), dtype=np.complex128), cfg)

    def test_rejects_wrong_shape_and_nonfinite_input(self):
        with self.assertRaises(ValueError):
            estimate_noise_power_from_rx_dd(np.zeros((32, 16), dtype=np.complex128), config())
        rx = np.zeros((16, 32), dtype=np.complex128)
        rx[0, 15] = np.nan
        with self.assertRaises(ValueError):
            estimate_noise_power_from_rx_dd(rx, config())


if __name__ == "__main__":
    unittest.main()

"""Tests for the target-free pilot-CE noise and channel estimators."""
from __future__ import annotations

import types
import unittest

import numpy as np

from models.supervisor.pilot_noise_estimator import (
    estimate_channel_matrix_from_rx_dd,
    estimate_noise_power_from_rx_dd,
    pilot_guard_noise_samples,
)


def config():
    return types.SimpleNamespace(
        otfs=types.SimpleNamespace(
            M=32, N=16, carrier_frequency_ghz=4.0, subcarrier_spacing_khz=15.0
        ),
        pilot=types.SimpleNamespace(
            pilot_delay_length=1, pilot_doppler_length=1, pilot_snr_db=25.0
        ),
        channel=types.SimpleNamespace(max_delay=4, num_paths=4),
        representation=types.SimpleNamespace(
            expected_data_symbols=368,
            expected_channel_shape=types.SimpleNamespace(rows=432, cols=368),
        ),
    )


class PilotGuardNoiseEstimatorTests(unittest.TestCase):
    def _frame_with_one_empty_ce_delay(self, rng, noise_power):
        rx = np.full((16, 32), 1e4 + 1e4j, dtype=np.complex128)
        noise = np.sqrt(noise_power / 2) * (
            rng.standard_normal(16) + 1j * rng.standard_normal(16)
        )
        # The CE region is columns 15..19. Four path delays occupy four
        # columns; column 17 is the single noise-only CE delay column.
        rx[:, 15:20] = 0.0
        for col in (15, 16, 18, 19):
            signal = 2.0 * (rng.standard_normal(16) + 1j * rng.standard_normal(16))
            rx[:, col] = signal + noise
        rx[:, 17] = noise
        return rx, noise

    def test_selects_minimum_energy_ce_delay_column_not_fixed_pilot_column(self):
        rng = np.random.default_rng(2026)
        rx, noise = self._frame_with_one_empty_ce_delay(rng, 0.04)
        selected = pilot_guard_noise_samples(rx, config())
        np.testing.assert_array_equal(selected, noise)
        self.assertAlmostEqual(
            estimate_noise_power_from_rx_dd(rx, config()),
            float(np.mean(np.abs(noise) ** 2)),
        )

    def test_estimate_is_unbiased_with_one_noise_only_ce_column(self):
        rng = np.random.default_rng(53)
        noise_power = 0.07
        estimates = []
        for _ in range(2000):
            rx, _ = self._frame_with_one_empty_ce_delay(rng, noise_power)
            estimates.append(estimate_noise_power_from_rx_dd(rx, config()))
        self.assertAlmostEqual(float(np.mean(estimates)), noise_power, delta=0.002)

    def test_rebuilds_channel_matrix_without_transmitted_or_true_channel_inputs(self):
        rx = np.zeros((16, 32), dtype=np.complex64)
        estimated = estimate_channel_matrix_from_rx_dd(rx, 0.1, config())
        self.assertEqual(estimated.shape, (432, 368))
        self.assertTrue(np.isfinite(estimated.real).all())
        self.assertTrue(np.isfinite(estimated.imag).all())

    def test_rejects_incompatible_multi_pilot_configuration(self):
        cfg = config()
        cfg.pilot.pilot_delay_length = 2
        with self.assertRaises(ValueError):
            estimate_noise_power_from_rx_dd(np.zeros((16, 32), dtype=np.complex128), cfg)

    def test_rejects_too_many_paths_for_noise_only_ce_column_assumption(self):
        cfg = config()
        cfg.channel.num_paths = 5
        with self.assertRaises(ValueError):
            estimate_noise_power_from_rx_dd(np.zeros((16, 32), dtype=np.complex128), cfg)

    def test_rejects_wrong_shape_and_nonfinite_input(self):
        with self.assertRaises(ValueError):
            estimate_noise_power_from_rx_dd(np.zeros((32, 16), dtype=np.complex128), config())
        rx = np.zeros((16, 32), dtype=np.complex128)
        rx[0, 17] = np.nan
        with self.assertRaises(ValueError):
            estimate_noise_power_from_rx_dd(rx, config())


if __name__ == "__main__":
    unittest.main()

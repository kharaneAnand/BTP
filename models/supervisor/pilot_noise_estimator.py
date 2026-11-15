"""Pilot-guard noise power estimator for the current OTFS grid.

The current simulator inserts a single embedded pilot at the centre of the DD
grid and generates channel delays only in [1, max_delay]. With the configured
rectangular pulse and symmetric delay guards, the received column at the pilot's
transmit delay index contains AWGN but no transmitted pilot/data contribution:
positive channel delays move the pilot response to later delay columns, while
the data guard prevents data leakage into this column.

For circular AWGN, the sample mean of |n|^2 is an unbiased estimate of the
complex noise variance. This function reads rx_dd and fixed public grid
configuration only. It never reads SNR metadata, tx_dd, h_dd, or channel paths.

This estimator is valid only for the current single-pilot, positive-delay
channel model. It deliberately raises on incompatible pilot settings rather
than silently applying an invalid assumption to a different waveform.
"""
from __future__ import annotations

import numpy as np


def pilot_guard_noise_samples(rx_dd: np.ndarray, config) -> np.ndarray:
    """Return the noise-only DD samples at the known pilot delay index."""
    received = np.asarray(rx_dd)
    expected = (int(config.otfs.N), int(config.otfs.M))
    if received.shape != expected:
        raise ValueError(f"rx_dd shape {received.shape} does not match {expected}.")
    if int(config.pilot.pilot_delay_length) != 1 or int(config.pilot.pilot_doppler_length) != 1:
        raise ValueError("Pilot-guard estimator currently requires one embedded pilot.")
    if int(config.channel.max_delay) <= 0:
        raise ValueError("Pilot-guard estimator requires strictly positive channel delays.")
    if not np.isfinite(received.real).all() or not np.isfinite(received.imag).all():
        raise ValueError("rx_dd contains non-finite values.")

    pilot_delay = (int(config.otfs.M) - 1) // 2
    # OTFSResGrid.setPilot2Center uses floor((M - pilot_delay_length) / 2).
    # Channel generation uses lmin=1, so no path response remains at this
    # transmit-pilot delay. The guard region protects it from data leakage.
    return received[:, pilot_delay].astype(np.complex128, copy=False)


def estimate_noise_power_from_rx_dd(rx_dd: np.ndarray, config) -> float:
    """Estimate complex AWGN variance from the known pilot-delay guard column."""
    samples = pilot_guard_noise_samples(rx_dd, config)
    estimate = float(np.mean(np.abs(samples) ** 2))
    if not np.isfinite(estimate) or estimate <= 0.0:
        raise FloatingPointError("Pilot-guard noise estimate is not finite and positive.")
    return estimate

"""Pilot-guard noise and channel estimation for the current single-pilot OTFS grid.

The channel generator creates four paths with integer delays in [0, max_delay]:
it first samples delays in [1, max_delay] and then moves the minimum-delay path
to zero. The embedded-pilot channel-estimation region spans the pilot delay
column through pilot_delay + max_delay. Therefore at most four of the five
delay columns contain pilot responses; at least one column has no pilot-path
response. Fractional Doppler spreads a path across Doppler bins but does not
change its delay column.

The provisional noise estimator selects the lowest-energy delay column inside
that pilot CE region and estimates complex AWGN variance by the sample mean of
|rx_dd|^2 across its Doppler bins. It reads only rx_dd and fixed public
configuration; it never reads SNR metadata, tx_dd, h_dd, or channel paths.

Important limitation: if multiple CE delay columns contain no path response,
selecting the minimum of several noise-only sample means introduces downward
selection bias. A multiplicative calibration may be fitted on training data
only. Do not deploy without held-out validation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def pilot_ce_column_powers(rx_dd: np.ndarray, config) -> tuple[np.ndarray, np.ndarray]:
    """Return delay-column powers and their grid indices in the pilot CE region."""
    received = np.asarray(rx_dd)
    expected = (int(config.otfs.N), int(config.otfs.M))
    if received.shape != expected:
        raise ValueError(f"rx_dd shape {received.shape} does not match {expected}.")
    if int(config.pilot.pilot_delay_length) != 1 or int(config.pilot.pilot_doppler_length) != 1:
        raise ValueError("Pilot-guard estimator currently requires one embedded pilot.")
    if int(config.channel.max_delay) <= 0:
        raise ValueError("Pilot-guard estimator requires positive max_delay.")
    if int(config.channel.num_paths) > int(config.channel.max_delay):
        raise ValueError("Estimator requires fewer paths than pilot CE delay columns.")
    if not np.isfinite(received.real).all() or not np.isfinite(received.imag).all():
        raise ValueError("rx_dd contains non-finite values.")

    pilot_delay = (int(config.otfs.M) - 1) // 2
    delay_indices = np.arange(pilot_delay, pilot_delay + int(config.channel.max_delay) + 1)
    if delay_indices[-1] >= int(config.otfs.M):
        raise ValueError("Pilot CE delay region exceeds the DD grid.")
    ce = received[:, delay_indices]
    powers = np.mean(np.abs(ce) ** 2, axis=0)
    return powers.astype(np.float64), delay_indices


def pilot_guard_noise_column_index(rx_dd: np.ndarray, config) -> int:
    """Select the minimum-energy delay column in the embedded-pilot CE region."""
    powers, delay_indices = pilot_ce_column_powers(rx_dd, config)
    return int(delay_indices[int(np.argmin(powers))])


def pilot_guard_noise_samples(rx_dd: np.ndarray, config) -> np.ndarray:
    """Return Doppler samples from the selected minimum-energy CE delay column."""
    received = np.asarray(rx_dd)
    selected = pilot_guard_noise_column_index(received, config)
    return received[:, selected].astype(np.complex128, copy=False)


def estimate_noise_power_from_rx_dd(rx_dd: np.ndarray, config) -> float:
    """Estimate complex AWGN variance from the selected pilot-CE delay column."""
    samples = pilot_guard_noise_samples(rx_dd, config)
    estimate = float(np.mean(np.abs(samples) ** 2))
    if not np.isfinite(estimate) or estimate <= 0.0:
        raise FloatingPointError("Pilot-CE noise estimate is not finite and positive.")
    return estimate


def estimate_channel_matrix_from_rx_dd(
    rx_dd: np.ndarray,
    noise_power_estimate: float,
    config,
) -> np.ndarray:
    """Rebuild the pilot-derived data-domain channel matrix without simulator labels.

    The current V1 pilot power is defined by a configured pilot-SNR target, so
    its amplitude is reconstructed from the estimated noise power. A V2 config
    may instead provide a fixed, known pilot.power_linear value.
    """
    if not np.isfinite(noise_power_estimate) or noise_power_estimate <= 0.0:
        raise ValueError("noise_power_estimate must be finite and positive.")

    otfs_root = Path(__file__).resolve().parents[2] / "Phy_Mod_OTFS"
    if str(otfs_root) not in sys.path:
        sys.path.insert(0, str(otfs_root))
    from OTFS import OTFS
    from OTFSResGrid import OTFSResGrid

    received = np.asarray(rx_dd)
    expected_grid = (int(config.otfs.N), int(config.otfs.M))
    if received.shape != expected_grid:
        raise ValueError(f"rx_dd shape {received.shape} does not match {expected_grid}.")

    pilot_power = getattr(config.pilot, "power_linear", None)
    if pilot_power is None:
        pilot_power = noise_power_estimate * 10.0 ** (
            float(config.pilot.pilot_snr_db) / 10.0
        )
    pilot_power = float(pilot_power)
    if not np.isfinite(pilot_power) or pilot_power <= 0.0:
        raise ValueError("Pilot power must be finite and positive.")

    resource_grid = OTFSResGrid(int(config.otfs.M), int(config.otfs.N))
    resource_grid.setPulse2Recta()
    resource_grid.setPilot2Center(
        int(config.pilot.pilot_delay_length),
        int(config.pilot.pilot_doppler_length),
    )
    resource_grid.setGuard(
        int(config.channel.max_delay),
        int(config.channel.max_delay),
        guard_doppl_full=True,
    )
    resource_grid.map(
        np.zeros(int(config.representation.expected_data_symbols), dtype=np.complex64),
        pilots_pow=pilot_power,
    )
    resource_grid.setContent(received)

    otfs = OTFS(
        fc=float(config.otfs.carrier_frequency_ghz),
        fq_sp=float(config.otfs.subcarrier_spacing_khz),
    )
    otfs.modulate(resource_grid)
    _, gains, delays, dopplers = resource_grid.demap(
        isData=False,
        threshold=3.0 * np.sqrt(float(noise_power_estimate)),
    )

    expected_channel = config.representation.expected_channel_shape
    expected_shape = (int(expected_channel.rows), int(expected_channel.cols))
    if gains is None or delays is None or dopplers is None or len(np.atleast_1d(gains)) == 0:
        return np.zeros(expected_shape, dtype=np.complex64)

    estimated = np.asarray(otfs.getChannel(gains, delays, dopplers), dtype=np.complex64)
    if estimated.shape != expected_shape:
        raise ValueError(
            f"Estimated channel shape {estimated.shape} does not match {expected_shape}."
        )
    return estimated

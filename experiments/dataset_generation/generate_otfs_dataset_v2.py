"""Generate OTFS V2 with fixed pilot power and a receiver-side noise/CSI front end.

Unlike V1, V2 does not use simulator noise variance to set pilot amplitude or
the deployable channel-estimation threshold. Pilot power is fixed and public.
After the train/validation/test split is created, a scalar noise-estimator
calibration is fitted on the training split only. Every frame's noise estimate
and h_hat are then reconstructed from rx_dd and fixed waveform configuration.
The true noise power and true channel are retained only for offline audit.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OTFS_ROOT = ROOT / "Phy_Mod_OTFS"
sys.path.insert(0, str(OTFS_ROOT))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from OTFS import OTFS
from OTFSResGrid import OTFSResGrid
from experiments.dataset_generation.generate_otfs_dataset import (
    calculate_doppler,
    calculate_num_data_symbols,
    get_num_paths,
    qpsk_symbols,
    validate_dataset_contract,
)
from experiments.dataset_preprocessing.preprocess_otfs_dataset import (
    create_split,
    validate_split,
)
from models.supervisor.pilot_noise_estimator import (
    estimate_channel_matrix_from_rx_dd,
    estimate_noise_power_from_rx_dd,
)
from src.config.loader import load_config


def generate_dataset_v2(config, dataset_root: Path, seed: int):
    dataset_root = dataset_root.resolve()
    config.dataset.root = str(dataset_root)
    config.reproducibility.seed = int(seed)
    config.split.random_seed = int(seed)
    config.dataset.split_metadata_file = "dataset_v2_split.csv"

    raw_dir = dataset_root / config.dataset.raw_dir
    metadata_dir = dataset_root / config.dataset.metadata_dir
    processed_dir = dataset_root / config.dataset.processed_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    M, N = int(config.otfs.M), int(config.otfs.N)
    snr_values = list(config.channel.snr_db)
    velocities = list(config.channel.velocity_kmh)
    frames_per_condition = int(config.dataset.frames_per_condition)
    num_data_symbols = calculate_num_data_symbols(config)
    pilot_power = float(config.pilot.power_linear)
    if pilot_power <= 0.0:
        raise ValueError("V2 fixed pilot power must be positive.")

    expected_samples = int(config.dataset.expected_samples)
    total_samples = len(snr_values) * len(velocities) * frames_per_condition
    if total_samples != expected_samples:
        raise ValueError(f"V2 config expects {expected_samples} samples, sweep generates {total_samples}.")

    metadata_rows = []
    sample_id = 0
    for velocity in velocities:
        kmax = calculate_doppler(
            float(velocity),
            float(config.otfs.carrier_frequency_ghz),
            float(config.otfs.subcarrier_spacing_khz),
            N,
        )
        num_paths = get_num_paths(config, kmax)
        for snr_db in snr_values:
            noise_power = 10.0 ** (-float(snr_db) / 10.0)
            pilot_snr_db_actual = 10.0 * np.log10(pilot_power / noise_power)
            for frame in range(frames_per_condition):
                tx_data = qpsk_symbols(num_data_symbols, rng)
                rg = OTFSResGrid(M, N)
                rg.setPulse2Recta()
                rg.setPilot2Center(
                    int(config.pilot.pilot_delay_length),
                    int(config.pilot.pilot_doppler_length),
                )
                rg.setGuard(
                    int(config.channel.max_delay),
                    int(config.channel.max_delay),
                    guard_doppl_full=True,
                )
                rg.map(tx_data, pilots_pow=pilot_power)

                otfs = OTFS(
                    fc=float(config.otfs.carrier_frequency_ghz),
                    fq_sp=float(config.otfs.subcarrier_spacing_khz),
                )
                otfs.modulate(rg)
                otfs.setChannel(
                    num_paths,
                    int(config.channel.max_delay),
                    kmax,
                    force_frac=bool(config.channel.force_fractional_doppler),
                    rng=rng,
                )
                otfs.passChannel(noise_power, rng=rng)
                his, lis, kis = otfs.getCSI(sort_by_delay_doppler=True)
                rg_rx = otfs.demodulate()
                rx_dd = np.asarray(rg_rx.getContent(), dtype=np.complex64)
                h_dd = np.asarray(otfs.getChannel(his, lis, kis), dtype=np.complex64)

                # Oracle threshold is retained only as a non-deployable audit baseline.
                _, his_oracle, lis_oracle, kis_oracle = rg_rx.demap(
                    isData=False,
                    threshold=3.0 * np.sqrt(noise_power),
                )
                if (
                    his_oracle is None or lis_oracle is None or kis_oracle is None
                    or len(np.atleast_1d(his_oracle)) == 0
                ):
                    h_hat_oracle = np.zeros_like(h_dd)
                else:
                    h_hat_oracle = np.asarray(
                        otfs.getChannel(his_oracle, lis_oracle, kis_oracle),
                        dtype=np.complex64,
                    )

                validate_dataset_contract(config, rx_dd, tx_data, h_dd, h_hat_oracle)
                filename = f"sample_{sample_id:06d}.npz"
                np.savez_compressed(
                    raw_dir / filename,
                    rx_dd=rx_dd,
                    tx_dd=tx_data,
                    h_dd=h_dd,
                    h_hat_oracle=h_hat_oracle,
                    noise_power=np.asarray(noise_power, dtype=np.float32),
                    pilot_power=np.asarray(pilot_power, dtype=np.float32),
                    his=his,
                    lis=lis,
                    kis=kis,
                )
                metadata_rows.append({
                    "sample_id": sample_id,
                    "file": filename,
                    "snr_db": int(snr_db),
                    "velocity_kmh": int(velocity),
                    "kmax": float(kmax),
                    "num_paths": int(num_paths),
                    "max_delay": int(config.channel.max_delay),
                    "noise_power": float(noise_power),
                    "pilot_power": pilot_power,
                    "pilot_snr_db_actual": float(pilot_snr_db_actual),
                    "M": M,
                    "N": N,
                })
                sample_id += 1

    metadata = pd.DataFrame(metadata_rows)
    metadata_path = metadata_dir / "dataset_v2_metadata.csv"
    metadata.to_csv(metadata_path, index=False)
    split_metadata = create_split(metadata, config)
    validate_split(split_metadata, config)
    split_path = processed_dir / "dataset_v2_split.csv"
    split_metadata.to_csv(split_path, index=False)

    train_rows = split_metadata.loc[
        split_metadata["split"].astype(str).str.lower() == "train"
    ].reset_index(drop=True)
    train_ratios = []
    for _, row in train_rows.iterrows():
        with np.load(raw_dir / str(row["file"]), allow_pickle=False) as sample:
            rx = np.asarray(sample["rx_dd"])
        raw_estimate = estimate_noise_power_from_rx_dd(rx, config)
        train_ratios.append(float(row["noise_power"]) / raw_estimate)
    calibration_scale = float(np.exp(np.mean(np.log(np.asarray(train_ratios)))))

    # Freeze the train-only calibration, then reconstruct deployable CSI for all splits.
    for _, row in split_metadata.iterrows():
        sample_path = raw_dir / str(row["file"])
        with np.load(sample_path, allow_pickle=False) as sample:
            rx = np.asarray(sample["rx_dd"], dtype=np.complex64)
            payload = {key: np.asarray(sample[key]) for key in sample.files}
        raw_estimate = estimate_noise_power_from_rx_dd(rx, config)
        calibrated_estimate = raw_estimate * calibration_scale
        h_hat_estimated = estimate_channel_matrix_from_rx_dd(
            rx, calibrated_estimate, config
        )
        payload["noise_estimate_raw"] = np.asarray(raw_estimate, dtype=np.float32)
        payload["noise_estimate"] = np.asarray(calibrated_estimate, dtype=np.float32)
        payload["h_hat"] = h_hat_estimated
        np.savez_compressed(sample_path, **payload)

    calibration = {
        "dataset_version": "v2",
        "seed": int(seed),
        "pilot_power_linear_fixed": pilot_power,
        "pilot_power_db_relative_to_unit_data": float(10.0 * np.log10(pilot_power)),
        "noise_estimator": "minimum mean power among the five pilot CE delay columns",
        "calibration_method": "geometric-mean multiplicative correction from training split only",
        "calibration_scale": calibration_scale,
        "training_samples": len(train_rows),
        "validation_or_test_noise_labels_used_for_calibration": False,
        "simulator_noise_power_used_for_deployable_h_hat": False,
        "simulator_noise_power_used_for_deployable_noise_input": False,
        "oracle_h_hat_saved_separately": True,
    }
    (processed_dir / "noise_estimator_calibration_v2.json").write_text(
        json.dumps(calibration, indent=2)
    )
    return raw_dir, split_metadata, calibration


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.seed < 57:
        raise ValueError("Use fresh independent V2 seeds 57 or higher.")
    config = load_config(ROOT / "configs" / "experiment_v2.yaml")
    raw_dir, metadata, calibration = generate_dataset_v2(
        config, args.output_dir / "dataset", args.seed
    )
    print(json.dumps({
        "seed": args.seed,
        "raw_dir": str(raw_dir),
        "split_counts": metadata["split"].value_counts().to_dict(),
        "noise_calibration": calibration,
    }, indent=2))


if __name__ == "__main__":
    main()

"""Evaluate and training-calibrate the target-free pilot-CE noise estimator."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.supervisor.pilot_noise_estimator import (
    estimate_noise_power_from_rx_dd,
    pilot_guard_noise_column_index,
)
from run_pipeline import prepare_generated_dataset
from src.config.loader import load_config


def stratified_bootstrap_mean(values: np.ndarray, metadata: pd.DataFrame,
                              seed: int, draws: int = 10000) -> list[float]:
    rng = np.random.default_rng(seed)
    groups = [g.index.to_numpy() for _, g in metadata.groupby(
        ["snr_db", "velocity_kmh"], sort=True
    )]
    samples = np.empty(draws, dtype=np.float64)
    for b in range(draws):
        ids = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        samples[b] = float(np.mean(values[ids]))
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def estimate_split(metadata: pd.DataFrame, raw_dir: Path, config, include_diagnostics: bool):
    rows = []
    column_rows = []
    for idx, row in metadata.reset_index(drop=True).iterrows():
        with np.load(raw_dir / str(row["file"]), allow_pickle=False) as sample:
            rx_dd = np.asarray(sample["rx_dd"])
            true_path_delays = np.asarray(sample["lis"], dtype=int) if include_diagnostics else None
        raw_estimate = estimate_noise_power_from_rx_dd(rx_dd, config)
        truth = float(row["noise_power"])
        selected_col = pilot_guard_noise_column_index(rx_dd, config)
        record = {
            "sample_index": idx,
            "snr_db": int(row["snr_db"]),
            "velocity_kmh": int(row["velocity_kmh"]),
            "raw_estimated_noise_power": raw_estimate,
            "simulator_noise_power_for_offline_audit_only": truth,
            "raw_ratio_estimated_to_true": raw_estimate / truth,
            "raw_relative_error": (raw_estimate - truth) / truth,
            "raw_absolute_log_error": abs(float(np.log(raw_estimate / truth))),
            "selected_noise_delay_column": selected_col,
        }
        if include_diagnostics:
            column_power = np.mean(np.abs(rx_dd) ** 2, axis=0)
            record["true_path_delays_offline_audit_only"] = json.dumps(true_path_delays.tolist())
            record["unique_true_path_delays_offline_audit_only"] = len(np.unique(true_path_delays))
            record["minimum_delay_column_power"] = float(column_power.min())
            record["minimum_delay_column_index"] = int(column_power.argmin())
            record.update({f"delay_col_power_{j:02d}": float(v) for j, v in enumerate(column_power)})
            for delay_index, power in enumerate(column_power):
                column_rows.append({
                    "sample_index": idx,
                    "snr_db": int(row["snr_db"]),
                    "velocity_kmh": int(row["velocity_kmh"]),
                    "delay_index": delay_index,
                    "mean_received_power": float(power),
                    "simulator_noise_power_for_offline_audit_only": truth,
                    "power_to_noise_ratio": float(power / truth),
                    "contains_true_path_delay_offline_audit_only": bool(
                        delay_index - ((int(config.otfs.M) - 1) // 2) in true_path_delays
                    ),
                })
        rows.append(record)
    return pd.DataFrame(rows), pd.DataFrame(column_rows)


def summarize_split(results: pd.DataFrame, metadata: pd.DataFrame, scale: float, seed: int):
    calibrated = results["raw_estimated_noise_power"].to_numpy(dtype=np.float64) * scale
    truth = results["simulator_noise_power_for_offline_audit_only"].to_numpy(dtype=np.float64)
    ratio = calibrated / truth
    log_error = np.abs(np.log(ratio))
    return {
        "frames": len(results),
        "calibrated_ratio_mean": float(ratio.mean()),
        "calibrated_ratio_median": float(np.median(ratio)),
        "calibrated_ratio_std": float(ratio.std(ddof=1)),
        "calibrated_ratio_mean_ci95_condition_stratified": stratified_bootstrap_mean(
            ratio, metadata.reset_index(drop=True), seed
        ),
        "calibrated_absolute_log_error_mean": float(log_error.mean()),
        "calibrated_absolute_log_error_median": float(np.median(log_error)),
        "calibrated_within_20_percent_fraction": float(np.mean(np.abs(ratio - 1.0) <= 0.20)),
        "calibrated_within_30_percent_fraction": float(np.mean(np.abs(ratio - 1.0) <= 0.30)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.seed < 55:
        raise ValueError("Use fresh independent seeds 55 or higher.")
    random.seed(args.seed)
    np.random.seed(args.seed)

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = load_config(ROOT / "configs" / "experiment_v1.yaml")
    raw_dir, _, full_meta, run_config = prepare_generated_dataset(
        config, output / "generated", args.seed
    )
    train_meta = full_meta.loc[full_meta.split.astype(str).str.lower() == "train"].reset_index(drop=True)
    validation_meta = full_meta.loc[full_meta.split.astype(str).str.lower() == "validation"].reset_index(drop=True)
    test_meta = full_meta.loc[full_meta.split.astype(str).str.lower() == "test"].reset_index(drop=True)

    # The multiplicative calibration is fitted on training labels only.
    train_rows, _ = estimate_split(train_meta, raw_dir, run_config, include_diagnostics=False)
    train_ratio = (
        train_rows["simulator_noise_power_for_offline_audit_only"].to_numpy(dtype=np.float64)
        / train_rows["raw_estimated_noise_power"].to_numpy(dtype=np.float64)
    )
    calibration_scale = float(np.exp(np.mean(np.log(train_ratio))))

    validation_rows, _ = estimate_split(validation_meta, raw_dir, run_config, include_diagnostics=False)
    test_rows, column_frame = estimate_split(test_meta, raw_dir, run_config, include_diagnostics=True)

    for frame in (validation_rows, test_rows):
        frame["calibrated_estimated_noise_power"] = frame["raw_estimated_noise_power"] * calibration_scale
        frame["calibrated_ratio_estimated_to_true"] = (
            frame["calibrated_estimated_noise_power"]
            / frame["simulator_noise_power_for_offline_audit_only"]
        )
        frame["calibrated_relative_error"] = frame["calibrated_ratio_estimated_to_true"] - 1.0
        frame["calibrated_absolute_log_error"] = np.abs(
            np.log(frame["calibrated_ratio_estimated_to_true"])
        )

    test_rows.to_csv(output / "per_sample_noise_estimates.csv", index=False)
    column_frame.to_csv(output / "delay_column_power_diagnostics.csv", index=False)
    condition = test_rows.groupby(["snr_db", "velocity_kmh"], as_index=False).agg(
        frames=("sample_index", "count"),
        mean_raw_ratio=("raw_ratio_estimated_to_true", "mean"),
        mean_calibrated_ratio=("calibrated_ratio_estimated_to_true", "mean"),
        median_calibrated_ratio=("calibrated_ratio_estimated_to_true", "median"),
        calibrated_relative_error_std=("calibrated_relative_error", "std"),
        calibrated_log_error_mean=("calibrated_absolute_log_error", "mean"),
    )
    condition.to_csv(output / "per_condition_noise_estimates.csv", index=False)

    selected_counts = test_rows["selected_noise_delay_column"].value_counts().sort_index().to_dict()
    summary = {
        "seed": args.seed,
        "estimator": "minimum mean received power among pilot CE delay columns",
        "pilot_ce_delay_indices": list(range(
            (int(run_config.otfs.M) - 1) // 2,
            (int(run_config.otfs.M) - 1) // 2 + int(run_config.channel.max_delay) + 1
        )),
        "pilot_samples_per_column": int(run_config.otfs.N),
        "training_calibration_samples": len(train_rows),
        "calibration_method": "geometric-mean multiplicative correction fitted on training split only",
        "calibration_scale": calibration_scale,
        "inputs_used_at_inference": ["rx_dd", "fixed public OTFS/pilot/channel configuration", "frozen training calibration scalar"],
        "simulator_noise_power_used_at_inference": False,
        "tx_dd_or_h_dd_used_at_inference": False,
        "simulator_noise_power_used_for_offline_calibration_and_audit_only": True,
        "selected_noise_column_counts_test": selected_counts,
        "raw_test": summarize_split(test_rows, test_meta, 1.0, args.seed + 8000),
        "calibrated_validation": summarize_split(validation_rows, validation_meta, calibration_scale, args.seed + 8500),
        "calibrated_test": summarize_split(test_rows, test_meta, calibration_scale, args.seed + 9000),
        "condition_results": condition.to_dict(orient="records"),
        "diagnostic_note": "Per-delay-column powers and true path delays are exported for offline diagnosis only; they are not inputs to the estimator.",
        "limitations": [
            "The minimum-energy CE-column estimator can be downward-biased when multiple noise-only columns compete; the training-only scalar calibration corrects average bias but not all frame-level variation.",
            "Only 16 complex samples are available per selected column, so per-frame estimates retain sampling variance.",
            "This experiment evaluates noise estimation, not receiver BER improvement. Receiver fine-tuning with estimated noise is a separate experiment.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

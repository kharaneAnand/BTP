"""Evaluate candidate pilot-CE noise estimators on fresh held-out OTFS datasets."""
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
    test_meta = full_meta.loc[full_meta.split.astype(str).str.lower() == "test"].reset_index(drop=True)
    pilot_delay = (int(run_config.otfs.M) - 1) // 2
    ce_delay_indices = list(range(pilot_delay, pilot_delay + int(run_config.channel.max_delay) + 1))

    rows = []
    column_rows = []
    for idx, row in test_meta.iterrows():
        with np.load(raw_dir / str(row["file"]), allow_pickle=False) as sample:
            rx_dd = np.asarray(sample["rx_dd"])
            true_path_delays = np.asarray(sample["lis"], dtype=int)
        estimate = estimate_noise_power_from_rx_dd(rx_dd, run_config)
        selected_col = pilot_guard_noise_column_index(rx_dd, run_config)
        truth = float(row["noise_power"])
        column_power = np.mean(np.abs(rx_dd) ** 2, axis=0)
        record = {
            "sample_index": idx,
            "snr_db": int(row["snr_db"]),
            "velocity_kmh": int(row["velocity_kmh"]),
            "estimated_noise_power": estimate,
            "selected_noise_delay_column": selected_col,
            "simulator_noise_power_for_offline_audit_only": truth,
            "ratio_estimated_to_true": estimate / truth,
            "relative_error": (estimate - truth) / truth,
            "absolute_log_error": abs(float(np.log(estimate / truth))),
            "true_path_delays_offline_audit_only": json.dumps(true_path_delays.tolist()),
            "unique_true_path_delays_offline_audit_only": len(np.unique(true_path_delays)),
            "minimum_delay_column_power": float(column_power.min()),
            "minimum_delay_column_index": int(column_power.argmin()),
        }
        record.update({f"delay_col_power_{j:02d}": float(v) for j, v in enumerate(column_power)})
        rows.append(record)
        for delay_index, power in enumerate(column_power):
            column_rows.append({
                "sample_index": idx,
                "snr_db": int(row["snr_db"]),
                "velocity_kmh": int(row["velocity_kmh"]),
                "delay_index": delay_index,
                "mean_received_power": float(power),
                "simulator_noise_power_for_offline_audit_only": truth,
                "power_to_noise_ratio": float(power / truth),
                "contains_true_path_delay_offline_audit_only": bool(delay_index - pilot_delay in true_path_delays),
            })

    results = pd.DataFrame(rows)
    results.to_csv(output / "per_sample_noise_estimates.csv", index=False)
    column_frame = pd.DataFrame(column_rows)
    column_frame.to_csv(output / "delay_column_power_diagnostics.csv", index=False)
    condition = results.groupby(["snr_db", "velocity_kmh"], as_index=False).agg(
        frames=("sample_index", "count"),
        mean_estimated_noise=("estimated_noise_power", "mean"),
        mean_true_noise=("simulator_noise_power_for_offline_audit_only", "mean"),
        mean_ratio=("ratio_estimated_to_true", "mean"),
        median_ratio=("ratio_estimated_to_true", "median"),
        relative_error_std=("relative_error", "std"),
        log_error_mean=("absolute_log_error", "mean"),
    )
    condition.to_csv(output / "per_condition_noise_estimates.csv", index=False)

    ratios = results["ratio_estimated_to_true"].to_numpy(dtype=np.float64)
    log_errors = results["absolute_log_error"].to_numpy(dtype=np.float64)
    summary = {
        "seed": args.seed,
        "test_frames": len(results),
        "estimator": "minimum mean received power among pilot CE delay columns",
        "pilot_ce_delay_indices": ce_delay_indices,
        "pilot_samples_per_column": int(run_config.otfs.N),
        "inputs_used": ["rx_dd", "fixed public OTFS/pilot/channel configuration"],
        "simulator_noise_power_used_at_inference": False,
        "tx_dd_or_h_dd_used_at_inference": False,
        "simulator_noise_power_used_for_offline_audit_only": True,
        "assumption_under_test": "four or fewer channel paths occupy at most four of the five pilot CE delay columns; fractional Doppler spreads within a delay column",
        "selected_noise_column_counts": results["selected_noise_delay_column"].value_counts().sort_index().to_dict(),
        "ratio_estimated_to_true": {
            "mean": float(ratios.mean()),
            "median": float(np.median(ratios)),
            "std": float(ratios.std(ddof=1)),
            "mean_ci95_condition_stratified": stratified_bootstrap_mean(ratios, test_meta, args.seed + 9000),
        },
        "absolute_log_error": {
            "mean": float(log_errors.mean()),
            "median": float(np.median(log_errors)),
        },
        "within_20_percent_fraction": float(np.mean(np.abs(ratios - 1.0) <= 0.20)),
        "within_30_percent_fraction": float(np.mean(np.abs(ratios - 1.0) <= 0.30)),
        "mean_received_power_by_delay_column_overall": column_frame.groupby("delay_index").mean_received_power.mean().to_dict(),
        "condition_results": condition.to_dict(orient="records"),
        "diagnostic_note": "Per-delay-column powers and true path delays are exported for offline diagnosis only; they are not inputs to the estimator.",
        "limitations": [
            "Selecting the minimum among multiple noise-only delay columns can introduce downward selection bias.",
            "Only 16 complex samples are available per selected column, so per-frame estimates have sampling variance.",
            "This experiment evaluates noise estimation, not receiver BER improvement. Receiver fine-tuning with estimated noise is a separate experiment.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

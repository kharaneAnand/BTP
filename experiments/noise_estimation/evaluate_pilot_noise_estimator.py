"""Evaluate the pilot-guard noise estimator on fresh held-out OTFS datasets."""
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

from models.supervisor.pilot_noise_estimator import estimate_noise_power_from_rx_dd
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

    rows = []
    for idx, row in test_meta.iterrows():
        with np.load(raw_dir / str(row["file"]), allow_pickle=False) as sample:
            rx_dd = np.asarray(sample["rx_dd"])
        estimate = estimate_noise_power_from_rx_dd(rx_dd, run_config)
        truth = float(row["noise_power"])
        rows.append({
            "sample_index": idx,
            "snr_db": int(row["snr_db"]),
            "velocity_kmh": int(row["velocity_kmh"]),
            "estimated_noise_power": estimate,
            "simulator_noise_power_for_offline_audit_only": truth,
            "ratio_estimated_to_true": estimate / truth,
            "relative_error": (estimate - truth) / truth,
            "absolute_log_error": abs(float(np.log(estimate / truth))),
        })

    results = pd.DataFrame(rows)
    results.to_csv(output / "per_sample_noise_estimates.csv", index=False)
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
        "estimator": "sample mean of |rx_dd[:, pilot_delay]|^2",
        "pilot_delay_index": (int(run_config.otfs.M) - 1) // 2,
        "pilot_samples_per_frame": int(run_config.otfs.N),
        "inputs_used": ["rx_dd", "fixed public OTFS/pilot/channel configuration"],
        "simulator_noise_power_used_at_inference": False,
        "tx_dd_or_h_dd_used_at_inference": False,
        "simulator_noise_power_used_for_offline_audit_only": True,
        "assumption": "single embedded pilot; positive channel delays [1,max_delay]; configured guard geometry isolates pilot-delay column from signal",
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
        "condition_results": condition.to_dict(orient="records"),
        "limitations": [
            "Only 16 complex guard samples are available per frame, so per-frame noise estimates have sampling variance.",
            "The estimator depends on the current positive-delay and pilot/guard layout; it must be re-derived if the waveform or channel delay support changes.",
            "This experiment evaluates noise estimation, not receiver BER improvement. Receiver fine-tuning with estimated noise is a separate experiment.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

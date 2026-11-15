"""Aggregate frozen multi-seed noise-aware OTFS receiver experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def stratified_bootstrap(delta: np.ndarray, frame: pd.DataFrame, seed: int, draws: int = 20000):
    groups = [
        group.index.to_numpy()
        for _, group in frame.groupby(["seed", "snr_db", "velocity_kmh"], sort=True)
    ]
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=np.float64)
    for b in range(draws):
        ids = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        samples[b] = float(delta[ids].mean())
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def find_column(frame: pd.DataFrame, candidates: list[str]) -> str:
    for name in candidates:
        if name in frame.columns:
            return name
    raise KeyError(f"None of the expected columns were found: {candidates}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[58, 59, 60, 61])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_frames = []
    summaries = {}
    for seed in args.seeds:
        seed_dir = args.results_root / f"seed{seed}"
        if not (seed_dir / "summary.json").exists():
            seed_dir = args.results_root / f"noise-aware-seed-{seed}"
        if not (seed_dir / "summary.json").exists():
            raise FileNotFoundError(f"Missing completed result summary for seed {seed}: {seed_dir}")
        summary = json.loads((seed_dir / "summary.json").read_text())
        metrics = pd.read_csv(seed_dir / "selected_policy_metrics.csv")
        if len(metrics) != int(summary["test_frames"]):
            raise ValueError(f"Seed {seed} policy metrics do not match test frame count.")
        metrics["seed"] = seed
        all_frames.append(metrics)
        summaries[seed] = summary

    frame = pd.concat(all_frames, ignore_index=True)
    frame = frame.sort_values(["seed", "sample_index"]).reset_index(drop=True)

    # Resolve stable output names from the policy merge.
    cols = {}
    for policy in ("velocity", "expanded"):
        for metric in ("ber", "ser", "nmse"):
            cols[(policy, "static", metric)] = find_column(
                frame, [f"static_{metric}_{policy}", f"static_{metric}"]
            )
            cols[(policy, "routed", metric)] = find_column(
                frame, [f"routed_{metric}_{policy}", f"routed_{metric}"]
            )

    per_seed_rows = []
    for seed in args.seeds:
        subset = frame.loc[frame.seed == seed].reset_index(drop=True)
        row = {"seed": seed, "test_frames": len(subset)}
        for metric in ("ber", "ser", "nmse"):
            static = subset[cols[("expanded", "static", metric)]].to_numpy(dtype=float)
            expanded = subset[cols[("expanded", "routed", metric)]].to_numpy(dtype=float)
            velocity = subset[cols[("velocity", "routed", metric)]].to_numpy(dtype=float)
            delta = expanded - static
            ci = stratified_bootstrap(delta, subset, 7000 + seed + len(metric))
            row[f"static_{metric}"] = float(static.mean())
            row[f"expanded_{metric}"] = float(expanded.mean())
            row[f"expanded_minus_static_{metric}"] = float(delta.mean())
            row[f"expanded_minus_static_{metric}_ci95_low"] = ci[0]
            row[f"expanded_minus_static_{metric}_ci95_high"] = ci[1]
            row[f"velocity_only_{metric}"] = float(velocity.mean())
        row["noise_estimation_ratio_mean"] = float(
            summaries[seed]["noise_estimation_test"]["estimated_to_true_ratio_mean"]
        )
        row["estimated_channel_nmse_mean"] = float(
            summaries[seed]["channel_estimation_nmse"]["estimated_front_end_mean"]
        )
        row["oracle_channel_nmse_mean"] = float(
            summaries[seed]["channel_estimation_nmse"]["oracle_front_end_mean"]
        )
        row["expanded_selected_receiver_counts"] = json.dumps(
            summaries[seed]["single_pass_inference"]["receiver_selection_counts"],
            sort_keys=True,
        )
        per_seed_rows.append(row)
    per_seed = pd.DataFrame(per_seed_rows)
    per_seed.to_csv(args.output_dir / "per_seed_summary.csv", index=False)

    pooled = {}
    for metric in ("ber", "ser", "nmse"):
        static = frame[cols[("expanded", "static", metric)]].to_numpy(dtype=float)
        expanded = frame[cols[("expanded", "routed", metric)]].to_numpy(dtype=float)
        velocity = frame[cols[("velocity", "routed", metric)]].to_numpy(dtype=float)
        delta = expanded - static
        incremental = expanded - velocity
        ci = stratified_bootstrap(delta, frame, 17000 + len(metric))
        inc_ci = stratified_bootstrap(incremental, frame, 19000 + len(metric))
        pooled[metric] = {
            "static_generalist_mean": float(static.mean()),
            "expanded_bank_mean": float(expanded.mean()),
            "expanded_minus_static_mean": float(delta.mean()),
            "expanded_minus_static_ci95": ci,
            "velocity_only_router_mean": float(velocity.mean()),
            "expanded_minus_velocity_only_mean": float(incremental.mean()),
            "expanded_minus_velocity_only_ci95": inc_ci,
        }

    per_condition_rows = []
    for (snr, velocity), group in frame.groupby(["snr_db", "velocity_kmh"], sort=True):
        row = {"snr_db": int(snr), "velocity_kmh": int(velocity), "frames": len(group)}
        for metric in ("ber", "ser", "nmse"):
            row[f"static_{metric}"] = float(group[cols[("expanded", "static", metric)]].mean())
            row[f"velocity_only_{metric}"] = float(group[cols[("velocity", "routed", metric)]].mean())
            row[f"expanded_{metric}"] = float(group[cols[("expanded", "routed", metric)]].mean())
            row[f"expanded_minus_static_{metric}"] = (
                row[f"expanded_{metric}"] - row[f"static_{metric}"]
            )
        per_condition_rows.append(row)
    pd.DataFrame(per_condition_rows).to_csv(
        args.output_dir / "per_condition_pooled.csv", index=False
    )

    ber_upper = pooled["ber"]["expanded_minus_static_ci95"][1]
    ser_margin = 0.01 * pooled["ser"]["static_generalist_mean"]
    nmse_margin = 0.01 * pooled["nmse"]["static_generalist_mean"]
    ser_upper = pooled["ser"]["expanded_minus_static_ci95"][1]
    nmse_upper = pooled["nmse"]["expanded_minus_static_ci95"][1]
    seed_ber_wins = int((per_seed["expanded_minus_static_ber"] < 0).sum())
    integrity = all(
        bool(s["test_labels_used_for_router_fit_or_actions"] is False)
        and bool(s["test_h_dd_or_tx_dd_used_before_inference_completed"] is False)
        and bool(s["simulator_noise_power_used_for_deployable_receiver_input"] is False)
        and bool(s["simulator_h_hat_used_for_deployable_receiver_input"] is False)
        and sum(s["single_pass_inference"]["receiver_selection_counts"].values()) == int(s["test_frames"])
        for s in summaries.values()
    )
    primary_accuracy_gate = (
        ber_upper < 0.0
        and seed_ber_wins >= 3
        and ser_upper < ser_margin
        and nmse_upper < nmse_margin
        and integrity
    )
    # A separate frozen unseen-SNR/velocity evaluation is intentionally not
    # inferred from these in-range V2 seeds.
    unseen_condition_evaluation_completed = False
    overall_paper_ready = primary_accuracy_gate and unseen_condition_evaluation_completed
    pooled_summary = {
        "seeds": args.seeds,
        "test_frames": len(frame),
        "pooled_metrics": pooled,
        "seed_level_ber_wins": seed_ber_wins,
        "seed_count": len(args.seeds),
        "ser_noninferiority_margin_absolute": ser_margin,
        "nmse_noninferiority_margin_absolute": nmse_margin,
        "integrity_contract_passed": integrity,
        "primary_accuracy_gate_passed": bool(primary_accuracy_gate),
        "unseen_condition_evaluation_completed": unseen_condition_evaluation_completed,
        "overall_paper_ready": bool(overall_paper_ready),
        "claim_boundary": (
            "Primary confirmatory accuracy/integrity gate passed on the tested V2 simulator. Do not claim broad generalization until the separate unseen-SNR/velocity evaluation is completed; no measured-radio validation is included."
            if primary_accuracy_gate
            else "Do not claim superiority: one or more preregistered pooled/per-seed/noninferiority/integrity criteria failed."
        ),
        "noise_estimator_mean_ratio_across_seeds": float(
            np.mean([summaries[s]["noise_estimation_test"]["estimated_to_true_ratio_mean"] for s in args.seeds])
        ),
        "estimated_channel_nmse_mean_across_seeds": float(
            np.mean([summaries[s]["channel_estimation_nmse"]["estimated_front_end_mean"] for s in args.seeds])
        ),
        "mean_selected_router_latency_seconds_per_frame": float(
            np.mean([summaries[s]["single_pass_inference"]["router_plus_selected_receiver_seconds_per_frame_mean"] for s in args.seeds])
        ),
        "mean_static_generalist_latency_seconds_per_frame": float(
            np.mean([summaries[s]["single_pass_inference"]["static_generalist_seconds_per_frame_mean"] for s in args.seeds])
        ),
        "per_seed_summary": per_seed_rows,
        "source_artifacts_recomputed": True,
    }
    (args.output_dir / "pooled_summary.json").write_text(json.dumps(pooled_summary, indent=2))
    print(json.dumps(pooled_summary, indent=2))


if __name__ == "__main__":
    main()

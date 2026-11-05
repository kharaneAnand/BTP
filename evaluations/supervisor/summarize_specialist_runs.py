"""Aggregate independent specialist-router runs with paired, condition-stratified bootstrap CIs."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

SNRS = (10, 15, 20)
VELOCITIES = (30, 120, 500)
FRAMES_PER_CONDITION = 15


def load_run(seed: int, folder: Path):
    summary = json.loads((folder / "summary.json").read_text())
    scores = pd.read_csv(folder / "test_receiver_metrics.csv")
    decisions = pd.read_csv(folder / "decision_log.csv").sort_values("sample_index")
    if len(decisions) != 135 or scores.sample_index.nunique() != 135:
        raise ValueError(f"Seed {seed}: expected 135 test frames.")
    metadata = pd.DataFrame({"sample_index": np.arange(135)})
    metadata["snr_db"] = [SNRS[i // 45] for i in range(135)]
    metadata["velocity_kmh"] = [VELOCITIES[(i // 15) % 3] for i in range(135)]
    metadata["seed"] = seed
    chosen = decisions.selected_receiver.to_numpy()
    values = {}
    for metric in ("ber", "ser", "nmse"):
        table = scores.pivot(index="sample_index", columns="receiver", values=metric).sort_index()
        static = table["oamp_dl_generalist"].to_numpy(float)
        routed = np.array([table.iloc[i][chosen[i]] for i in range(135)], dtype=float)
        oracle = table.min(axis=1).to_numpy(float)
        values[metric] = {"static": static, "routed": routed, "oracle": oracle}
    counts = decisions.decision.value_counts().to_dict()
    destinations = decisions.loc[decisions.decision == "SWITCH", "selected_receiver"].value_counts().to_dict()
    return summary, metadata, values, counts, destinations


def bootstrap_ci(delta, metadata, draws=20000, seed=2026):
    rng = np.random.default_rng(seed)
    groups = [g.index.to_numpy() for _, g in metadata.groupby(
        ["seed", "snr_db", "velocity_kmh"], sort=True)]
    samples = np.empty(draws, dtype=float)
    for b in range(draws):
        indices = np.concatenate([rng.choice(group, size=len(group), replace=True) for group in groups])
        samples[b] = delta[indices].mean()
    return [float(np.quantile(samples, .025)), float(np.quantile(samples, .975))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="SEED=RESULT_DIRECTORY; repeat per seed")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=20000)
    args = parser.parse_args()
    runs = []
    for item in args.run:
        seed, folder = item.split("=", 1)
        runs.append(load_run(int(seed), Path(folder)))
    if len(runs) < 2:
        raise ValueError("At least two independent seeds are required.")
    metadata = pd.concat([r[1] for r in runs], ignore_index=True)
    result = {
        "seeds": [int(r[0]["seed"]) for r in runs],
        "test_frames": len(metadata),
        "decision_counts": {},
        "switch_destinations": {},
        "metrics": {},
        "bootstrap": {
            "method": "paired percentile bootstrap stratified by seed, SNR, and velocity",
            "draws": args.draws,
            "seed": 2026,
        },
    }
    for run_summary, _, _, counts, destinations in runs:
        seed = str(int(run_summary["seed"]))
        result["decision_counts"][seed] = counts
        result["switch_destinations"][seed] = destinations
    for metric in ("ber", "ser", "nmse"):
        routed = np.concatenate([r[2][metric]["routed"] for r in runs])
        static = np.concatenate([r[2][metric]["static"] for r in runs])
        oracle = np.concatenate([r[2][metric]["oracle"] for r in runs])
        delta = routed - static
        result["metrics"][metric] = {
            "always_oamp_dl": float(static.mean()),
            "specialist_router": float(routed.mean()),
            "oracle_upper_bound_not_deployable": float(oracle.mean()),
            "routed_minus_static_mean": float(delta.mean()),
            "routed_minus_static_ci95": bootstrap_ci(delta, metadata, args.draws),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

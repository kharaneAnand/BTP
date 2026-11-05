"""Re-score the saved V1 test stream with the corrected validation-only controller."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from models.supervisor.reliability_controller import (
    ControllerThresholds,
    ReliabilityController,
)

ROOT = Path(__file__).resolve().parents[2]


def bootstrap_ci(
    differences: np.ndarray,
    conditions: list[tuple[int, int]],
    *,
    replicates: int = 5000,
    seed: int = 2026,
) -> tuple[float, float]:
    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    for value, condition in zip(differences, conditions):
        grouped[condition].append(float(value))
    if any(len(values) < 2 for values in grouped.values()):
        raise ValueError("Every SNR/velocity cell must contain at least two frames.")
    rng = np.random.default_rng(seed)
    totals = np.zeros(replicates, dtype=np.float64)
    for values in grouped.values():
        array = np.asarray(values, dtype=np.float64)
        indices = rng.integers(0, len(array), size=(replicates, len(array)))
        totals += array[indices].sum(axis=1)
    totals /= len(differences)
    low, high = np.quantile(totals, [0.025, 0.975])
    return float(low), float(high)


def load_receiver_metrics() -> dict[int, dict[str, dict[str, float]]]:
    oamp = pd.read_csv(ROOT / "experiments/oamp_dl/per_sample_results.csv")
    gnn = pd.read_csv(ROOT / "experiments/gnn_mmse_init/per_sample_results.csv")
    pi = pd.read_csv(ROOT / "gpu_results/pi_egnn/per_sample_results.csv")
    output: dict[int, dict[str, dict[str, float]]] = {}
    for _, row in oamp.iterrows():
        index = int(row["sample_index"])
        output[index] = {
            "mmse": {
                metric: float(row[f"mmse_{metric}"])
                for metric in ("ber", "ser", "nmse")
            },
            "oamp_dl": {
                metric: float(row[f"oamp_dl_{metric}"])
                for metric in ("ber", "ser", "nmse")
            },
        }
    for _, row in gnn.iterrows():
        index = int(row["sample_index"])
        output[index]["original_gnn"] = {
            metric: float(row[metric]) for metric in ("ber", "ser", "nmse")
        }
    for _, row in pi.iterrows():
        index = int(row["sample_index"])
        output[index]["pi_egnn"] = {
            metric: float(row[f"pi_egnn_{metric}"])
            for metric in ("ber", "ser", "nmse")
        }
    return output


def build_controller() -> ReliabilityController:
    with (ROOT / "experiments/reliability_detector/references.json").open() as file:
        references = json.load(file)["receivers"]
    with (ROOT / "configs/reliability_controller_v1.json").open() as file:
        config = json.load(file)
    threshold_values = config["thresholds"]
    thresholds = ControllerThresholds(
        low_z_threshold=float(threshold_values["low_confidence_z"]),
        switch_z_gap=float(threshold_values["switch_confidence_z_gap"]),
        consecutive_low_frames=int(threshold_values["consecutive_low_frames"]),
        minimum_frames_between_switches=int(
            threshold_values["minimum_frames_between_switches"]
        ),
    )
    gate = config["absolute_performance_gate"]
    return ReliabilityController(
        references=references,
        thresholds=thresholds,
        initial_model=str(config["active_model_initial"]),
        validated_ber=gate["source_results"],
        performance_margin=float(gate["noninferiority_margin_absolute_ber"]),
        best_reference_receiver=str(gate["best_reference_receiver"]),
    )


def main() -> None:
    controller = build_controller()
    metrics = load_receiver_metrics()
    log_path = ROOT / "experiments/reliability_controller/decision_log.csv"
    decision_rows = []
    selected_metrics = []
    baseline_metrics = []
    conditions = []

    with log_path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            index = int(row["sample_index"])
            if controller.active_model != row["active_model_before"]:
                raise ValueError(
                    "Saved per-frame margins no longer match the replayed controller state; "
                    "the original raw V1 frames are required for a fresh inference run."
                )
            candidate_margins = json.loads(row["candidate_confidence_margins"] or "{}")
            decision = controller.step(
                active_margin=float(row["active_confidence_margin"]),
                environment_distance=float(row["environment_distance"]),
                candidate_provider=lambda values=candidate_margins: values,
            )
            used_receiver = decision.active_before
            selected_metrics.append(metrics[index][used_receiver])
            baseline_metrics.append(metrics[index]["oamp_dl"])
            conditions.append((int(row["snr_db"]), int(row["velocity_kmh"])))
            decision_rows.append({
                "sample_index": index,
                "receiver_used_for_frame": used_receiver,
                "active_model_after": decision.active_after,
                "decision": decision.decision,
                "recommended_adaptation_model": decision.recommended_adaptation_model,
            })

    result = {
        "split": "original V1 test split",
        "frames": len(decision_rows),
        "controller_gate_source": "validation split only",
        "decision_counts": {
            action: int(sum(row["decision"] == action for row in decision_rows))
            for action in ("KEEP", "SWITCH", "ADAPT")
        },
        "switch_destinations": dict(Counter(
            row["active_model_after"]
            for row in decision_rows
            if row["decision"] == "SWITCH"
        )),
        "adaptation_recommendations": dict(Counter(
            row["recommended_adaptation_model"]
            for row in decision_rows
            if row["decision"] == "ADAPT"
        )),
        "receiver_used_for_current_frame": "active_model_before",
        "adaptive_pipeline": {},
        "always_oamp_dl": {},
        "paired_adaptive_minus_oamp_dl": {},
    }

    for offset, metric in enumerate(("ber", "ser", "nmse")):
        adaptive = np.asarray([row[metric] for row in selected_metrics])
        baseline = np.asarray([row[metric] for row in baseline_metrics])
        differences = adaptive - baseline
        low, high = bootstrap_ci(
            differences,
            conditions,
            seed=2026 + offset,
        )
        result["adaptive_pipeline"][metric] = float(adaptive.mean())
        result["always_oamp_dl"][metric] = float(baseline.mean())
        result["paired_adaptive_minus_oamp_dl"][metric] = {
            "mean": float(differences.mean()),
            "ci95_low": low,
            "ci95_high": high,
        }

    output = ROOT / "experiments/reliability_controller/validation_gate_replay.json"
    with output.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
    pd.DataFrame(decision_rows).to_csv(
        ROOT / "experiments/reliability_controller/validation_gate_replay.csv",
        index=False,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

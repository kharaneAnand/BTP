"""Causal, decision-directed online adaptation of OAMP-DL without transmitted labels.

The adaptation routine receives only packed received observations (y, H_hat, noise).
Targets are held outside the adaptation API and are used only after inference for
offline BER/SER/NMSE scoring. Updates use past windows only; the current evaluation
window is never used to adapt itself.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.oamp.oamp_dl import OAMPDLDetector
from run_pipeline import prepare_generated_dataset
from src.config.loader import load_config
from training.oamp.dataset import OAMPDataset


TRAINABLE = {"step_logits", "damping_logits", "variance_logits"}
BATCH_SIZE = 8
TRIGGER_NIS = 1.5
ADAPT_STEPS = 5
ADAPT_LR = 2e-4
ANCHOR_WEIGHT = 0.02
CONSTELLATION_WEIGHT = 0.05
MAX_RELATIVE_DRIFT = 0.20


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def qpsk_metrics(pred: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    pred_bits = np.stack((pred.real >= 0, pred.imag >= 0), axis=-1)
    target_bits = np.stack((target.real >= 0, target.imag >= 0), axis=-1)
    errors = pred_bits != target_bits
    return (
        float(errors.mean()),
        float(np.any(errors, axis=-1).mean()),
        float(np.sum(np.abs(pred-target)**2) / max(np.sum(np.abs(target)**2), 1e-12)),
    )


def normalized_innovation(model: OAMPDLDetector, packed: torch.Tensor) -> torch.Tensor:
    y, h, noise = model._unpack(packed)
    estimate = model(packed)
    residual = y - torch.bmm(h, estimate.unsqueeze(-1)).squeeze(-1)
    return (residual.abs().square().mean(dim=-1) / noise.clamp_min(1e-8)).detach()


def self_supervised_objective(
    model: OAMPDLDetector,
    packed: torch.Tensor,
    anchor: dict[str, torch.Tensor],
) -> torch.Tensor:
    y, h, noise = model._unpack(packed)
    estimate = model(packed)
    residual = y - torch.bmm(h, estimate.unsqueeze(-1)).squeeze(-1)
    normalized_residual = residual.abs().square().mean(dim=-1) / noise.clamp_min(1e-8)
    data_consistency = torch.log1p(normalized_residual).mean()
    qpsk_level = 1.0 / np.sqrt(2.0)
    constellation = (
        (estimate.real.abs() - qpsk_level).square().mean()
        + (estimate.imag.abs() - qpsk_level).square().mean()
    )
    drift = torch.cat([
        (parameter - anchor[name]).reshape(-1)
        for name, parameter in model.named_parameters() if name in TRAINABLE
    ]).square().mean()
    return data_consistency + CONSTELLATION_WEIGHT * constellation + ANCHOR_WEIGHT * drift


def adapt_from_past_observations(
    model: OAMPDLDetector,
    past_packed: torch.Tensor,
    device: torch.device,
) -> dict:
    """Update scalar controls using received samples only; no tx symbols accepted."""
    model.eval()
    before_state = copy.deepcopy(model.state_dict())
    anchor = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters() if name in TRAINABLE
    }
    params = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in TRAINABLE)
        if parameter.requires_grad:
            params.append(parameter)
    optimizer = torch.optim.AdamW(params, lr=ADAPT_LR, weight_decay=1e-4)
    before = float(self_supervised_objective(model, past_packed, anchor).detach().cpu())
    history = []
    for step in range(ADAPT_STEPS):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = self_supervised_objective(model, past_packed, anchor)
        if not torch.isfinite(loss):
            model.load_state_dict(before_state)
            return {"accepted": False, "reason": "non_finite_loss", "loss_before": before}
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 0.25)
        optimizer.step()
        history.append(float(loss.detach().cpu()))
    model.eval()
    after = float(self_supervised_objective(model, past_packed, anchor).detach().cpu())
    drift = float(torch.sqrt(torch.cat([
        (parameter.detach() - anchor[name]).reshape(-1)
        for name, parameter in model.named_parameters() if name in TRAINABLE
    ]).square().sum()).cpu())
    anchor_norm = float(torch.sqrt(torch.cat([v.reshape(-1) for v in anchor.values()]).square().sum()).cpu())
    relative_drift = drift / max(anchor_norm, 1e-8)
    accepted = np.isfinite(after) and after < before and relative_drift <= MAX_RELATIVE_DRIFT
    if not accepted:
        model.load_state_dict(before_state)
    return {
        "accepted": bool(accepted),
        "reason": "accepted" if accepted else "objective_or_drift_guard",
        "loss_before": before,
        "loss_after": after,
        "relative_parameter_drift": relative_drift,
        "steps": ADAPT_STEPS,
    }


@torch.no_grad()
def infer(model: OAMPDLDetector, packed: torch.Tensor) -> np.ndarray:
    model.eval()
    return model(packed).detach().cpu().numpy()


def paired_bootstrap(delta: np.ndarray, metadata: pd.DataFrame, seed: int, draws: int = 10000):
    rng = np.random.default_rng(seed)
    groups = [g.index.to_numpy() for _, g in metadata.groupby(
        ["snr_db", "velocity_kmh"], sort=True
    )]
    samples = np.empty(draws)
    for b in range(draws):
        ids = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        samples[b] = delta[ids].mean()
    return [float(np.quantile(samples, .025)), float(np.quantile(samples, .975))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.seed < 49:
        raise ValueError("Use fresh independent seeds >= 49.")
    seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)

    config = load_config(ROOT / "configs" / "experiment_v1.yaml")
    raw_dir, _, full_meta, run_config = prepare_generated_dataset(
        config, output / "generated", args.seed
    )
    test_meta = full_meta.loc[full_meta.split == "test"].sample(
        frac=1.0, random_state=args.seed
    ).reset_index(drop=True)
    dataset = OAMPDataset(test_meta, raw_dir, run_config)
    obs = int(run_config.representation.expected_channel_shape.rows)
    symbols = int(run_config.representation.expected_channel_shape.cols)
    iterations = int(run_config.oamp_dl.iterations)
    model = OAMPDLDetector(obs, symbols, iterations).to(device)
    checkpoint = torch.load(
        ROOT / "experiments" / "oamp_dl" / "best_model.pt",
        map_location=device, weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    static = copy.deepcopy(model).to(device).eval()

    # Keep targets segregated from the adaptation function and its inputs.
    packed_stream = [dataset[i][0] for i in range(len(dataset))]
    chunks = [packed_stream[i:i+BATCH_SIZE] for i in range(0, len(packed_stream), BATCH_SIZE)]
    predictions_static, predictions_adapted, targets, rows = [], [], [], []
    previous_chunk = None
    accepted_count = rejected_count = adapt_count = keep_count = 0

    for chunk_index, chunk in enumerate(chunks):
        if previous_chunk is not None:
            past = torch.stack(previous_chunk).to(device)
            nis = float(normalized_innovation(model, past).mean().cpu())
            if nis > TRIGGER_NIS:
                adapt_count += 1
                result = adapt_from_past_observations(model, past, device)
                accepted_count += int(result["accepted"])
                rejected_count += int(not result["accepted"])
                action = "ADAPT"
            else:
                keep_count += 1
                result = {"accepted": False, "reason": "innovation_below_threshold",
                          "loss_before": None, "loss_after": None}
                action = "KEEP"
        else:
            nis = None
            result = {"accepted": False, "reason": "warmup_no_past_window"}
            action = "WARMUP"

        current = torch.stack(chunk).to(device)
        static_pred = infer(static, current)
        adapted_pred = infer(model, current)
        start = chunk_index * BATCH_SIZE
        end = start + len(chunk)
        # No target is read or retained during stream adaptation or inference.
        for j, (sp, ap) in enumerate(zip(static_pred, adapted_pred)):
            predictions_static.append(sp)
            predictions_adapted.append(ap)
            row = test_meta.iloc[start+j]
            rows.append({
                "sample_index": start+j,
                "snr_db": int(row.snr_db),
                "velocity_kmh": int(row.velocity_kmh),
                "action_before_window": action,
                "innovation_on_past_window": nis,
                "adaptation_accepted": bool(result.get("accepted", False)),
                "adaptation_reason": result.get("reason"),
                "adaptation_loss_before": result.get("loss_before"),
                "adaptation_loss_after": result.get("loss_after"),
            })
        previous_chunk = chunk

    # Only after every online update and prediction is complete do we load labels
    # for the offline evaluation stage.
    target_array = np.asarray([dataset[i][1].numpy() for i in range(len(dataset))])
    static_array = np.asarray(predictions_static)
    adapted_array = np.asarray(predictions_adapted)
    score_rows = []
    for i in range(len(target_array)):
        sm = qpsk_metrics(static_array[i], target_array[i])
        am = qpsk_metrics(adapted_array[i], target_array[i])
        score_rows.append({
            **rows[i],
            "static_ber": sm[0], "adapted_ber": am[0],
            "static_ser": sm[1], "adapted_ser": am[1],
            "static_nmse": sm[2], "adapted_nmse": am[2],
        })
    results = pd.DataFrame(score_rows)
    results.to_csv(output / "stream_metrics.csv", index=False)
    decision_counts = {
        "WARMUP": int(sum(r["action_before_window"] == "WARMUP" for r in rows)),
        "KEEP": int(sum(r["action_before_window"] == "KEEP" for r in rows)),
        "ADAPT": int(sum(r["action_before_window"] == "ADAPT" for r in rows)),
    }
    summary = {
        "seed": args.seed,
        "test_frames": len(results),
        "adaptation_protocol": "causal past-window decision-directed self-supervision",
        "adaptation_input_contract": "packed received y, estimated H, and noise only; no transmitted symbols",
        "target_symbols_used_for_adaptation": False,
        "current_or_future_frame_used_to_update_its_own_detector": False,
        "adaptation_trigger_normalized_innovation_threshold": TRIGGER_NIS,
        "batch_size": BATCH_SIZE,
        "adaptation_steps": ADAPT_STEPS,
        "trainable_parameters": sorted(TRAINABLE),
        "decision_counts_by_frame": decision_counts,
        "adaptation_attempts": adapt_count,
        "accepted_updates": accepted_count,
        "rejected_updates": rejected_count,
        "metrics": {},
        "paired_bootstrap": "10,000 paired resamples stratified by SNR and velocity",
        "limitations": [
            "Decision-directed self-supervision can suffer error propagation.",
            "The synthetic receiver input still uses simulator-provided noise variance.",
            "This is frame-batched online adaptation in simulation, not hardware validation.",
        ],
    }
    for metric in ("ber", "ser", "nmse"):
        static_values = results[f"static_{metric}"].to_numpy()
        adapted_values = results[f"adapted_{metric}"].to_numpy()
        delta = adapted_values - static_values
        summary["metrics"][metric] = {
            "static_oamp_dl": float(static_values.mean()),
            "online_adapted_oamp_dl": float(adapted_values.mean()),
            "adapted_minus_static": float(delta.mean()),
            "ci95": paired_bootstrap(delta, test_meta, args.seed + 9000),
        }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

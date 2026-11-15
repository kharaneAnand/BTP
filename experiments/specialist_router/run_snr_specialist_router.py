"""Compare velocity-only and expanded velocity+SNR OAMP-DL specialist banks.

All specialists are fine-tuned on the training split only. Both routers are fitted
on validation-only BER labels and inference-available features. Test labels are
used only for final reporting; no SNR/velocity metadata or transmitted symbols
are supplied to the routing decision.
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
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.oamp.oamp_dl import OAMPDLDetector
from run_pipeline import prepare_generated_dataset
from src.config.loader import load_config
from training.oamp.dataset import OAMPDataset
from experiments.specialist_router.run_specialist_router import (
    fit_ridge_router,
    metrics,
    paired_bootstrap,
    train_specialist,
)


def inference_features(model, packed: torch.Tensor, estimate: torch.Tensor) -> np.ndarray:
    y, h, noise = model._unpack(packed)
    residual = y - torch.bmm(h, estimate.unsqueeze(-1)).squeeze(-1)
    margins = torch.cat((estimate.real.abs(), estimate.imag.abs()), dim=-1)
    row_energy = h.abs().square().mean(dim=-1)
    col_energy = h.abs().square().mean(dim=-2)
    row_p = row_energy / row_energy.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    col_p = col_energy / col_energy.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    entropy_r = -(row_p * row_p.clamp_min(1e-12).log()).sum(dim=-1)
    entropy_c = -(col_p * col_p.clamp_min(1e-12).log()).sum(dim=-1)
    values = torch.stack((
        h.abs().square().mean(dim=(-2, -1)).log(),
        row_energy.std(dim=-1) / row_energy.mean(dim=-1).clamp_min(1e-12),
        col_energy.std(dim=-1) / col_energy.mean(dim=-1).clamp_min(1e-12),
        entropy_r, entropy_c,
        torch.linalg.vector_norm(y, dim=-1).log(),
        torch.linalg.vector_norm(residual, dim=-1)
            / torch.linalg.vector_norm(y, dim=-1).clamp_min(1e-12),
        margins.mean(dim=-1),
        margins.std(dim=-1),
        (margins < 0.1).float().mean(dim=-1),
        estimate.abs().mean(dim=-1),
        noise.clamp_min(1e-12).log(),
    ), dim=-1)
    return values.detach().cpu().numpy().astype(np.float64)


def train_snr_specialist(base_state, obs_count, sym_count, iterations, train_meta,
                         raw_dir, config, snr_db, output_path, device):
    subset = train_meta.loc[train_meta.snr_db.astype(int) == snr_db].copy()
    if subset.empty:
        raise RuntimeError(f"No training samples for SNR regime {snr_db} dB.")
    dataset = OAMPDataset(subset, raw_dir, config)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    model = OAMPDLDetector(obs_count, sym_count, iterations).to(device)
    model.load_state_dict(copy.deepcopy(base_state))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    history = []
    for epoch in range(8):
        model.train()
        losses = []
        for packed, target in loader:
            packed, target = packed.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(packed)
            loss = (pred.real - target.real).square().mean() + (pred.imag - target.imag).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite SNR-specialist training loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch + 1, "train_mse": float(np.mean(losses))})
    model.eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "snr_training_regime_db": snr_db,
        "epochs": 8,
        "training_samples": len(dataset),
    }, output_path)
    return model, history, len(dataset)


@torch.no_grad()
def score_split(models, metadata, raw_dir, config, device):
    dataset = OAMPDataset(metadata, raw_dir, config)
    rows = []
    for idx in range(len(dataset)):
        packed, target = dataset[idx]
        packed_b = packed.unsqueeze(0).to(device)
        target_np = target.numpy()
        for name, model in models.items():
            model.eval()
            estimate = model(packed_b)
            feat = inference_features(model, packed_b, estimate)[0]
            m = metrics(estimate[0].cpu().numpy(), target_np)
            rows.append({
                "sample_index": idx, "receiver": name,
                "ber": m[0], "ser": m[1], "nmse": m[2],
                **{f"x{i}": float(v) for i, v in enumerate(feat)},
            })
    return pd.DataFrame(rows)


def evaluate_router(chosen, test_scores, test_meta, receiver_names, seed):
    tables = {
        metric: test_scores.pivot(index="sample_index", columns="receiver", values=metric).sort_index()
        for metric in ("ber", "ser", "nmse")
    }
    routed, static, oracle, paired = {}, {}, {}, {}
    for metric, table in tables.items():
        static[metric] = table["oamp_dl_generalist"].to_numpy()
        routed[metric] = np.asarray([table.iloc[i][chosen[i]] for i in range(len(chosen))])
        oracle[metric] = table.min(axis=1).to_numpy()
        delta = routed[metric] - static[metric]
        paired[metric] = {
            "mean": float(delta.mean()),
            "ci95": paired_bootstrap(delta, test_meta.reset_index(drop=True), seed=2026 + seed),
        }
    decisions = pd.Series(chosen).value_counts().to_dict()
    return {
        "routed": {k: float(v.mean()) for k, v in routed.items()},
        "always_oamp_dl": {k: float(v.mean()) for k, v in static.items()},
        "oracle_upper_bound_not_deployable": {k: float(v.mean()) for k, v in oracle.items()},
        "routed_minus_static_paired": paired,
        "decision_counts": {k: int(v) for k, v in decisions.items()},
        "test_receiver_ber": {
            k: float(v) for k, v in tables["ber"].mean(axis=0).to_dict().items()
        },
        "oracle_frame_fraction": float(np.mean(oracle["ber"] < static["ber"])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=53)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.seed < 53:
        raise ValueError("Use a fresh independent seed (53 or higher), not a reported seed.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    config = load_config(ROOT / "configs" / "experiment_v1.yaml")
    raw_dir, _, full_meta, run_config = prepare_generated_dataset(
        config, out / "generated", args.seed
    )
    train_meta = full_meta.loc[full_meta.split == "train"].reset_index(drop=True)
    val_meta = full_meta.loc[full_meta.split == "validation"].reset_index(drop=True)
    test_meta = full_meta.loc[full_meta.split == "test"].reset_index(drop=True)
    obs = int(run_config.representation.expected_channel_shape.rows)
    sym = int(run_config.representation.expected_channel_shape.cols)
    iterations = int(run_config.oamp_dl.iterations)

    base = OAMPDLDetector(obs, sym, iterations).to(device)
    checkpoint = torch.load(
        ROOT / "experiments" / "oamp_dl" / "best_model.pt",
        map_location=device, weights_only=False,
    )
    base.load_state_dict(checkpoint["model_state_dict"])
    base.eval()
    models = {"oamp_dl_generalist": base}
    training = {"velocity": {}, "snr": {}}

    # Preserve the existing three velocity-specialists unchanged.
    for velocity in (30, 120, 500):
        model, history, count = train_specialist(
            base.state_dict(), obs, sym, iterations, train_meta, raw_dir,
            run_config, velocity,
            out / "checkpoints" / f"oamp_dl_specialist_v{velocity}.pt", device,
        )
        models[f"specialist_v{velocity}"] = model
        training["velocity"][str(velocity)] = {
            "training_samples": count, "history": history,
        }

    # Add three orthogonal SNR-specialists; no existing receiver is removed.
    for snr in (10, 15, 20):
        model, history, count = train_snr_specialist(
            base.state_dict(), obs, sym, iterations, train_meta, raw_dir,
            run_config, snr,
            out / "checkpoints" / f"oamp_dl_specialist_snr{snr}.pt", device,
        )
        models[f"specialist_snr{snr}"] = model
        training["snr"][str(snr)] = {
            "training_samples": count, "history": history,
        }

    val_scores = score_split(models, val_meta, raw_dir, run_config, device)
    test_scores = score_split(models, test_meta, raw_dir, run_config, device)
    val_scores.to_csv(out / "validation_receiver_metrics.csv", index=False)
    test_scores.to_csv(out / "test_receiver_metrics.csv", index=False)

    velocity_names = [
        "oamp_dl_generalist", "specialist_v30", "specialist_v120", "specialist_v500"
    ]
    velocity_val = val_scores.loc[val_scores.receiver.isin(velocity_names)]
    velocity_test = test_scores.loc[test_scores.receiver.isin(velocity_names)]
    velocity_choice, _, _ = fit_ridge_router(velocity_val, velocity_test)
    combined_choice, _, calibration = fit_ridge_router(val_scores, test_scores)

    velocity_result = evaluate_router(
        velocity_choice, velocity_test, test_meta, velocity_names, args.seed
    )
    combined_result = evaluate_router(
        combined_choice, test_scores, test_meta, list(models), args.seed
    )
    decision_log = pd.DataFrame({
        "sample_index": np.arange(len(combined_choice)),
        "velocity_only_selection": velocity_choice,
        "combined_bank_selection": combined_choice,
    })
    decision_log.to_csv(out / "decision_log.csv", index=False)

    summary = {
        "seed": args.seed,
        "frames": len(test_meta),
        "split_counts": full_meta.split.value_counts().to_dict(),
        "specialist_training": training,
        "router_features": [
            "pilot_channel_matrix_statistics", "received_observation_energy",
            "candidate_output_confidence", "candidate_residual_consistency",
            "log_noise_variance_already_supplied_to_receiver",
        ],
        "router_uses_snr_or_velocity_labels": False,
        "test_labels_used_for_router_fit_or_actions": False,
        "transmitted_symbols_used_for_routing": False,
        "router": "validation-calibrated per-receiver ridge BER prediction; lowest predicted BER",
        "velocity_only_bank": velocity_result,
        "combined_velocity_plus_snr_bank": combined_result,
        "caveats": [
            "SNR/velocity labels define training subsets only; neither is a routing input.",
            "The receiver and router use simulator-provided noise variance; deployment needs a noise estimator.",
            "The oracle is an unattainable test-label upper bound and is never used for routing.",
            "This is a pilot comparison on independent synthetic seeds, not hardware validation.",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

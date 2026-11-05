"""Train velocity-regime OTFS OAMP-DL specialists and a validation-calibrated router.

The routing model never receives velocity labels or test BER. Velocity is used only
to define specialist training subsets. A ridge calibration model is fit on the
validation split using observable channel/receiver-output features and validation
BER labels; the held-out test split is used once for final evaluation.
"""
from __future__ import annotations

import argparse
import copy
import json
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
from training.oamp.dataset import OAMPDataset, load_split_metadata


def metrics(pred: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    pb = np.stack((pred.real >= 0, pred.imag >= 0), axis=-1)
    tb = np.stack((target.real >= 0, target.imag >= 0), axis=-1)
    return (float(np.mean(pb != tb)),
            float(np.mean(np.any(pb != tb, axis=-1))),
            float(np.sum(np.abs(pred-target)**2) / max(np.sum(np.abs(target)**2), 1e-12)))


def features(model, packed: torch.Tensor, estimate: torch.Tensor) -> np.ndarray:
    y, h, noise = model._unpack(packed)
    residual = y - torch.bmm(h, estimate.unsqueeze(-1)).squeeze(-1)
    abs_est = estimate.abs()
    margins = torch.cat((estimate.real.abs(), estimate.imag.abs()), dim=-1)
    row_energy = h.abs().square().mean(dim=-1)
    col_energy = h.abs().square().mean(dim=-2)
    row_p = row_energy / row_energy.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    col_p = col_energy / col_energy.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    entropy_r = -(row_p * row_p.clamp_min(1e-12).log()).sum(dim=-1)
    entropy_c = -(col_p * col_p.clamp_min(1e-12).log()).sum(dim=-1)
    values = torch.stack((
        noise.log().squeeze(-1) if noise.ndim > 1 else noise.log(),
        h.abs().square().mean(dim=(-2,-1)).log(),
        row_energy.std(dim=-1) / row_energy.mean(dim=-1).clamp_min(1e-12),
        col_energy.std(dim=-1) / col_energy.mean(dim=-1).clamp_min(1e-12),
        entropy_r, entropy_c,
        torch.linalg.vector_norm(y, dim=-1).log(),
        torch.linalg.vector_norm(residual, dim=-1) /
            torch.linalg.vector_norm(y, dim=-1).clamp_min(1e-12),
        margins.mean(dim=-1),
        margins.std(dim=-1),
        (margins < 0.1).float().mean(dim=-1),
        abs_est.mean(dim=-1),
    ), dim=-1)
    return values.detach().cpu().numpy().astype(np.float64)


def train_specialist(base_state, obs_count, sym_count, iterations, train_meta,
                     raw_dir, config, velocity, output_path, device):
    subset = train_meta.loc[train_meta.velocity_kmh.astype(int) == velocity].copy()
    if subset.empty:
        raise RuntimeError(f"No training samples for velocity regime {velocity}.")
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
            loss = (pred.real-target.real).square().mean() + (pred.imag-target.imag).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite specialist training loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch+1, "train_mse": float(np.mean(losses))})
    model.eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "velocity_training_regime_kmh": velocity,
                "epochs": 8, "training_samples": len(dataset)}, output_path)
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
            feat = features(model, packed_b, estimate)[0]
            m = metrics(estimate[0].cpu().numpy(), target_np)
            rows.append({"sample_index": idx, "receiver": name,
                         "ber": m[0], "ser": m[1], "nmse": m[2],
                         **{f"x{i}": float(v) for i, v in enumerate(feat)}})
    return pd.DataFrame(rows)


def fit_ridge_router(validation: pd.DataFrame, test: pd.DataFrame, alpha=10.0):
    feature_cols = [c for c in validation.columns if c.startswith("x")]
    names = list(validation.receiver.unique())
    test_predictions = {}
    coefficients = {}
    for name in names:
        v = validation.loc[validation.receiver == name].sort_values("sample_index")
        t = test.loc[test.receiver == name].sort_values("sample_index")
        x = v[feature_cols].to_numpy(float)
        xt = t[feature_cols].to_numpy(float)
        mean, scale = x.mean(axis=0), x.std(axis=0)
        scale[scale < 1e-8] = 1.0
        x = (x-mean)/scale
        xt = (xt-mean)/scale
        x = np.column_stack((np.ones(len(x)), x))
        xt = np.column_stack((np.ones(len(xt)), xt))
        penalty = np.eye(x.shape[1])*alpha
        penalty[0,0] = 0.0
        beta = np.linalg.solve(x.T@x + penalty, x.T@v.ber.to_numpy(float))
        test_predictions[name] = xt@beta
        coefficients[name] = {"validation_ber_mean": float(v.ber.mean())}
    pred = pd.DataFrame(test_predictions)
    chosen = pred.idxmin(axis=1).to_numpy()
    return chosen, pred, coefficients


def paired_bootstrap(delta, metadata, seed=2026, draws=5000):
    rng = np.random.default_rng(seed)
    groups = [g.index.to_numpy() for _, g in metadata.groupby(["snr_db","velocity_kmh"])]
    means = np.empty(draws)
    for b in range(draws):
        ids = np.concatenate([rng.choice(ix, size=len(ix), replace=True) for ix in groups])
        means[b] = float(np.mean(delta[ids]))
    return [float(np.quantile(means, .025)), float(np.quantile(means, .975))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu","cuda"), default="cpu")
    args = parser.parse_args()
    if args.seed in (42,43,44):
        raise ValueError("Use a new independent seed, not a previously reported seed.")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    config = load_config(ROOT/"configs"/"experiment_v1.yaml")
    raw_dir, dataset_root, full_meta, run_config = prepare_generated_dataset(config, out/"generated", args.seed)
    train_meta = full_meta.loc[full_meta.split == "train"].reset_index(drop=True)
    val_meta = full_meta.loc[full_meta.split == "validation"].reset_index(drop=True)
    test_meta = full_meta.loc[full_meta.split == "test"].reset_index(drop=True)
    obs = int(run_config.representation.expected_channel_shape.rows)
    sym = int(run_config.representation.expected_channel_shape.cols)
    iterations = int(run_config.oamp_dl.iterations)
    base = OAMPDLDetector(obs, sym, iterations).to(device)
    checkpoint = torch.load(ROOT/"experiments"/"oamp_dl"/"best_model.pt", map_location=device, weights_only=False)
    base.load_state_dict(checkpoint["model_state_dict"])
    models = {"oamp_dl_generalist": base}
    specialist_history = {}
    for velocity in (30,120,500):
        model, history, count = train_specialist(
            base.state_dict(), obs, sym, iterations, train_meta, raw_dir,
            run_config, velocity, out/"checkpoints"/f"oamp_dl_specialist_v{velocity}.pt", device)
        models[f"specialist_v{velocity}"] = model
        specialist_history[str(velocity)] = {"training_samples": count, "history": history}
    val_scores = score_split(models, val_meta, raw_dir, run_config, device)
    test_scores = score_split(models, test_meta, raw_dir, run_config, device)
    val_scores.to_csv(out/"validation_receiver_metrics.csv", index=False)
    test_scores.to_csv(out/"test_receiver_metrics.csv", index=False)
    chosen, predicted, calibration = fit_ridge_router(val_scores, test_scores)
    actual = test_scores.pivot(index="sample_index", columns="receiver", values="ber").sort_index()
    static = actual["oamp_dl_generalist"].to_numpy()
    routed = np.array([actual.iloc[i][chosen[i]] for i in range(len(chosen))])
    oracle = actual.min(axis=1).to_numpy()
    meta = test_meta.reset_index(drop=True)
    delta = routed-static
    ci = paired_bootstrap(delta, meta)
    decision_counts = {"KEEP": int(np.sum(chosen=="oamp_dl_generalist")),
                       "SWITCH": int(np.sum(chosen!="oamp_dl_generalist")),
                       "ADAPT": 0}
    decisions = pd.DataFrame({"sample_index": np.arange(len(chosen)),
                              "selected_receiver": chosen,
                              "decision": np.where(chosen=="oamp_dl_generalist","KEEP","SWITCH"),
                              "static_oamp_ber": static, "routed_ber": routed,
                              "oracle_ber": oracle})
    decisions.to_csv(out/"decision_log.csv", index=False)
    summary = {
        "seed": args.seed, "frames": len(test_meta),
        "split_counts": full_meta.split.value_counts().to_dict(),
        "specialist_training": specialist_history,
        "routing_features": ["pilot_channel_matrix_statistics", "received-observation energy",
                             "candidate-output confidence", "candidate residual consistency"],
        "velocity_labels_used_at_inference": False,
        "test_labels_used_for_router_fit_or_actions": False,
        "router": "validation-calibrated per-receiver ridge BER prediction; select lowest predicted BER",
        "decision_counts": decision_counts,
        "routed": {"ber": float(routed.mean())},
        "always_oamp_dl": {"ber": float(static.mean())},
        "oracle_upper_bound_not_deployable": {"ber": float(oracle.mean())},
        "routed_minus_static_ber": {"mean": float(delta.mean()), "ci95": ci},
        "frames_where_non_generalist_wins_oracle": int(np.sum(oracle < static)),
        "oracle_frame_fraction": float(np.mean(oracle < static)),
        "test_receiver_ber": {k: float(v.mean()) for k,v in actual.items()},
        "caveat": "Oracle uses test labels and is an unattainable upper bound. Router is evaluated only on held-out test labels after fitting on validation."
    }
    (out/"summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

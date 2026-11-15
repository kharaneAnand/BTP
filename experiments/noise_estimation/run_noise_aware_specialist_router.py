"""Train and evaluate a noise-aware OTFS OAMP-DL specialist bank on V2 data.

All deployable receiver inputs (noise variance and H_hat) are reconstructed from
rx_dd and fixed public waveform configuration. The V2 generator uses fixed pilot
power and calibrates the noise estimator on the training split only. Validation
labels fit the shared-feature router; test labels are opened only after every
test-frame decision and receiver inference has completed.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.dataset_generation.generate_otfs_dataset_v2 import generate_dataset_v2
from models.oamp.oamp_dl import OAMPDLDetector
from models.receivers.linear_mmse_torch import LinearMMSEDetector
from run_pipeline import hard_decisions
from src.config.loader import load_config
from src.receivers.mmse import build_data_observation
from training.dnn.trainer import DNNTrainer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class V2ReceiverDataset(Dataset):
    """Training dataset with explicit estimated-noise or oracle-reference mode."""

    def __init__(self, metadata: pd.DataFrame, raw_dir: Path, config, mode: str):
        if mode not in {"estimated", "oracle"}:
            raise ValueError("mode must be estimated or oracle")
        self.metadata = metadata.reset_index(drop=True)
        self.raw_dir = raw_dir
        self.config = config
        self.mode = mode

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int):
        row = self.metadata.iloc[index]
        with np.load(self.raw_dir / str(row["file"]), allow_pickle=False) as sample:
            rx_dd = np.asarray(sample["rx_dd"], dtype=np.complex64)
            tx_dd = np.asarray(sample["tx_dd"], dtype=np.complex64)
            if self.mode == "estimated":
                h_hat = np.asarray(sample["h_hat"], dtype=np.complex64)
                noise_power = float(sample["noise_estimate"])
            else:
                h_hat = np.asarray(sample["h_hat_oracle"], dtype=np.complex64)
                noise_power = float(sample["noise_power"])
        y_data = build_data_observation(rx_dd, self.config)
        packed = np.concatenate((
            np.asarray(y_data).reshape(-1),
            h_hat.reshape(-1),
            np.asarray([noise_power], dtype=np.complex64),
        )).astype(np.complex64)
        return torch.from_numpy(packed), torch.from_numpy(tx_dd)


def pack_inputs(metadata: pd.DataFrame, raw_dir: Path, config, mode: str) -> list[torch.Tensor]:
    """Load only live receiver inputs. No tx_dd, h_dd, SNR or true noise is read."""
    packed_samples = []
    for _, row in metadata.reset_index(drop=True).iterrows():
        with np.load(raw_dir / str(row["file"]), allow_pickle=False) as sample:
            rx_dd = np.asarray(sample["rx_dd"], dtype=np.complex64)
            if mode == "estimated":
                h_hat = np.asarray(sample["h_hat"], dtype=np.complex64)
                noise_power = float(sample["noise_estimate"])
            elif mode == "oracle":
                h_hat = np.asarray(sample["h_hat_oracle"], dtype=np.complex64)
                noise_power = float(sample["noise_power"])
            else:
                raise ValueError("mode must be estimated or oracle")
        y_data = build_data_observation(rx_dd, config)
        packed = np.concatenate((
            np.asarray(y_data).reshape(-1),
            h_hat.reshape(-1),
            np.asarray([noise_power], dtype=np.complex64),
        )).astype(np.complex64)
        packed_samples.append(torch.from_numpy(packed))
    return packed_samples


def load_offline_labels(metadata: pd.DataFrame, raw_dir: Path) -> dict[str, np.ndarray]:
    """Open transmitted symbols/true channel/noise only after inference is complete."""
    targets, channels, noise, noise_est, h_est, h_oracle = [], [], [], [], [], []
    for _, row in metadata.reset_index(drop=True).iterrows():
        with np.load(raw_dir / str(row["file"]), allow_pickle=False) as sample:
            targets.append(np.asarray(sample["tx_dd"], dtype=np.complex64))
            channels.append(np.asarray(sample["h_dd"], dtype=np.complex64))
            noise.append(float(sample["noise_power"]))
            noise_est.append(float(sample["noise_estimate"]))
            h_est.append(np.asarray(sample["h_hat"], dtype=np.complex64))
            h_oracle.append(np.asarray(sample["h_hat_oracle"], dtype=np.complex64))
    return {
        "tx_dd": np.asarray(targets),
        "h_dd": np.asarray(channels),
        "noise_power": np.asarray(noise, dtype=np.float64),
        "noise_estimate": np.asarray(noise_est, dtype=np.float64),
        "h_hat": np.asarray(h_est),
        "h_hat_oracle": np.asarray(h_oracle),
    }


def qpsk_metrics(pred: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    pred_bits = np.stack((pred.real >= 0, pred.imag >= 0), axis=-1)
    target_bits = np.stack((target.real >= 0, target.imag >= 0), axis=-1)
    errors = pred_bits != target_bits
    ber = float(errors.mean())
    ser = float(np.any(errors, axis=-1).mean())
    nmse = float(np.sum(np.abs(pred - target) ** 2) / max(np.sum(np.abs(target) ** 2), 1e-12))
    return ber, ser, nmse


def channel_nmse(estimate: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sum(np.abs(estimate - truth) ** 2) / max(np.sum(np.abs(truth) ** 2), 1e-12))


def shared_features(model: OAMPDLDetector, packed: torch.Tensor) -> torch.Tensor:
    """Candidate-independent router features from rx, estimated H and estimated noise."""
    y, h, noise = model._unpack(packed)
    h_h = h.conj().transpose(-2, -1)
    column_power = (h.abs() ** 2).sum(dim=-2).clamp_min(1e-8)
    matched = torch.bmm(h_h, y.unsqueeze(-1)).squeeze(-1) / column_power
    residual = y - torch.bmm(h, matched.unsqueeze(-1)).squeeze(-1)

    row_energy = h.abs().square().mean(dim=-1)
    col_energy = h.abs().square().mean(dim=-2)
    row_cv = row_energy.std(dim=-1) / row_energy.mean(dim=-1).clamp_min(1e-12)
    col_cv = col_energy.std(dim=-1) / col_energy.mean(dim=-1).clamp_min(1e-12)
    row_prob = row_energy / row_energy.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    col_prob = col_energy / col_energy.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    row_entropy = -(row_prob * row_prob.clamp_min(1e-12).log()).sum(dim=-1)
    col_entropy = -(col_prob * col_prob.clamp_min(1e-12).log()).sum(dim=-1)
    margin = torch.cat((matched.real.abs(), matched.imag.abs()), dim=-1)
    y_norm = torch.linalg.vector_norm(y, dim=-1).clamp_min(1e-12)
    residual_ratio = torch.linalg.vector_norm(residual, dim=-1) / y_norm
    values = torch.stack((
        h.abs().square().mean(dim=(-2, -1)).clamp_min(1e-12).log(),
        row_cv,
        col_cv,
        row_entropy,
        col_entropy,
        y_norm.log(),
        noise.clamp_min(1e-12).log(),
        residual_ratio,
        margin.mean(dim=-1),
        margin.std(dim=-1),
        (margin < 0.1).float().mean(dim=-1),
        matched.abs().mean(dim=-1),
    ), dim=-1)
    return values


@torch.no_grad()
def infer_all_models(
    models: dict[str, OAMPDLDetector],
    feature_model: OAMPDLDetector,
    packed_samples: list[torch.Tensor],
    device: torch.device,
    batch_size: int = 1,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Run frame-by-frame inference to match the deployed single-frame path.

    Batched GEMM kernels can produce small floating-point differences from a
    batch-size-one forward pass. Since the final receiver is invoked one frame
    at a time, offline candidate predictions must use the same batch size;
    otherwise a strict equivalence audit can fail on harmless kernel drift.
    """
    del batch_size  # Kept for call-site compatibility; inference is deliberately B=1.
    predictions = {name: [] for name in models}
    feature_rows = []
    for index, packed_cpu in enumerate(packed_samples):
        packed = packed_cpu.unsqueeze(0).to(device)
        row = shared_features(feature_model, packed).cpu().numpy()[0]
        feature_rows.append({
            "sample_index": index,
            **{f"x{i}": float(v) for i, v in enumerate(row)},
        })
        for name, model in models.items():
            model.eval()
            estimate = model(packed).cpu().numpy()[0]
            predictions[name].append(estimate)
    return {k: np.asarray(v) for k, v in predictions.items()}, pd.DataFrame(feature_rows)


def metrics_frame(predictions: dict[str, np.ndarray], labels: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for receiver, estimates in predictions.items():
        for i, estimate in enumerate(estimates):
            ber, ser, nmse = qpsk_metrics(estimate, labels["tx_dd"][i])
            rows.append({
                "sample_index": i,
                "receiver": receiver,
                "ber": ber,
                "ser": ser,
                "nmse": nmse,
            })
    return pd.DataFrame(rows)


def fit_shared_router(
    validation_features: pd.DataFrame,
    validation_metrics: pd.DataFrame,
    test_features: pd.DataFrame,
    candidates: list[str],
    alpha: float = 10.0,
) -> tuple[np.ndarray, pd.DataFrame, dict]:
    feature_cols = [c for c in validation_features.columns if c.startswith("x")]
    val_features = validation_features.sort_values("sample_index")
    test_features = test_features.sort_values("sample_index")
    predictions = {}
    parameters = {}
    for receiver in candidates:
        receiver_metrics = validation_metrics.loc[
            validation_metrics.receiver == receiver
        ][["sample_index", "ber"]]
        merged = receiver_metrics.merge(
            val_features, on="sample_index", how="inner", validate="one_to_one"
        ).sort_values("sample_index")
        x = merged[feature_cols].to_numpy(dtype=np.float64)
        y = merged["ber"].to_numpy(dtype=np.float64)
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale[scale < 1e-8] = 1.0
        xz = (x - mean) / scale
        test_x = test_features[feature_cols].to_numpy(dtype=np.float64)
        x_design = np.column_stack((np.ones(len(xz)), xz))
        penalty = np.eye(x_design.shape[1]) * alpha
        penalty[0, 0] = 0.0
        beta = np.linalg.solve(x_design.T @ x_design + penalty, x_design.T @ y)
        # Match predict_one's per-frame arithmetic exactly to avoid BLAS batch
        # rounding changing a near-tied receiver selection.
        predictions[receiver] = np.asarray([
            np.dot(np.concatenate(([1.0], (row - mean) / scale)), beta)
            for row in test_x
        ], dtype=np.float64)
        parameters[receiver] = {
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "beta": beta.tolist(),
            "validation_ber_mean": float(y.mean()),
        }
    prediction_frame = pd.DataFrame(predictions)
    chosen = prediction_frame.idxmin(axis=1).to_numpy()
    return chosen, prediction_frame, parameters


def predict_one(features: np.ndarray, parameters: dict) -> str:
    scores = {}
    for receiver, params in parameters.items():
        mean = np.asarray(params["mean"], dtype=np.float64)
        scale = np.asarray(params["scale"], dtype=np.float64)
        beta = np.asarray(params["beta"], dtype=np.float64)
        xz = (features - mean) / scale
        scores[receiver] = float(np.dot(np.concatenate(([1.0], xz)), beta))
    return min(scores, key=scores.get)


def paired_bootstrap(delta: np.ndarray, metadata: pd.DataFrame, seed: int, draws: int = 10000):
    rng = np.random.default_rng(seed)
    groups = [g.index.to_numpy() for _, g in metadata.groupby(
        ["snr_db", "velocity_kmh"], sort=True
    )]
    samples = np.empty(draws)
    for b in range(draws):
        ids = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        samples[b] = float(delta[ids].mean())
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def policy_result(
    chosen: np.ndarray,
    receiver_metrics: pd.DataFrame,
    metadata: pd.DataFrame,
    seed: int,
    baseline: str = "noise_est_generalist",
) -> tuple[dict, pd.DataFrame]:
    pivot = {
        metric: receiver_metrics.pivot(
            index="sample_index", columns="receiver", values=metric
        ).sort_index()
        for metric in ("ber", "ser", "nmse")
    }
    rows = []
    result = {}
    for metric, table in pivot.items():
        static = table[baseline].to_numpy()
        selected = np.asarray([table.iloc[i][chosen[i]] for i in range(len(chosen))])
        delta = selected - static
        result[metric] = {
            "static_generalist": float(static.mean()),
            "routed": float(selected.mean()),
            "routed_minus_static": float(delta.mean()),
            "paired_ci95": paired_bootstrap(delta, metadata, seed + 9000 + len(metric)),
        }
    for i, receiver in enumerate(chosen):
        row = {"sample_index": i, "selected_receiver": receiver}
        for metric, table in pivot.items():
            row[f"static_{metric}"] = float(table.iloc[i][baseline])
            row[f"routed_{metric}"] = float(table.iloc[i][receiver])
        rows.append(row)
    return result, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.seed < 57:
        raise ValueError("Use fresh independent V2 seeds 57 or higher.")
    set_seed(args.seed)
    torch.set_num_threads(2)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = load_config(ROOT / "configs" / "experiment_v2.yaml")
    raw_dir, full_meta, calibration = generate_dataset_v2(
        config, output / "dataset", args.seed
    )
    train_meta = full_meta.loc[full_meta.split == "train"].reset_index(drop=True)
    val_meta = full_meta.loc[full_meta.split == "validation"].reset_index(drop=True)
    test_meta = full_meta.loc[full_meta.split == "test"].reset_index(drop=True)

    obs = int(config.representation.expected_channel_shape.rows)
    symbols = int(config.representation.expected_channel_shape.cols)
    iterations = int(config.oamp_dl.iterations)
    base_checkpoint = torch.load(
        ROOT / "experiments" / "oamp_dl" / "best_model.pt",
        map_location=device, weights_only=False,
    )

    def train_generalist(mode: str, name: str) -> OAMPDLDetector:
        set_seed(args.seed)
        model = OAMPDLDetector(obs, symbols, iterations).to(device)
        model.load_state_dict(base_checkpoint["model_state_dict"])
        train_loader = DataLoader(
            V2ReceiverDataset(train_meta, raw_dir, config, mode),
            batch_size=8, shuffle=True, num_workers=0,
        )
        validation_loader = DataLoader(
            V2ReceiverDataset(val_meta, raw_dir, config, mode),
            batch_size=8, shuffle=False, num_workers=0,
        )
        trainer = DNNTrainer(
            model=model,
            train_loader=train_loader,
            validation_loader=validation_loader,
            epochs=15,
            learning_rate=5e-5,
            weight_decay=1e-5,
            gradient_clip_norm=1.0,
            early_stopping_patience=5,
            early_stopping_min_delta=1e-7,
            scheduler_factor=0.5,
            scheduler_patience=2,
            checkpoint_path=output / "checkpoints" / f"{name}.pt",
            history_path=output / "training" / f"{name}.csv",
            device=device,
        )
        trainer.fit()
        return model.eval()

    # Same initialization and training schedule; oracle mode is a non-deployable reference.
    oracle_generalist = train_generalist("oracle", "oracle_noise_generalist")
    estimated_generalist = train_generalist("estimated", "noise_est_generalist")

    # A structurally different, non-learned detector is included as a candidate.
    # It uses the same deployable (y, H_hat, noise estimate) contract as OAMP-DL.
    linear_mmse = LinearMMSEDetector(obs, symbols).to(device).eval()
    estimated_models = {
        "noise_est_generalist": estimated_generalist,
        "linear_mmse": linear_mmse,
    }
    specialist_history = {"velocity": {}, "snr": {}}

    def train_specialist(subset: pd.DataFrame, regime_name: str, regime_value: int):
        set_seed(args.seed + int(regime_value))
        dataset = V2ReceiverDataset(subset, raw_dir, config, "estimated")
        loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
        model = OAMPDLDetector(obs, symbols, iterations).to(device)
        model.load_state_dict(copy.deepcopy(estimated_generalist.state_dict()))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        history = []
        for epoch in range(8):
            model.train()
            losses = []
            for packed, target in loader:
                packed, target = packed.to(device), target.to(device)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(packed)
                loss = (prediction.real - target.real).square().mean() + (
                    prediction.imag - target.imag
                ).square().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite specialist training loss.")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            history.append({"epoch": epoch + 1, "train_mse": float(np.mean(losses))})
        model.eval()
        name = f"specialist_{regime_name}{regime_value}"
        torch.save(
            {"model_state_dict": model.state_dict(), "regime": regime_name,
             "regime_value": regime_value, "training_samples": len(dataset), "epochs": 8},
            output / "checkpoints" / f"{name}.pt",
        )
        return name, model, {"training_samples": len(dataset), "history": history}

    for velocity in (30, 120, 500):
        subset = train_meta.loc[train_meta.velocity_kmh.astype(int) == velocity]
        name, model, history = train_specialist(subset, "v", velocity)
        estimated_models[name] = model
        specialist_history["velocity"][str(velocity)] = history

    for snr in (10, 15, 20):
        subset = train_meta.loc[train_meta.snr_db.astype(int) == snr]
        name, model, history = train_specialist(subset, "snr", snr)
        estimated_models[name] = model
        specialist_history["snr"][str(snr)] = history

    # Inference inputs are loaded without tx_dd, h_dd, SNR labels or true noise.
    val_inputs = pack_inputs(val_meta, raw_dir, config, "estimated")
    test_inputs = pack_inputs(test_meta, raw_dir, config, "estimated")
    val_predictions, val_features = infer_all_models(
        estimated_models, estimated_generalist, val_inputs, device
    )
    test_predictions, test_features = infer_all_models(
        estimated_models, estimated_generalist, test_inputs, device
    )

    # Validation targets are opened only after validation inference, to fit the router.
    val_labels = load_offline_labels(val_meta, raw_dir)
    val_metrics = metrics_frame(val_predictions, val_labels)
    val_metrics.to_csv(output / "validation_receiver_metrics.csv", index=False)
    test_features.to_csv(output / "test_router_features.csv", index=False)
    val_features.to_csv(output / "validation_router_features.csv", index=False)

    velocity_candidates = [
        "noise_est_generalist", "specialist_v30", "specialist_v120", "specialist_v500"
    ]
    expanded_candidates = list(estimated_models)
    velocity_choice, velocity_pred, velocity_params = fit_shared_router(
        val_features, val_metrics, test_features, velocity_candidates
    )
    expanded_choice, expanded_pred, expanded_params = fit_shared_router(
        val_features, val_metrics, test_features, expanded_candidates
    )

    # Execute the frozen test policy using received inputs only. No test labels are open.
    selected_elapsed = []
    static_elapsed = []
    selected_receiver_counts = {}
    single_pass_prediction_diffs = []
    with torch.no_grad():
        for i, packed_cpu in enumerate(test_inputs):
            packed = packed_cpu.unsqueeze(0).to(device)
            t0 = time.perf_counter()
            feature = shared_features(estimated_generalist, packed).cpu().numpy()[0]
            receiver = predict_one(feature, expanded_params)
            if receiver != expanded_choice[i]:
                raise AssertionError("Single-pass router decision differs from batch router decision.")
            selected_receiver_counts[receiver] = selected_receiver_counts.get(receiver, 0) + 1
            selected_output = estimated_models[receiver](packed).cpu().numpy()[0]
            selected_elapsed.append(time.perf_counter() - t0)
            reference_output = test_predictions[receiver][i]
            absolute_difference = np.abs(selected_output - reference_output)
            reference_scale = max(float(np.max(np.abs(reference_output))), 1e-8)
            single_pass_prediction_diffs.append({
                "sample_index": i,
                "receiver": receiver,
                "max_absolute_difference": float(np.max(absolute_difference)),
                "mean_absolute_difference": float(np.mean(absolute_difference)),
                "max_relative_difference": float(np.max(absolute_difference) / reference_scale),
            })
            # A small floating-point discrepancy is logged rather than silently
            # treated as a failed scientific experiment. Large disagreement still
            # stops the run and reports its magnitude for diagnosis.
            if float(np.max(absolute_difference) / reference_scale) > 1e-3:
                raise AssertionError(
                    "Single-pass selected inference materially differs from offline "
                    f"candidate prediction (frame={i}, receiver={receiver}, "
                    f"max_abs={np.max(absolute_difference):.6e}, "
                    f"max_rel={np.max(absolute_difference) / reference_scale:.6e})."
                )

            t1 = time.perf_counter()
            _ = estimated_generalist(packed)
            static_elapsed.append(time.perf_counter() - t1)

    # Non-deployable oracle-noise generalist inference is also completed before labels are opened.
    oracle_inputs = pack_inputs(test_meta, raw_dir, config, "oracle")
    oracle_predictions, _ = infer_all_models(
        {"oracle_noise_generalist": oracle_generalist},
        oracle_generalist, oracle_inputs, device
    )

    # Only now open test targets, true channel and true noise for offline scoring.
    test_labels = load_offline_labels(test_meta, raw_dir)
    test_metrics = metrics_frame(test_predictions, test_labels)
    oracle_metrics = metrics_frame(oracle_predictions, test_labels)
    test_metrics.to_csv(output / "test_receiver_metrics.csv", index=False)
    oracle_metrics.to_csv(output / "oracle_noise_reference_metrics.csv", index=False)

    velocity_result, velocity_policy = policy_result(
        velocity_choice, test_metrics, test_meta, args.seed
    )
    expanded_result, expanded_policy = policy_result(
        expanded_choice, test_metrics, test_meta, args.seed + 1
    )

    # Offline channel-estimation and noise-estimation audits, after decisions are frozen.
    channel_estimated_nmse = np.asarray([
        np.sum(np.abs(test_labels["h_hat"][i] - test_labels["h_dd"][i]) ** 2)
        / max(np.sum(np.abs(test_labels["h_dd"][i]) ** 2), 1e-12)
        for i in range(len(test_meta))
    ])
    channel_oracle_nmse = np.asarray([
        np.sum(np.abs(test_labels["h_hat_oracle"][i] - test_labels["h_dd"][i]) ** 2)
        / max(np.sum(np.abs(test_labels["h_dd"][i]) ** 2), 1e-12)
        for i in range(len(test_meta))
    ])
    noise_ratio = test_labels["noise_estimate"] / test_labels["noise_power"]
    channel_nmse_per_frame = np.asarray([
        np.sum(np.abs(test_labels["h_hat"][i] - test_labels["h_dd"][i]) ** 2)
        / max(np.sum(np.abs(test_labels["h_dd"][i]) ** 2), 1e-12)
        for i in range(len(test_meta))
    ])
    noise_channel_audit = test_meta[["snr_db", "velocity_kmh", "file"]].reset_index(drop=True).copy()
    noise_channel_audit["sample_index"] = np.arange(len(test_meta))
    noise_channel_audit["noise_estimate"] = test_labels["noise_estimate"]
    noise_channel_audit["noise_true"] = test_labels["noise_power"]
    noise_channel_audit["noise_estimate_to_true_ratio"] = noise_ratio
    noise_channel_audit["channel_estimation_nmse"] = channel_nmse_per_frame
    noise_channel_audit.to_csv(output / "noise_channel_audit.csv", index=False)
    pd.DataFrame(single_pass_prediction_diffs).to_csv(
        output / "single_pass_prediction_consistency.csv", index=False
    )

    # Save policy metrics and a complete selection trace.
    policy = pd.DataFrame({
        "sample_index": np.arange(len(test_meta)),
        "velocity_only_selection": velocity_choice,
        "expanded_selection": expanded_choice,
    })
    policy = policy.merge(velocity_policy, on="sample_index", suffixes=("", "_velocity"))
    policy = policy.merge(expanded_policy, on="sample_index", suffixes=("_velocity", "_expanded"))
    policy = policy.merge(test_meta[["snr_db", "velocity_kmh"]].reset_index(drop=True).assign(
        sample_index=np.arange(len(test_meta))
    ), on="sample_index", validate="one_to_one")
    policy.to_csv(output / "selected_policy_metrics.csv", index=False)

    decisions = pd.DataFrame({
        "sample_index": np.arange(len(test_meta)),
        "velocity_only_selection": velocity_choice,
        "expanded_selection": expanded_choice,
        **{f"predicted_ber_velocity_{k}": v for k, v in velocity_pred.to_dict(orient="list").items()},
        **{f"predicted_ber_expanded_{k}": v for k, v in expanded_pred.to_dict(orient="list").items()},
    })
    decisions.to_csv(output / "router_decisions.csv", index=False)
    test_meta[["snr_db", "velocity_kmh", "file"]].assign(
        sample_index=np.arange(len(test_meta))
    ).to_csv(output / "test_frame_metadata.csv", index=False)

    # Summarize static and routed policies using identical held-out frames.
    oracle_static = {
        metric: float(oracle_metrics[metric].mean())
        for metric in ("ber", "ser", "nmse")
    }
    summary = {
        "seed": args.seed,
        "dataset_version": "V2_fixed_pilot_power",
        "test_frames": len(test_meta),
        "split_counts": full_meta.split.value_counts().to_dict(),
        "pilot_power_linear_fixed": float(config.pilot.power_linear),
        "noise_estimator_calibration": calibration,
        "inference_input_contract": [
            "rx_dd",
            "fixed public pilot/OTFS configuration",
            "training-calibrated noise estimate from pilot CE delay columns",
            "channel matrix re-estimated from rx_dd using estimated noise",
        ],
        "simulator_noise_power_used_for_deployable_receiver_input": False,
        "simulator_h_hat_used_for_deployable_receiver_input": False,
        "test_labels_used_for_router_fit_or_actions": False,
        "test_h_dd_or_tx_dd_used_before_inference_completed": False,
        "router": "single-pass per-candidate ridge BER prediction from shared candidate-independent features",
        "router_uses_snr_or_velocity_labels": False,
        "router_candidates": expanded_candidates,
        "specialist_training": specialist_history,
        "oracle_noise_generalist_non_deployable_reference": oracle_static,
        "noise_estimated_generalist": {
            metric: float(test_metrics.loc[test_metrics.receiver == "noise_est_generalist", metric].mean())
            for metric in ("ber", "ser", "nmse")
        },
        "velocity_only_router": velocity_result,
        "expanded_velocity_plus_snr_router": expanded_result,
        "channel_estimation_nmse": {
            "estimated_front_end_mean": float(channel_estimated_nmse.mean()),
            "oracle_front_end_mean": float(channel_oracle_nmse.mean()),
            "estimated_minus_oracle_mean": float((channel_estimated_nmse - channel_oracle_nmse).mean()),
        },
        "noise_estimation_test": {
            "estimated_to_true_ratio_mean": float(noise_ratio.mean()),
            "estimated_to_true_ratio_median": float(np.median(noise_ratio)),
            "estimated_to_true_ratio_std": float(noise_ratio.std(ddof=1)),
            "within_20_percent_fraction": float(np.mean(np.abs(noise_ratio - 1.0) <= 0.20)),
            "within_30_percent_fraction": float(np.mean(np.abs(noise_ratio - 1.0) <= 0.30)),
        },
        "single_pass_inference": {
            "model_calls_per_frame": 1,
            "receiver_selection_counts": selected_receiver_counts,
            "router_plus_selected_receiver_seconds_per_frame_mean": float(np.mean(selected_elapsed)),
            "static_generalist_seconds_per_frame_mean": float(np.mean(static_elapsed)),
            "latency_ratio_to_static_generalist": float(np.mean(selected_elapsed) / max(np.mean(static_elapsed), 1e-12)),
            "max_single_pass_prediction_absolute_difference": float(
                max(row["max_absolute_difference"] for row in single_pass_prediction_diffs)
            ),
            "mean_single_pass_prediction_absolute_difference": float(
                np.mean([row["mean_absolute_difference"] for row in single_pass_prediction_diffs])
            ),
            "max_single_pass_prediction_relative_difference": float(
                max(row["max_relative_difference"] for row in single_pass_prediction_diffs)
            ),
        },
        "limitations": [
            "The noise estimator uses a pilot-guard geometry-specific minimum-energy CE column and a training-only multiplicative calibration.",
            "Only 16 Doppler samples contribute to each frame's selected-column noise estimate; residual per-frame variance is expected.",
            "The V2 pilot power is fixed and known; this differs from V1's simulator-SNR-dependent pilot power.",
            "The oracle-noise generalist is non-deployable and is reported only as an upper-bound reference.",
            "Synthetic channel family only; unseen-condition and hardware validation remain separate requirements.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

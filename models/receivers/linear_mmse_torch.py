"""Torch implementation of the classical linear MMSE OTFS detector.

The detector is deliberately parameter-free: it solves
(H^H H + sigma^2 I) x = H^H y using the receiver-side channel and
noise estimates. It shares the packed input contract with OAMP-DL.
"""
from __future__ import annotations

import torch
from torch import nn


class LinearMMSEDetector(nn.Module):
    """Batched complex-valued linear MMSE detector."""

    def __init__(self, observation_count: int, symbol_count: int) -> None:
        super().__init__()
        self.observation_count = int(observation_count)
        self.symbol_count = int(symbol_count)
        if self.observation_count <= 0 or self.symbol_count <= 0:
            raise ValueError("System dimensions must be positive.")

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        if packed.ndim != 2:
            raise ValueError("packed input must have shape (B, F).")
        matrix_size = self.observation_count * self.symbol_count
        expected = self.observation_count + matrix_size + 1
        if packed.shape[-1] != expected:
            raise ValueError(
                f"Unexpected packed feature count: {packed.shape[-1]} != {expected}."
            )

        y = packed[:, :self.observation_count]
        h = packed[
            :, self.observation_count:self.observation_count + matrix_size
        ].reshape(-1, self.observation_count, self.symbol_count)
        noise = packed[:, -1].real.abs().clamp_min(1e-8)

        h_h = h.conj().transpose(-2, -1)
        gram = torch.bmm(h_h, h)
        eye = torch.eye(
            self.symbol_count, dtype=packed.dtype, device=packed.device
        ).expand(packed.shape[0], -1, -1)
        regularized = gram + noise[:, None, None] * eye
        rhs = torch.bmm(h_h, y.unsqueeze(-1))
        return torch.linalg.solve(regularized, rhs).squeeze(-1)

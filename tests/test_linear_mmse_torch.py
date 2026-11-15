import numpy as np
import pytest
import torch

from models.receivers.linear_mmse_torch import LinearMMSEDetector


def pack(y, h, noise):
    return torch.cat((y, h.reshape(1, -1), torch.tensor([[noise]], dtype=torch.complex64)), dim=1)


def test_linear_mmse_matches_numpy_reference():
    rng = np.random.default_rng(19)
    h_np = (
        rng.normal(size=(5, 3)) + 1j * rng.normal(size=(5, 3))
    ).astype(np.complex64)
    y_np = (
        rng.normal(size=5) + 1j * rng.normal(size=5)
    ).astype(np.complex64)
    noise = 0.17
    expected = np.linalg.solve(
        h_np.conj().T @ h_np + noise * np.eye(3, dtype=np.complex64),
        h_np.conj().T @ y_np,
    )
    packed = pack(
        torch.from_numpy(y_np).reshape(1, -1),
        torch.from_numpy(h_np),
        noise,
    )
    actual = LinearMMSEDetector(5, 3)(packed).detach().numpy()[0]
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_linear_mmse_rejects_wrong_feature_count():
    detector = LinearMMSEDetector(5, 3)
    with pytest.raises(ValueError):
        detector(torch.zeros(1, 4, dtype=torch.complex64))

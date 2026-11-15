"""Unit tests for inference-available SNR-specialist router features."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from experiments.specialist_router.run_snr_specialist_router import inference_features
from models.oamp.oamp_dl import OAMPDLDetector


class SNRRouterFeatureTests(unittest.TestCase):
    def test_feature_shape_finiteness_and_noise_feature(self) -> None:
        model = OAMPDLDetector(observation_count=4, symbol_count=3, iterations=2)
        y = torch.randn(2, 4, dtype=torch.complex64)
        h = torch.randn(2, 4, 3, dtype=torch.complex64)
        noise = torch.tensor([0.1, 0.01], dtype=torch.float32)
        packed = torch.cat((y, h.reshape(2, -1), noise.to(torch.complex64).unsqueeze(-1)), dim=-1)
        estimate = torch.randn(2, 3, dtype=torch.complex64)

        features = inference_features(model, packed, estimate)

        self.assertEqual(features.shape, (2, 12))
        self.assertTrue(np.isfinite(features).all())
        np.testing.assert_allclose(features[:, -1], np.log(noise.numpy()), rtol=1e-5)

    def test_lower_noise_power_is_encoded_as_lower_log_variance(self) -> None:
        model = OAMPDLDetector(observation_count=4, symbol_count=3, iterations=2)
        y = torch.ones(2, 4, dtype=torch.complex64)
        h = torch.ones(2, 4, 3, dtype=torch.complex64)
        noise = torch.tensor([0.1, 0.01], dtype=torch.float32)
        packed = torch.cat((y, h.reshape(2, -1), noise.to(torch.complex64).unsqueeze(-1)), dim=-1)
        estimate = torch.ones(2, 3, dtype=torch.complex64)

        features = inference_features(model, packed, estimate)

        self.assertGreater(features[0, -1], features[1, -1])


if __name__ == "__main__":
    unittest.main()

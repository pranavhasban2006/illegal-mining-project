import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import pytest

from mining_detector.hybrid_models.models import build_model, MODEL_REGISTRY, LITE_KWARGS, SIAMESE_MODELS


@pytest.mark.parametrize("name", list(MODEL_REGISTRY))
def test_lite_variant_has_fewer_params_than_full(name):
    full = build_model(name, lite=False)
    lite = build_model(name, lite=True)
    n_full = sum(p.numel() for p in full.parameters())
    n_lite = sum(p.numel() for p in lite.parameters())
    assert n_lite < n_full


@pytest.mark.parametrize("name", [n for n in MODEL_REGISTRY if n not in SIAMESE_MODELS])
def test_lite_variant_forward_pass_shape_matches_full(name):
    full = build_model(name, lite=False)
    lite = build_model(name, lite=True)
    x = torch.rand(2, 3, 64, 64)
    assert full(x).shape == lite(x).shape == (2,)


def test_lite_siamese_forward_pass_shape():
    lite = build_model("siamese_cnn_transformer", lite=True)
    x0 = torch.rand(2, 3, 64, 64)
    x1 = torch.rand(2, 3, 64, 64)
    assert lite(x0, x1).shape == (2,)


def test_every_registry_entry_has_lite_kwargs():
    assert set(LITE_KWARGS.keys()) == set(MODEL_REGISTRY.keys())

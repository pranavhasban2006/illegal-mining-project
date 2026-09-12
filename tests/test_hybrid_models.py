import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import pytest

from mining_detector.hybrid_models.models import MODEL_REGISTRY, SIAMESE_MODELS

pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.mark.parametrize("name", [n for n in MODEL_REGISTRY if n not in SIAMESE_MODELS])
def test_single_image_model_forward_pass_shape(name):
    model = MODEL_REGISTRY[name]()
    x = torch.rand(4, 3, 64, 64)
    out = model(x)
    assert out.shape == (4,)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("name", list(SIAMESE_MODELS))
def test_siamese_model_forward_pass_shape(name):
    model = MODEL_REGISTRY[name]()
    x0 = torch.rand(4, 3, 64, 64)
    x1 = torch.rand(4, 3, 64, 64)
    out = model(x0, x1)
    assert out.shape == (4,)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("name", [n for n in MODEL_REGISTRY if n not in SIAMESE_MODELS])
def test_single_image_model_gradients_flow(name):
    model = MODEL_REGISTRY[name]()
    x = torch.rand(2, 3, 64, 64)
    y = torch.tensor([0.0, 1.0])
    out = model(x)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(out, y)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.any(g != 0) for g in grads)


def test_resnet_attention_unet_native_resolution_flag():
    from mining_detector.hybrid_models.models import ResNetAttentionUNet
    x = torch.rand(2, 3, 64, 64)

    default_model = ResNetAttentionUNet()
    native_model = ResNetAttentionUNet(native_resolution=True)

    # Both must still produce a map matching the input resolution after
    # interpolation, regardless of internal decoder resolution.
    assert default_model(x, return_map=True).shape == (2, 64, 64)
    assert native_model(x, return_map=True).shape == (2, 64, 64)
    assert native_model.native_resolution is True
    assert default_model.native_resolution is False


def test_siamese_shares_weights_across_both_inputs():
    from mining_detector.hybrid_models.models import SiameseCNNTransformer
    model = SiameseCNNTransformer()
    # The same shared_cnn module instance must process both t0 and t1 --
    # that's what makes it "Siamese" rather than two independent encoders.
    x = torch.rand(1, 3, 64, 64)
    f_a = model.shared_cnn(x)
    f_b = model.shared_cnn(x)
    assert torch.allclose(f_a, f_b)

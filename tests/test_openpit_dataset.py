import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
import torch

from mining_detector.hybrid_models import openpit_dataset as opd

pytestmark = pytest.mark.skipif(
    not os.path.isdir(opd.IMG_DIR),
    reason="Open-Pit Mining Areas Dataset not present locally (gitignored, see README)",
)


def test_tiling_produces_correct_patch_shape():
    img = np.random.rand(3, 512, 512).astype(np.float32)
    mask = np.random.randint(0, 2, (512, 512)).astype(np.float32)
    patches = opd._tile_patches(img, mask, patch_size=64)
    assert len(patches) == 64  # (512/64)^2
    for img_p, mask_p in patches:
        assert img_p.shape == (3, 64, 64)
        assert mask_p.shape == (64, 64)


def test_train_test_split_has_no_overlap_and_holds_out_test_tiles():
    train_patches, test_patches = opd.build_patch_lists()
    assert len(train_patches) > 0
    assert len(test_patches) > 0
    # 8 train tiles + 2 test tiles, 64 patches each (512/64)^2.
    assert len(train_patches) == 8 * 64
    assert len(test_patches) == 2 * 64


def test_cls_dataset_label_derived_from_mining_threshold():
    img = np.zeros((3, 64, 64), dtype=np.float32)
    mostly_mining = np.ones((64, 64), dtype=np.float32)
    mostly_empty = np.zeros((64, 64), dtype=np.float32)
    ds = opd.ClsPatchDataset([(img, mostly_mining), (img, mostly_empty)], mining_threshold=0.05)
    _, label_mining = ds[0]
    _, label_empty = ds[1]
    assert label_mining.item() == 1.0
    assert label_empty.item() == 0.0


def test_seg_dataset_returns_tensors():
    img = np.random.rand(3, 64, 64).astype(np.float32)
    mask = np.random.randint(0, 2, (64, 64)).astype(np.float32)
    ds = opd.SegPatchDataset([(img, mask)])
    x, y = ds[0]
    assert isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)
    assert x.shape == (3, 64, 64)
    assert y.shape == (64, 64)


def test_segmentation_models_return_map_matches_input_resolution():
    from mining_detector.hybrid_models.models import UNetTransformer, ResNetAttentionUNet
    x = torch.rand(2, 3, 64, 64)
    for cls in (UNetTransformer, ResNetAttentionUNet):
        model = cls()
        seg_map = model(x, return_map=True)
        assert seg_map.shape == (2, 64, 64)
        assert torch.isfinite(seg_map).all()

"""Tests for src/mining_detector/hybrid_models.

Uses small synthetic data with known answers. The real Open-Pit dataset is not needed, and
these tests pass without it.
"""

import numpy as np
import pytest
import torch

from mining_detector.hybrid_models import models as M
from mining_detector.hybrid_models import openpit_dataset as D
from mining_detector.hybrid_models import train_openpit as T


# ----------------------------------------------------------- model shapes -------

@pytest.mark.parametrize("name", M.ALL_MODELS)
@pytest.mark.parametrize("lite", [False, True])
def test_forward_shape_and_size_budget(name, lite):
    model = M.build_model(name, in_ch=6, lite=lite)
    x = torch.randn(2, 6, 64, 64)
    out = model(x, x) if name in M.SIAMESE_MODELS else model(x)
    if name in M.SEGMENTATION_MODELS:
        assert tuple(out.shape) == (2, 1, 64, 64)
    else:
        assert tuple(out.shape) == (2, 2)
    assert M.count_parameters(model) < 50_000


def test_lite_is_smaller_than_full():
    for name in M.ALL_MODELS:
        assert M.count_parameters(M.build_model(name, lite=True)) < M.count_parameters(M.build_model(name))


def test_unknown_model_name_is_rejected():
    with pytest.raises(ValueError, match="Unknown model"):
        M.build_model("NoSuchNet")


@pytest.mark.parametrize("name", M.SEGMENTATION_MODELS + M.CLASSIFICATION_MODELS)
def test_one_training_step_lowers_loss(name):
    """On a fixed batch, a few Adam steps must reduce the loss. This checks gradients flow."""
    torch.manual_seed(0)
    model = M.build_model(name, in_ch=3, lite=True)
    x = torch.randn(8, 3, 64, 64)
    if name in M.SEGMENTATION_MODELS:
        y = (torch.rand(8, 1, 64, 64) > 0.7).float()
        loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        y = torch.tensor([0, 1] * 4)
        loss_fn = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    model.train()
    losses = []
    for _ in range(8):
        opt.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


def test_one_siamese_training_step_lowers_loss():
    torch.manual_seed(0)
    model = M.build_model("SiameseCNNTransformer", in_ch=3, lite=True)
    x0 = torch.randn(8, 3, 64, 64)
    x1 = x0 + 0.5 * torch.randn_like(x0)
    y = torch.tensor([0, 1] * 4)
    loss_fn = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(8):
        opt.zero_grad()
        loss = loss_fn(model(x0, x1), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


# ----------------------------------------------------------- tiling --------------

def test_tile_pair_cuts_known_patches():
    # Each pixel's value is its row number, so we can check exactly which rows went where.
    H, W = 128, 192
    img = np.broadcast_to(np.arange(H, dtype=np.float32)[None, :, None], (2, H, W)).copy()
    mask = np.zeros((H, W), np.uint8)
    mask[64:, 128:] = 1
    patches, masks = D.tile_pair(img, mask, size=64)
    assert patches.shape == (6, 2, 64, 64)                      # 2 rows x 3 columns
    assert masks.shape == (6, 64, 64)
    # Patch index 4 is row 1, column 1: rows 64..127, columns 64..127.
    assert patches[4, 0, 0, 0] == 64 and patches[4, 0, -1, 0] == 127
    assert masks[4].sum() == 0                                   # columns 64..127 have no mining
    assert masks[5].sum() == 64 * 64                             # row 1, column 2 is all mining


def test_tile_pair_drops_edge_pixels():
    img = np.zeros((3, 70, 70), np.float32)
    patches, masks = D.tile_pair(img, np.zeros((70, 70), np.uint8), size=64)
    assert len(patches) == 1 and len(masks) == 1


def test_tile_pair_smaller_than_patch_gives_empty():
    patches, masks = D.tile_pair(np.zeros((2, 30, 30), np.float32), np.zeros((30, 30), np.uint8), 64)
    assert patches.shape[0] == 0 and masks.shape[0] == 0


def test_patch_labels_use_positive_share():
    masks = np.zeros((3, 10, 10), np.uint8)
    masks[1, :1, :] = 1      # 10 of 100 pixels = 10 %: above the 1 % threshold
    masks[2, 0, 0] = 1       # 1 of 100 pixels = 1 %: exactly at the threshold
    assert list(D.patch_labels(masks)) == [0, 1, 1]


# ----------------------------------------------------------- splits --------------

def _tagged_tiles(n: int):
    """Tile k is filled with the value k + 1, so each patch reveals which tile it came from."""
    return [(np.full((2, 64, 64), k + 1, np.float32), np.zeros((64, 64), np.uint8)) for k in range(n)]


def test_tile_indices_are_disjoint_and_cover_everything():
    train, val, test = D.tile_indices(10, seed=3)
    assert len(test) == D.N_TEST_TILES and len(val) == D.N_VAL_TILES
    assert sorted(train + val + test) == list(range(10))


def test_no_patch_is_in_two_sets():
    """Test patches must come only from the held-out test tiles (no spatial leakage)."""
    splits = D.build_splits(_tagged_tiles(10), seed=1)
    # Each tile has one distinct value, so the value of a patch identifies its source tile.
    train_ids = set(np.round(np.unique(splits.train_x[:, 0, 0, 0]), 6))
    test_ids = set(np.round(np.unique(splits.test_x[:, 0, 0, 0]), 6))
    val_ids = set(np.round(np.unique(splits.val_x[:, 0, 0, 0]), 6))
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids) and val_ids.isdisjoint(train_ids)
    assert len(test_ids) == D.N_TEST_TILES
    assert len(splits.test_tile_ids) == D.N_TEST_TILES


def test_normalisation_uses_training_statistics_only():
    tiles = _tagged_tiles(10)
    splits = D.build_splits(tiles, seed=1)
    # Training patches are normalised with their own mean and std, so they centre on zero.
    assert abs(splits.train_x.mean()) < 1e-4
    assert abs(splits.train_x.std() - 1.0) < 1e-3


def test_augment_keeps_image_and_mask_aligned():
    rng = np.random.default_rng(0)
    img = np.zeros((1, 8, 8), np.float32)
    img[0, 0, 0] = 1.0            # marker pixel in the top-left corner
    mask = np.zeros((8, 8), np.uint8)
    mask[0, 0] = 1                # same marker in the mask
    for _ in range(10):
        a, m = D.augment(img, mask, rng)
        assert a[0].argmax() == m.argmax()  # marker lands on the same pixel in both


def test_missing_dataset_gives_clear_message(tmp_path):
    with pytest.raises(D.DatasetMissing, match="was not found"):
        D.list_pairs(tmp_path / "absent")


def test_unpaired_mask_is_reported(tmp_path):
    (tmp_path / "scene_a.tif").write_bytes(b"")
    (tmp_path / "scene_b_mask.tif").write_bytes(b"")
    with pytest.raises(D.DatasetMissing, match="no matching image"):
        D.list_pairs(tmp_path)


def test_read_pair_reads_real_geotiff(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    profile = dict(driver="GTiff", height=8, width=8, count=2, dtype="float32",
                   transform=from_origin(0, 0, 1, 1))
    img_path, mask_path = tmp_path / "s1.tif", tmp_path / "s1_mask.tif"
    with rasterio.open(img_path, "w", **profile) as dst:
        dst.write(np.ones((2, 8, 8), np.float32) * 5)
    with rasterio.open(mask_path, "w", driver="GTiff", height=8, width=8, count=1,
                       dtype="uint8", transform=from_origin(0, 0, 1, 1)) as dst:
        dst.write(np.full((1, 8, 8), 2, np.uint8))
    pairs = D.list_pairs(tmp_path)
    img, mask = D.read_pair(*pairs[0])
    assert img.shape == (2, 8, 8) and img[0, 0, 0] == 5
    assert mask.max() == 1 and mask.sum() == 64      # any value above 0 counts as mining


# ----------------------------------------------------------- scores --------------

def test_segmentation_scores_known_answer():
    target = np.array([[1, 1, 0, 0]])
    logits = np.array([[2.0, -1.0, 3.0, -2.0]])  # predicts pixels 0 and 2
    s = T.segmentation_scores(logits, target)
    # tp=1 (pixel 0), fp=1 (pixel 2), fn=1 (pixel 1): IoU = 1/3, Dice = 2/4 = 0.5
    assert s["iou"] == pytest.approx(1 / 3, abs=1e-4)
    assert s["dice"] == pytest.approx(0.5, abs=1e-4)


def test_classification_scores_known_answer():
    pred = np.array([1, 1, 0, 0, 1])
    target = np.array([1, 0, 0, 1, 1])
    s = T.classification_scores(pred, target)
    # correct: 3 of 5. tp=2, fp=1, fn=1: F1 = 4 / 6
    assert s["accuracy"] == pytest.approx(0.6)
    assert s["f1"] == pytest.approx(4 / 6, abs=1e-4)


# ----------------------------------------------------------- training ------------

def test_standin_training_runs_and_reports_honestly():
    """One short run on stand-in data: the code path works and the result is flagged demo-only."""
    torch.manual_seed(0)
    data = D.build_splits(D.standin_tiles(n_scenes=6), seed=0, source="synthetic_standin")
    model, scores = T.train_segmentation("UNetTransformer", True, data, epochs=1)
    assert {"iou", "dice", "best_epoch", "history"} <= set(scores)
    assert 0.0 <= scores["iou"] <= 1.0
    model2, scores2 = T.train_classification("CNNBiLSTM", True, data, epochs=1)
    assert {"accuracy", "f1"} <= set(scores2)


def test_fit_stops_early_when_validation_stalls():
    """Early stopping: once validation loss stops improving for PATIENCE epochs, training ends."""
    model = torch.nn.Linear(2, 1)
    calls = {"n": 0}

    def forward_loss(opt, batch):
        return 0.0

    def train_iter(epoch):
        return [None]

    def val_loss():
        calls["n"] += 1
        return 1.0  # never improves after the first epoch

    info = T._fit(model, forward_loss, train_iter, val_loss, epochs=50)
    assert info["epochs_run"] == 1 + T.PATIENCE
    assert info["best_epoch"] == 1


def test_siamese_scene_split_has_no_shared_scene():
    (tr0, _, tr_y), (va0, _, va_y), (te0, _, te_y) = T.siamese_scenes(4)
    assert len(tr0) > 0 and len(va0) > 0 and len(te0) > 0
    assert tr0.shape[1:] == (6, 64, 64)


# ----------------------------------------------------------- benchmark -----------

def test_latency_is_positive():
    from mining_detector.hybrid_models.benchmark_lite import latency_ms

    model = M.build_model("CNNBiLSTM", lite=True)
    assert latency_ms(model, in_ch=6, size=64, siamese=False) > 0

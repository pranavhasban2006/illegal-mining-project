"""Real labeled segmentation dataset: 10 image/mask TIFF pairs (512x512,
3-band, binary per-pixel mining mask), tiled into 64x64 patches for
training. Source: public Open-Pit Mining Areas Dataset, downloaded to
data/external/open_pit_mining_dataset/ (gitignored -- large binary data;
re-download to reproduce, see README).

This is real ground truth -- genuine pixel-level masks, not the
percentile/concession proxies used elsewhere in this project -- but only
10 source images, drawn from 2 underlying scenes (image filenames encode a
scene id + tile index, e.g. "0000820418_109"). Held-out test tiles are
chosen from BOTH scenes so evaluation isn't limited to one location, but
generalization to genuinely unseen terrain is still an open question with
only 2 scenes total -- stated plainly, not glossed over.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "data", "external",
    "open_pit_mining_dataset", "Data",
)
IMG_DIR = os.path.join(DATA_DIR, "1-images")
LBL_DIR = os.path.join(DATA_DIR, "2-labels")

PATCH_SIZE = 64

# Held out entirely from training -- one tile from each of the two source
# scenes, so the test set isn't drawn from a single location.
TEST_TILES = {"image_0000820418_95.TIF", "image_0005680453_254.TIF"}


def _list_tiles():
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(IMG_DIR, "*.TIF")))


def _load_pair(tile_name: str):
    img_path = os.path.join(IMG_DIR, tile_name)
    lbl_path = os.path.join(LBL_DIR, "label_" + tile_name.replace("image_", ""))
    # Files are actually named image_<id>.TIF / label_<id>.TIF already.
    lbl_path = os.path.join(LBL_DIR, tile_name.replace("image_", "label_"))

    with rasterio.open(img_path) as src:
        img = src.read().astype(np.float32)  # (3, 512, 512)
    with rasterio.open(lbl_path) as src:
        mask = src.read(1).astype(np.float32)  # (512, 512)

    img = np.clip(img / 4500.0, 0, 1)  # normalize using the dataset's observed max range
    return img, mask


def _tile_patches(img: np.ndarray, mask: np.ndarray, patch_size: int = PATCH_SIZE):
    """Non-overlapping patch grid over a (3,H,W) image and (H,W) mask."""
    _, h, w = img.shape
    patches = []
    for r in range(0, h - patch_size + 1, patch_size):
        for c in range(0, w - patch_size + 1, patch_size):
            img_patch = img[:, r:r + patch_size, c:c + patch_size]
            mask_patch = mask[r:r + patch_size, c:c + patch_size]
            patches.append((img_patch, mask_patch))
    return patches


def build_patch_lists():
    """Returns (train_patches, test_patches), each a list of (img, mask) arrays."""
    train_patches, test_patches = [], []
    for tile_name in _list_tiles():
        img, mask = _load_pair(tile_name)
        patches = _tile_patches(img, mask)
        if tile_name in TEST_TILES:
            test_patches.extend(patches)
        else:
            train_patches.extend(patches)
    return train_patches, test_patches


class SegPatchDataset(Dataset):
    """(image, pixel-mask) pairs for the segmentation-capable architectures
    (U-Net+Transformer, ResNet+Attention U-Net)."""

    def __init__(self, patches):
        self.patches = patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        img, mask = self.patches[idx]
        return torch.from_numpy(img), torch.from_numpy(mask)


class ClsPatchDataset(Dataset):
    """(image, patch-level label) pairs for the classification-only
    architectures (YOLO+ViT+LSTM, CNN+BiLSTM). Label = 1 if >=5% of the
    patch's pixels are mining, matching a "meaningfully contains mining"
    threshold rather than "any single mining pixel"."""

    def __init__(self, patches, mining_threshold: float = 0.05):
        self.patches = patches
        self.threshold = mining_threshold

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        img, mask = self.patches[idx]
        label = 1.0 if mask.mean() >= self.threshold else 0.0
        return torch.from_numpy(img), torch.tensor(label, dtype=torch.float32)

"""Load the Open-Pit Mining Areas Dataset, cut it into patches, and split it fairly.

The dataset is not included in this project. Place its Data folder at
data/external/open_pit_mining_dataset/Data/ (see scripts/fetch_datasets.py openpit).

Fair split
----------
The dataset has about 10 source image/mask tiles. Patches cut from one tile look alike,
so if patches from the same tile land in both training and test, the score is inflated.
This module therefore holds out WHOLE tiles: 2 for test, 1 for validation, the rest for
training. No patch is in two sets.

Patches
-------
Each tile is cut into non-overlapping 64 x 64 patches. Edge pixels that do not fill a
whole patch are dropped. Patch-level labels (for classifiers) are "mining" when at least
PATCH_POSITIVE_FRACTION of the pixels are mining.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_DIR = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = REPO_DIR / "data" / "external" / "open_pit_mining_dataset" / "Data"
PATCH_SIZE = 64
PATCH_POSITIVE_FRACTION = 0.01
N_TEST_TILES = 2
N_VAL_TILES = 1


class DatasetMissing(RuntimeError):
    """The Open-Pit folder is absent or incomplete. The message says what to do."""


@dataclass
class Splits:
    """Patches for each set. Images are (N, C, H, W); masks are (N, H, W) with 0/1."""

    train_x: np.ndarray
    train_y: np.ndarray
    val_x: np.ndarray
    val_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    test_tile_ids: list[int]
    mean: np.ndarray        # per-channel mean, from training patches only
    std: np.ndarray         # per-channel standard deviation, from training patches only
    source: str             # "open_pit" or "synthetic_standin"


# ----------------------------------------------------------------- files ---------

def _is_mask(path: Path) -> bool:
    return "mask" in path.stem.lower()


def list_pairs(folder: Path) -> list[tuple[Path, Path]]:
    """Pair each image with its mask by file name, the same way fetch_datasets.py checks.

    A file is a mask if its name contains 'mask'. The pair key is the name with that
    word removed. Raises DatasetMissing if the folder or any pair is missing.
    """
    if not folder.is_dir():
        raise DatasetMissing(
            f"The Open-Pit folder was not found at {folder}. Put the dataset's Data folder there, "
            "or run: python scripts/fetch_datasets.py openpit"
        )
    tifs = sorted(p for p in folder.iterdir() if p.suffix.lower() in (".tif", ".tiff"))
    masks = {p.stem.lower().replace("mask", "").strip("_- "): p for p in tifs if _is_mask(p)}
    images = {p.stem.lower().strip("_- "): p for p in tifs if not _is_mask(p)}
    missing = sorted(set(masks) - set(images))
    if missing:
        raise DatasetMissing(f"These masks have no matching image: {missing}")
    pairs = [(images[k], masks[k]) for k in sorted(masks)]
    if not pairs:
        raise DatasetMissing(f"No image/mask pairs found in {folder}.")
    return pairs


def read_pair(image_path: Path, mask_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read one image/mask GeoTIFF pair. Returns (C, H, W) float32 and (H, W) uint8 mask."""
    import rasterio

    with rasterio.open(image_path) as src:
        img = src.read().astype(np.float32)
    with rasterio.open(mask_path) as src:
        mask = (src.read(1) > 0).astype(np.uint8)
    if img.shape[1:] != mask.shape:
        raise ValueError(f"{image_path.name} and {mask_path.name} have different sizes.")
    return img, mask


def load_open_pit(folder: Path = DEFAULT_DATA_DIR) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return a list of (image, mask) tiles from the Open-Pit folder."""
    return [read_pair(i, m) for i, m in list_pairs(folder)]


# ---------------------------------------------------------------- patches --------

def tile_pair(img: np.ndarray, mask: np.ndarray, size: int = PATCH_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """Cut one tile into non-overlapping size x size patches.

    Returns patches of shape (N, C, size, size) and masks of shape (N, size, size).
    Pixels at the right and bottom edges that do not fill a patch are dropped.
    """
    C, H, W = img.shape
    rows, cols = H // size, W // size
    patches, masks = [], []
    for r in range(rows):
        for c in range(cols):
            ys, xs = slice(r * size, (r + 1) * size), slice(c * size, (c + 1) * size)
            patches.append(img[:, ys, xs])
            masks.append(mask[ys, xs])
    if not patches:
        return np.zeros((0, C, size, size), np.float32), np.zeros((0, size, size), np.uint8)
    return np.stack(patches).astype(np.float32), np.stack(masks).astype(np.uint8)


def patch_labels(masks: np.ndarray) -> np.ndarray:
    """Patch-level label: 1 if the mining share of the patch is at least PATCH_POSITIVE_FRACTION."""
    share = masks.reshape(len(masks), -1).mean(axis=1)
    return (share >= PATCH_POSITIVE_FRACTION).astype(np.int64)


def tile_indices(n_tiles: int, n_test: int = N_TEST_TILES, n_val: int = N_VAL_TILES,
                 seed: int = 0) -> tuple[list[int], list[int], list[int]]:
    """Choose whole tiles for train, validation and test. Returns (train, val, test) index lists."""
    if n_tiles < n_test + n_val + 1:
        raise ValueError(f"Need at least {n_test + n_val + 1} tiles, got {n_tiles}.")
    order = np.random.default_rng(seed).permutation(n_tiles).tolist()
    test = sorted(order[:n_test])
    val = sorted(order[n_test:n_test + n_val])
    train = sorted(order[n_test + n_val:])
    return train, val, test


def build_splits(tiles: list[tuple[np.ndarray, np.ndarray]], seed: int = 0,
                 source: str = "open_pit") -> Splits:
    """Cut every tile, split by whole tile, and normalise with training statistics only."""
    train_ids, val_ids, test_ids = tile_indices(len(tiles), seed=seed)

    def collect(ids):
        xs, ys = [], []
        for i in ids:
            p, m = tile_pair(*tiles[i])
            xs.append(p)
            ys.append(m)
        return np.concatenate(xs), np.concatenate(ys)

    train_x, train_y = collect(train_ids)
    val_x, val_y = collect(val_ids)
    test_x, test_y = collect(test_ids)
    mean = train_x.mean(axis=(0, 2, 3))
    std = train_x.std(axis=(0, 2, 3)) + 1e-6
    norm = lambda x: (x - mean[None, :, None, None]) / std[None, :, None, None]  # noqa: E731
    return Splits(
        train_x=norm(train_x), train_y=train_y, val_x=norm(val_x), val_y=val_y,
        test_x=norm(test_x), test_y=test_y, test_tile_ids=test_ids,
        mean=mean, std=std, source=source,
    )


def augment(img: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Random flips and a random 90-degree rotation. Use on training patches only."""
    k = int(rng.integers(0, 4))
    img = np.rot90(img, k, axes=(1, 2))
    mask = np.rot90(mask, k, axes=(0, 1))
    if rng.random() < 0.5:
        img, mask = img[:, :, ::-1], mask[:, ::-1]
    if rng.random() < 0.5:
        img, mask = img[:, ::-1, :], mask[::-1, :]
    return np.ascontiguousarray(img), np.ascontiguousarray(mask)


# ---------------------------------------------------------- stand-in data -------

def standin_tiles(n_scenes: int = 10, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """Synthetic tiles that stand in for Open-Pit data when the real set is absent.

    They come from mining_detector.synthetic, so their answers are known. Results trained
    on them are only a check that the code runs. They say nothing about real mines.
    """
    from mining_detector.synthetic import make_synthetic_scene

    bands = ["B02", "B03", "B04", "B08", "B11", "B12"]
    tiles = []
    for k in range(n_scenes):
        scene = make_synthetic_scene(seed=seed + k, size=192)
        img = np.stack([scene.bands_t1[b][:192, :192] for b in bands]).astype(np.float32)
        mask = scene.truth_new_mask[:192, :192].astype(np.uint8)
        tiles.append((img, mask))
    return tiles


class PatchDataset:
    """Minimal in-memory dataset: returns (image, target) as numpy arrays.

    Training sets pass augment=True, so each epoch sees randomly flipped and rotated patches.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, augment_on: bool = False, seed: int = 0):
        self.x, self.y = x, y
        self.augment_on = augment_on
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int):
        img, target = self.x[i], self.y[i]
        if self.augment_on:
            img, target = augment(img, target, self.rng)
        return img, target

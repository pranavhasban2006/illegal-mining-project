"""Models page: how the deep-learning models scored, and a 'Try it' widget.

The scores come from files written by train_openpit.py and benchmark_lite.py. Nothing
on this page is typed in by hand.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from lib.paths import REPO_DIR

RESULTS_DIR = REPO_DIR / "data" / "hybrid_models_results"
WEIGHTS_DIR = RESULTS_DIR / "weights"
sys.path.insert(0, str(REPO_DIR / "src"))
from mining_detector.hybrid_models.openpit_dataset import DEFAULT_DATA_DIR as OPENPIT_DIR

st.set_page_config(page_title="Models: Illegal Mining Detector", page_icon="🧠", layout="wide")
st.title("Deep-learning models")

st.warning(
    "**Honesty panel.** The test patches come from only 2 source scenes of the Open-Pit "
    "dataset. These scores measure how well the models fit new patches from scenes they "
    "have seen during training. They do NOT show that a model will work in a new country or "
    "on a different kind of mine."
)


def load_json(path: Path):
    """Return the parsed JSON file, or None if it is missing or unreadable."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


openpit = load_json(RESULTS_DIR / "openpit_results.json")
standin = load_json(RESULTS_DIR / "standin_results.json")
results_doc = openpit or standin

if results_doc is None:
    st.info(
        "No trained-model results yet. Put the Open-Pit dataset in place, then run:\n\n"
        "`python -m mining_detector.hybrid_models.train_openpit`"
    )
    st.stop()

if results_doc["demo_only"]:
    st.warning("Trained on synthetic data. These scores are not real-world accuracy.")
else:
    st.success(f"Trained on the Open-Pit dataset ({results_doc['n_train_patches']} training patches).")

st.caption(results_doc["honesty_note"])

rows = pd.DataFrame(results_doc["results"])
rows["variant"] = rows["lite"].map({True: "lite", False: "full"})
st.subheader("Scores on the held-out test tiles")
st.dataframe(
    rows[["name", "variant", "task", "params", "train_seconds", "best_epoch", "iou", "dice",
          "accuracy", "f1"]],
    hide_index=True, use_container_width=True,
    column_config={
        "params": st.column_config.NumberColumn("Parameters", help="How many numbers the model learns."),
        "train_seconds": st.column_config.NumberColumn("Train time (s)", format="%.1f"),
        "iou": st.column_config.NumberColumn("IoU", help="Overlap of predicted and true mining pixels. 1 is perfect.", format="%.3f"),
        "dice": st.column_config.NumberColumn("Dice", help="Like IoU but gives more weight to overlap. 1 is perfect.", format="%.3f"),
        "f1": st.column_config.NumberColumn("F1", help="Balances missed sites and false alarms. 1 is perfect.", format="%.3f"),
        "accuracy": st.column_config.NumberColumn("Accuracy", help="Share of patches labelled correctly.", format="%.3f"),
    },
)

c1, c2 = st.columns(2)
seg = rows.dropna(subset=["iou"])
if not seg.empty:
    c1.markdown("**Segmentation: IoU** (higher is better)")
    c1.bar_chart(seg.assign(label=seg["name"] + " (" + seg["variant"] + ")").set_index("label")["iou"])
cls = rows.dropna(subset=["f1"])
if not cls.empty:
    c2.markdown("**Classification: F1** (higher is better)")
    c2.bar_chart(cls.assign(label=cls["name"] + " (" + cls["variant"] + ")").set_index("label")["f1"])

st.subheader("Full vs lite: size and speed")
bench = load_json(RESULTS_DIR / "benchmark_lite.json")
if bench:
    st.dataframe(pd.DataFrame(bench["rows"]), hide_index=True, use_container_width=True,
                 column_config={"latency_ms": st.column_config.NumberColumn(
                     "CPU latency (ms, batch of 8)", help="Time for one forward pass on this computer.")})
    st.caption("Latency was measured on the computer that ran benchmark_lite.py, so compare the rows to each other.")
else:
    st.caption("Run `python -m mining_detector.hybrid_models.benchmark_lite` to fill this table.")

# ---- Try it ----------------------------------------------------------------------
st.subheader("Try it: run a trained model on your own image")
st.write(
    "Upload a GeoTIFF crop with the same number of bands the model was trained on. The page "
    "tiles it into 64 x 64 patches and shows the prediction. The two-date Siamese model is not "
    "offered here, because it needs two images from two dates."
)

weight_files = sorted(p for p in WEIGHTS_DIR.glob("*.pt") if "SiameseCNNTransformer" not in p.name) \
    if WEIGHTS_DIR.is_dir() else []
if not weight_files:
    st.caption("No saved weights yet. Train the models first.")
else:
    choice = st.selectbox("Model", weight_files, format_func=lambda p: p.stem)
    upload = st.file_uploader("Satellite image crop (GeoTIFF)", type=["tif", "tiff"])
    if upload is not None and st.button("Run the model"):
        try:
            import torch
            from rasterio.io import MemoryFile
            from mining_detector.hybrid_models.models import SEGMENTATION_MODELS, build_model
            from mining_detector.hybrid_models.openpit_dataset import tile_pair

            saved = torch.load(choice, map_location="cpu", weights_only=True)
            with MemoryFile(upload.getvalue()) as mem, mem.open() as src:
                img = src.read().astype(np.float32)
            if img.shape[0] != saved["in_ch"]:
                raise ValueError(f"This model expects {saved['in_ch']} bands; your image has {img.shape[0]}.")
            mean = np.array(saved["mean"], np.float32)[:, None, None]
            std = np.array(saved["std"], np.float32)[:, None, None]
            norm = (img - mean) / std
            size = saved["image_size"]
            h, w = (norm.shape[1] // size) * size, (norm.shape[2] // size) * size
            if h == 0 or w == 0:
                raise ValueError(f"The image must be at least {size} x {size} pixels.")
            norm = norm[:, :h, :w]
            model = build_model(saved["name"], in_ch=saved["in_ch"], lite=saved["lite"],
                                image_size=size)
            model.load_state_dict(saved["state"])
            model.eval()
            patches, _ = tile_pair(norm, np.zeros((h, w), np.uint8), size)
            with torch.no_grad():
                out = model(torch.from_numpy(patches)).numpy()
            if saved["name"] in SEGMENTATION_MODELS:
                mask = np.zeros((h, w), bool)
                k = 0
                for r in range(h // size):
                    for c in range(w // size):
                        mask[r * size:(r + 1) * size, c * size:(c + 1) * size] = out[k, 0] > 0
                        k += 1
                st.image(np.where(mask, 255, 0).astype(np.uint8), clamp=True,
                         caption=f"Predicted mining pixels (white) from {choice.stem}. Threshold: probability 0.5.")
                st.metric("Share of crop predicted as mining", f"{mask.mean() * 100:.1f} %",
                          help="Predicted mining pixels divided by all pixels in the crop.")
            else:
                labels = out.argmax(1)
                grid = labels.reshape(h // size, w // size)
                st.write("Patch grid (1 = predicted mining, 0 = not mining):")
                st.dataframe(pd.DataFrame(grid), hide_index=True)
                st.metric("Patches predicted as mining", f"{int(labels.sum())} of {len(labels)}")
            st.caption("This prediction is only as reliable as the test scores above.")
        except (ValueError, RuntimeError, OSError) as err:  # show the reason, not a stack trace
            st.error(f"Could not run the model: {err}")

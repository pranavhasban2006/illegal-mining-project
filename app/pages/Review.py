"""Review page: label small satellite chips as mining, not mining, or unsure.

The labels are saved to CSV in the format the engine reads for validation
(columns lon, lat, label). The engine never trains on them unless you run it with
hard_cases_path, which keeps a separate hold-out check.
"""

import csv
from pathlib import Path

import pandas as pd
import streamlit as st

from lib.paths import OUTPUTS_DIR, REPO_DIR

st.set_page_config(page_title="Review: Illegal Mining Detector", page_icon="✅", layout="wide")
st.title("Review: label satellite chips")
st.write(
    "Each small picture is a place the detector was unsure about. Look at it and choose what you "
    "see. Your labels help check how good the detector is. Choose **Unsure** when you cannot tell."
)

LABELS = {"Mining": 1, "Not mining": 0, "Unsure": "unsure"}
SAVE_PATH = OUTPUTS_DIR / "review" / "hard_cases.csv"

templates = sorted(OUTPUTS_DIR.glob("*/chips/review_template.csv"), reverse=True)
if not templates:
    st.info(
        "No review chips yet. The engine creates them when a run is started with review chips "
        "switched on. Run an analysis with that option, then come back here."
    )
    st.stop()

chosen_template = st.selectbox("Review set", templates, format_func=lambda p: p.parent.parent.name)
chips = pd.read_csv(chosen_template)
chips_dir = chosen_template.parent

saved = pd.read_csv(SAVE_PATH) if SAVE_PATH.is_file() else pd.DataFrame(columns=["chip_id", "label"])
done_ids = set(saved["chip_id"].astype(str)) if not saved.empty else set()
st.progress(len(done_ids & set(chips["chip_id"].astype(str))) / max(len(chips), 1),
            text=f"{len(done_ids)} of {len(chips)} chips labelled")

index = st.number_input("Chip number", min_value=1, max_value=len(chips), value=1, step=1) - 1
row = chips.iloc[int(index)]
picture = chips_dir / str(row["file"])
if picture.is_file():
    st.image(str(picture), use_container_width=True)
else:
    st.warning(f"Picture missing: {picture.name}")
st.caption(f"Chip {row['chip_id']} · category: {row['category']} · "
           f"location: {row['lat']:.5f}, {row['lon']:.5f}")

choice = st.radio("What do you see?", list(LABELS), horizontal=True)
note = st.text_input("Notes (optional)")

if st.button("Save this label", type="primary"):
    rows = saved.to_dict("records") if not saved.empty else []
    rows = [r for r in rows if str(r["chip_id"]) != str(row["chip_id"])]
    rows.append({"lon": row["lon"], "lat": row["lat"], "label": LABELS[choice],
                 "chip_id": row["chip_id"], "category": row["category"], "notes": note})
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(SAVE_PATH, index=False, quoting=csv.QUOTE_MINIMAL)
    st.success(f"Saved to {SAVE_PATH.relative_to(REPO_DIR)}")
    st.rerun()

if not saved.empty:
    st.download_button("Download my labels (CSV)", data=SAVE_PATH.read_bytes(),
                       file_name="hard_cases.csv", mime="text/csv")
    st.caption("Keep the labels in the outputs folder. Pass the CSV to the engine's "
               "hard_cases_path option to test the detector on your labels.")

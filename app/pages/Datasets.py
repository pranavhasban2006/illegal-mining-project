"""Datasets page: what data is on disk, and buttons to fetch what is missing."""

import subprocess
import sys
from pathlib import Path

import streamlit as st

from lib.paths import REPO_DIR

SCRIPT = REPO_DIR / "scripts" / "fetch_datasets.py"
st.set_page_config(page_title="Datasets: Illegal Mining Detector", page_icon="📦", layout="wide")
st.title("Datasets")
st.write(
    "The detector needs a few data files. This page shows which ones are on your computer. "
    "Use the buttons to fetch the missing ones. The log shows exactly what happens."
)

# Import the checklist from the same script the buttons run, so both always agree.
sys.path.insert(0, str(SCRIPT.parent))
import fetch_datasets as fd  # noqa: E402


def run_fetch(args: list[str]) -> None:
    """Run fetch_datasets.py with args and stream its output into the page."""
    cmd = [sys.executable, str(SCRIPT), *args]
    st.code(" ".join(["python", "scripts/fetch_datasets.py", *args]), language="bash")
    log = st.empty()
    lines: list[str] = []
    with subprocess.Popen(cmd, cwd=REPO_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8", errors="replace") as proc:
        for line in proc.stdout:
            lines.append(line.rstrip("\r\n"))
            log.code("\n".join(lines[-40:]))
        code = proc.wait()
    if code == 0:
        st.success("Done. Refresh this page to see the updated checklist.")
    else:
        st.error("The command did not finish. Read the log above, then follow the manual steps it gives.")


st.subheader("Checklist")
for name, present, detail in fd.checklist():
    cols = st.columns([1, 3, 4])
    cols[0].markdown("✅ **Present**" if present else "✅ **Synthetic**")
    cols[1].markdown(f"**{name}**")
    cols[2].caption(detail)

st.divider()
st.subheader("Fetch the missing datasets")

mining_present = fd.MINING_TARGET.exists()
st.markdown("**Global mining footprint** (downloads about 24 MB, CC BY-SA 4.0)")
if st.button("Download mining footprint", disabled=mining_present, type="primary"):
    run_fetch(["mining"])
if mining_present:
    st.caption("Already downloaded.")

st.markdown("**Open-Pit mining dataset**")
st.write(
    "This one cannot be downloaded from this page. Put its `Data` folder here:"
)
st.code(str(fd.OPENPIT_DIR))
if st.button("Check the Open-Pit folder"):
    run_fetch(["openpit"])
st.caption(fd.OPENPIT_SOURCE_NOTE)

st.markdown("**Concessions** (the official mining lease map for your country)")
source = st.text_input("Local file path or download link",
                       placeholder=r"C:\Downloads\concessions.geojson",
                       help="GeoJSON, shapefile (.shp or a .zip of one) or GeoPackage.")
use_aoi = st.checkbox("Clip to my area of interest", value=False)
aoi_values = [0.0, 0.0, 0.0, 0.0]
if use_aoi:
    a1, a2, a3, a4 = st.columns(4)
    aoi_values = [
        a1.number_input("West", value=-71.0, format="%.4f"),
        a2.number_input("South", value=-13.0, format="%.4f"),
        a3.number_input("East", value=-70.8, format="%.4f"),
        a4.number_input("North", value=-12.8, format="%.4f"),
    ]
if st.button("Prepare concessions", disabled=not source.strip()):
    args = ["concessions", source.strip()]
    if use_aoi:
        args += ["--aoi", *[str(v) for v in aoi_values]]
    run_fetch(args)

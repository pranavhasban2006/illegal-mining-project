"""Analyze page: a four-step wizard that runs the detector on an area you choose.

Step 1  Pick the area (draw it, type coordinates, or use an example)
Step 2  Pick the two dates
Step 3  Optionally add a concessions file
Step 4  Run, watch progress, then see the results
"""

import sys
import time
from datetime import date, datetime
from pathlib import Path

import folium
import streamlit as st
from folium.plugins import Draw
from streamlit_folium import st_folium

from lib.area import (
    PRESETS, bbox_area_km2, default_dates, format_dates_note, parse_bbox_text, validate_bbox,
)
from lib.paths import OUTPUTS_DIR, REPO_DIR
from lib.runner import RunJob, what_to_try

sys.path.insert(0, str(REPO_DIR / "src"))
UPLOAD_DIR = REPO_DIR / "data" / "external" / "uploads"

st.set_page_config(page_title="Analyze: Illegal Mining Detector", page_icon="🛰️", layout="wide")
st.title("Analyze my area")

STEPS = ["1. Area", "2. Dates", "3. Concessions", "4. Run"]
step = st.session_state.get("wizard_step", 0)
st.progress((step + 1) / len(STEPS), text=f"Step {step + 1} of {len(STEPS)}: {STEPS[step][3:]}")
cols = st.columns(len(STEPS))
for i, name in enumerate(STEPS):
    cols[i].markdown(f"**{name}**" if i == step else name)

st.session_state.setdefault("bbox", PRESETS["Example: Peru (engine default)"])


def go(to: int) -> None:
    st.session_state["wizard_step"] = to


# ---- Step 1: area -----------------------------------------------------------------
if step == 0:
    st.write("Choose the part of the map to check. Keep it small: large areas take longer and "
             "use more memory.")
    mode = st.radio("How do you want to choose?",
                    ["Use an example", "Type the coordinates", "Draw on the map"], horizontal=True)
    if mode == "Use an example":
        name = st.selectbox("Example area", list(PRESETS))
        st.session_state["bbox"] = PRESETS[name]
        st.caption("The Ghana example is an approximate box. Adjust it if you know the area better.")
    elif mode == "Type the coordinates":
        text = st.text_input("West, south, east, north (decimal degrees)",
                             value=" ".join(f"{v:.4f}" for v in st.session_state["bbox"]),
                             help="West and east are longitudes; south and north are latitudes. "
                                  "Negative longitude is west of Greenwich.")
        try:
            st.session_state["bbox"] = parse_bbox_text(text)
        except ValueError as err:
            st.error(str(err))
    else:
        st.write("Draw a rectangle with the square tool on the left of the map.")
        west, south, east, north = st.session_state["bbox"]
        fmap = folium.Map(location=[(south + north) / 2, (west + east) / 2], zoom_start=10,
                          tiles="OpenStreetMap")
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri World Imagery", name="Satellite").add_to(fmap)
        Draw(export=False, draw_options={"polyline": False, "polygon": False, "circle": False,
                                         "marker": False, "circlemarker": False}).add_to(fmap)
        folium.LayerControl().add_to(fmap)
        out = st_folium(fmap, height=420, use_container_width=True,
                        returned_objects=["last_active_drawing"])
        drawn = (out or {}).get("last_active_drawing")
        if drawn:
            coords = drawn["geometry"]["coordinates"][0]
            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]
            st.session_state["bbox"] = [min(lons), min(lats), max(lons), max(lats)]
        st.caption("The box you drew is used. Draw again to replace it.")

    bbox = st.session_state["bbox"]
    problems = validate_bbox(bbox)
    st.write(f"Area in view: **{bbox_area_km2(bbox):,.1f} km²** "
             f"(west {bbox[0]:.4f}, south {bbox[1]:.4f}, east {bbox[2]:.4f}, north {bbox[3]:.4f})")
    for p in problems:
        st.warning(p)
    if st.button("Next: dates", type="primary", disabled=bool(problems)):
        go(1)
        st.rerun()

# ---- Step 2: dates ----------------------------------------------------------------
elif step == 1:
    st.write("The tool compares two satellite images of the same area. A year apart on the same "
             "calendar date is the default, because the season matches and seasonal greenery is less "
             "likely to be mistaken for mining.")
    t1 = st.date_input("Later date (t1)", value=st.session_state.get("t1", date(2025, 7, 27)),
                       max_value=date.today(),
                       help="The date of the newer image. Satellite passes are not daily, so the tool "
                            "uses the nearest clear image within 15 days.")
    t0_default, _ = default_dates(t1)
    t0 = st.date_input("Earlier date (t0)", value=st.session_state.get("t0", t0_default),
                       max_value=date.today(),
                       help="The date of the older image. It must be before the later date.")
    st.session_state["t0"], st.session_state["t1"] = t0, t1
    if t0 >= t1:
        st.error("The earlier date must be before the later date.")
    else:
        st.info(format_dates_note(t0, t1))
    c1, c2 = st.columns(2)
    if c1.button("Back"):
        go(0)
        st.rerun()
    if c2.button("Next: concessions", type="primary", disabled=t0 >= t1):
        go(2)
        st.rerun()

# ---- Step 3: concessions ----------------------------------------------------------
elif step == 2:
    st.write("A concession is an official permit to mine an area. The tool compares each new "
             "disturbance with these permits.")
    st.error("**Without a concessions file, every alert is marked 'illegal'.** The tool cannot tell "
             "a permitted mine from an unpermitted one without the permit map.")
    up = st.file_uploader("Concessions file (GeoJSON)", type=["geojson", "json"])
    if up is not None:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        dest = UPLOAD_DIR / f"concessions_{datetime.now():%Y%m%d_%H%M%S}.geojson"
        dest.write_bytes(up.getvalue())
        st.session_state["concessions_path"] = str(dest)
        st.success(f"Saved {up.name}.")
    if st.session_state.get("concessions_path") and up is None:
        st.caption(f"Using: {Path(st.session_state['concessions_path']).name}")
    c1, c2, c3 = st.columns(3)
    if c1.button("Back"):
        go(1)
        st.rerun()
    if c2.button("Skip (every alert will be 'illegal')"):
        st.session_state["concessions_path"] = None
        go(3)
        st.rerun()
    if c3.button("Next: run", type="primary"):
        go(3)
        st.rerun()

# ---- Step 4: run ------------------------------------------------------------------
elif step == 3:
    bbox = st.session_state["bbox"]
    t0, t1 = st.session_state.get("t0"), st.session_state.get("t1")
    st.write(f"Area: {bbox_area_km2(bbox):,.1f} km². Dates: {t0} and {t1}. "
             f"Concessions: {'none (all alerts will be illegal)' if not st.session_state.get('concessions_path') else Path(st.session_state['concessions_path']).name}.")
    job: RunJob | None = st.session_state.get("job")
    if job is None:
        c1, c2 = st.columns(2)
        if c1.button("Back"):
            go(2)
            st.rerun()
        if c2.button("Run the detector", type="primary"):
            from mining_detector.pipeline import run_live

            out_dir = OUTPUTS_DIR / f"run_{datetime.now():%Y%m%d_%H%M%S}"
            job = RunJob()
            job.start(run_live, aoi_bbox=tuple(bbox), t0_date=str(t0), t1_date=str(t1),
                      out_dir=str(out_dir), concessions_path=st.session_state.get("concessions_path"))
            st.session_state["job"] = job
            st.session_state["job_out"] = str(out_dir)
            st.rerun()
    else:
        bar = st.progress(job.fraction, text=job.message)
        status = st.empty()
        while job.is_running():
            bar.progress(job.fraction, text=job.message)
            status.caption(" · ".join(job.log[-3:]))
            time.sleep(0.5)
        if job.error:
            st.error(f"The run did not finish: {job.error}")
            st.info("What to try: " + what_to_try(job.error_type, job.error))
            if st.button("Start again"):
                st.session_state.pop("job", None)
                st.rerun()
        else:
            bar.progress(1.0, text="Finished")
            st.success("Done. Opening the results.")
            st.session_state["run_folder"] = st.session_state["job_out"]
            st.session_state.pop("job", None)
            st.switch_page("pages/1_Results.py")

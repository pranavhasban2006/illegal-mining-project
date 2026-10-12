"""Results page: map, summary numbers, alert table, downloads and caveats."""

from pathlib import Path

import streamlit as st
from streamlit_folium import st_folium

from lib.loaders import (
    STATUSES,
    alerts_to_csv,
    caveat_lines,
    choose_data_folder,
    filter_alerts,
    find_runs,
    load_run,
    summarize,
)
from lib.maps import build_map, legend_html
from lib.paths import OUTPUTS_DIR, SAMPLE_RUN_DIR
from lib.report_html import build_report_html

st.set_page_config(page_title="Results: Illegal Mining Detector", page_icon="🗺️", layout="wide")
st.title("Results")

# ---- Sidebar: which data folder is shown -------------------------------------------
st.sidebar.header("Data folder")
requested = st.session_state.get("run_folder")
default_folder, default_is_sample = choose_data_folder(OUTPUTS_DIR, SAMPLE_RUN_DIR)
if requested:
    folder, is_sample = Path(requested), Path(requested) == SAMPLE_RUN_DIR
else:
    folder, is_sample = default_folder, default_is_sample

runs = find_runs(OUTPUTS_DIR)
options = [str(r) for r in runs] + [str(SAMPLE_RUN_DIR)]
if folder is not None and str(folder) not in options:
    options.insert(0, str(folder))
if folder is None:
    st.error(
        f"No results found in `{OUTPUTS_DIR}` and the sample run is missing. "
        "Go to Home and try the demo."
    )
    st.stop()
choice = st.sidebar.selectbox(
    "Run to show",
    options=options,
    index=options.index(str(folder)),
    format_func=lambda p: ("SAMPLE: " if Path(p) == SAMPLE_RUN_DIR else "") + Path(p).name,
    help="Older runs stay in the outputs folder. Pick one here to compare.",
)
folder = Path(choice)
is_sample = folder == SAMPLE_RUN_DIR
st.sidebar.caption(f"Reading: `{folder}`")

try:
    run = load_run(folder, is_sample=is_sample)
except (FileNotFoundError, ValueError) as err:
    st.error(str(err))
    st.stop()

# ---- Banners -----------------------------------------------------------------------
if run.is_sample:
    st.warning(
        "**Showing SAMPLE data.** These alerts are invented for testing the app. "
        "They are not real mining detections."
    )
for msg in run.warnings:
    st.warning(msg)

report = run.report
if not run.is_sample and any("DEMO" in c.upper() for c in caveat_lines(report)):
    st.warning(
        "**This is DEMO output.** The engine made it from synthetic data to test the pipeline. "
        "It does not describe any real place."
    )
st.caption(
    f"Area: west {report['aoi_bbox'][0]}, south {report['aoi_bbox'][1]}, "
    f"east {report['aoi_bbox'][2]}, north {report['aoi_bbox'][3]}. "
    f"Compared {report['t0']} with {report['t1']}."
)

# ---- Caveats: always visible, never hidden -------------------------------------------
caveats = caveat_lines(report)
if caveats:
    with st.container(border=True):
        st.markdown("**Caveats: read before using these numbers**")
        for c in caveats:
            st.markdown(f"- {c}")

# ---- Summary cards -----------------------------------------------------------------
stats = summarize(run.alerts)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Alerts", stats["n_alerts"],
          help="Each alert is one area where the ground looks newly dug up or cleared.")
c2.metric("Total new disturbance", f"{stats['total_ha']:.2f} ha",
          help="ha = hectare = 10,000 square metres. The sum of all alert areas.")
c3.metric("Illegal", f"{stats['by_status_ha']['illegal']:.2f} ha",
          help="Outside every mining concession.")
c4.metric("Disputed", f"{stats['by_status_ha']['disputed']:.2f} ha",
          help="Inside an area whose status is unclear or contested.")

# ---- Filters -----------------------------------------------------------------------
st.subheader("Filter alerts")
f1, f2 = st.columns([2, 3])
chosen = f1.multiselect("Status", options=list(STATUSES), default=list(STATUSES),
                        help="Legal = inside a concession. Disputed = unclear. Illegal = outside.")
min_conf = f2.slider("Minimum confidence", 0.0, 1.0, 0.0, 0.05,
                     help="Confidence is how sure the model is that this is new mining. "
                          "Higher is more sure.")
shown = filter_alerts(run.alerts, chosen, min_conf)

# ---- Map and images ----------------------------------------------------------------
map_col, side_col = st.columns([3, 2])
with map_col:
    st.subheader("Map")
    st.markdown(legend_html(), unsafe_allow_html=True)
    visible_ids = set(shown["id"])
    visible_geojson = {
        **run.geojson,
        "features": [f for f in run.geojson["features"] if f["properties"]["id"] in visible_ids],
    }
    fmap = build_map(visible_geojson, shown, report["aoi_bbox"])
    st_folium(fmap, height=480, use_container_width=True, returned_objects=[])
    st.caption("Click a shape for its confidence and area. Use the layer button to switch to satellite.")

with side_col:
    st.subheader("Pictures")
    if run.change_map_path.is_file():
        st.image(str(run.change_map_path), caption="Change map: where the ground changed",
                 use_container_width=True)
    if run.probability_path.is_file():
        st.image(str(run.probability_path),
                 caption="Probability map: how likely each spot is new mining (0 to 1)",
                 use_container_width=True)

# ---- Table -------------------------------------------------------------------------
st.subheader(f"Alerts ({len(shown)} shown)")
st.dataframe(
    shown,
    use_container_width=True,
    hide_index=True,
    column_config={
        "confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1, format="%.2f"),
        "area_ha": st.column_config.NumberColumn("Area (ha)", format="%.2f"),
    },
)

# ---- Downloads ---------------------------------------------------------------------
st.subheader("Download")
d1, d2, d3 = st.columns(3)
d1.download_button("GeoJSON (for GIS software)", data=run.folder.joinpath("alerts.geojson").read_bytes(),
                   file_name="alerts.geojson", mime="application/geo+json", use_container_width=True)
d2.download_button("CSV (for spreadsheets)", data=alerts_to_csv(shown),
                   file_name="alerts_filtered.csv", mime="text/csv", use_container_width=True)
d3.download_button("One-page report (HTML)",
                   data=build_report_html(report, shown, is_sample=run.is_sample).encode("utf-8"),
                   file_name="mining_alert_report.html", mime="text/html", use_container_width=True,
                   help="Open in a browser and print to PDF.")

# ---- Model & trust -----------------------------------------------------------------
with st.expander("Model & trust: how the detector was checked"):
    model = report.get("model", {})
    st.write(f"Best model: **{model.get('best', 'unknown')}**")
    candidates = model.get("candidates", [])
    if candidates:
        st.dataframe(candidates, hide_index=True, use_container_width=True)
        st.caption("F1 is a score from 0 to 1 that balances missed sites and false alarms. "
                   "overfit_flag = True means the model did much better on the data it learned from "
                   "than on new data, so do not trust its score.")
    importance = model.get("feature_importance", {})
    if importance:
        st.bar_chart(importance, horizontal=True)
        st.caption("Feature importance: which measurements mattered most to the model. "
                   "Higher means more influence on the decision.")
    st.markdown("**Data sources**")
    for src in report.get("data_sources", []):
        st.markdown(f"- {src}")

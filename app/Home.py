"""Start page of the Illegal Mining Detector app.

Run it with scripts\\run_app.bat, or:  streamlit run app\\Home.py
"""

import streamlit as st

from lib.paths import OUTPUTS_DIR, SAMPLE_RUN_DIR

st.set_page_config(page_title="Illegal Mining Detector", page_icon="⛏️", layout="wide")

st.title("Illegal Mining Detector")
st.write(
    "This tool looks at two satellite pictures of the same area, taken a year apart, "
    "and finds places where the ground has been newly dug up or cleared, which is what "
    "surface mining looks like from space. It then compares each new site with the "
    "official mining concession maps. A site inside a legal concession is marked **legal**. "
    "A site in an area whose status is unclear is marked **disputed**. A site outside every "
    "concession is marked **illegal**. The output is a map and a list of alerts that you can download."
)

st.info(
    "Every number in this app is only as good as the satellite data and the maps behind it. "
    "The Results page always shows the caveats next to the numbers."
)

st.subheader("Step 1: See an example")
st.write(
    "The example uses a small sample run that is included with the project. Its values are "
    "invented for testing, so you can explore the app before you have real data."
)
if st.button("Try the demo", type="primary", use_container_width=True):
    st.session_state["run_folder"] = str(SAMPLE_RUN_DIR)
    st.switch_page("pages/1_Results.py")

st.subheader("Step 2: Analyze my area")
st.write(
    "Pick a region, two dates, and optionally a concessions file. The tool then downloads "
    "satellite images and runs the detection. A live run takes about 25 minutes on a normal "
    "computer, so start with a small area."
)
if st.button("Analyze my area", type="primary", use_container_width=True):
    st.switch_page("pages/Analyze.py")

st.caption(f"Results are read from: `{OUTPUTS_DIR}` (sample data is used if that folder is empty).")

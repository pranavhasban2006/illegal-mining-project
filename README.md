# Illegal Mining Detector

The Illegal Mining Detector looks at two Sentinel-2 satellite images of the same area, taken a
year apart, and finds places where the ground has been newly dug up or cleared. Each new site
is compared with the official mining concession map and marked **legal**, **disputed** or
**illegal**. The output is a map and a list of geo-referenced alerts you can download.

> **Alerts are not legal findings.** Check every site against official concession records
> before acting on it.

## 60-second quick start (Windows)

1. Open this folder and run **`scripts\setup.ps1`** from PowerShell:
   `powershell -ExecutionPolicy Bypass -File scripts\setup.ps1`. It creates the environment
   (it fetches Python 3.11 itself if you do not have it), installs the packages, downloads
   the global mining dataset and runs the tests. It is safe to run again. If you prefer, install
   Python 3.11 yourself from python.org first.
3. Double-click **`scripts\run_app.bat`**. Your browser opens the app.
4. Click **Try the demo** to see a sample run. Its numbers are invented for testing.

## What each page does

| Page | What you do there |
| --- | --- |
| **Home** | Read what the tool does, try the demo, or start your own analysis. |
| **Analyze my area** | Four steps: choose the area, choose two dates, add a concessions file, run. Progress is shown while it works. |
| **Results** | See the alerts on a map (street or satellite view), summary numbers, a filterable table, downloads (GeoJSON, CSV, a one-page HTML report), and the caveats. |
| **Datasets** | See which data files are present and fetch the missing ones with one click. |
| **Models** | See how the deep-learning models scored, and try one on your own image. |
| **Review** | Label small satellite chips as mining, not mining or unsure. Your labels can be used to check the detector. |

Hover over a **?** to read a plain-English explanation of a term. The glossary is in
[docs/USER_GUIDE.md](docs/USER_GUIDE.md).

## Use your own area

1. Open **Analyze my area**.
2. Choose *Draw on the map*, *Type the coordinates*, or an example.
3. Keep the area under the limit shown on the page (400 km² by default). Larger areas take
   longer and use more memory.
4. Add a concessions file (GeoJSON) if you have one. Without it, every alert is marked illegal.
   The Datasets page can convert a shapefile or a download link for you.
5. Run it. When it finishes, the Results page opens.

From the command line:

```
python -m mining_detector demo
python -m mining_detector live --example
python -m mining_detector live --bbox -72.10 -14.45 -71.95 -14.30 --t0 2024-07-27 --t1 2025-07-27 --concessions path\to\concessions.geojson
```

## FAQ

**Do I need an internet connection?** Yes, for a live run. The demo works offline.

**How long does a live run take?** On the test machine (Windows 11, CPU only), the example area took about 25 minutes (one run measured at 1,473 seconds). Most of that time is model comparison, so small areas and the Analyze page's progress bar are the way to keep it short.

**Why are my alerts all "illegal"?** You did not give a concessions file, so there was nothing
to compare them with.

**Why are there no alerts?** Cloud cover may have hidden the usable pixels, or nothing new may
have appeared between the two dates. Try a different pair of dates or a larger cloud allowance.

**Why does the Models page say DEMO ONLY?** The Open-Pit dataset was not available when those
models were trained, so they were trained on synthetic stand-in data. Their scores say nothing
about real mines.

**Where are the results saved?** In `outputs\`, one folder per run.

## Honest limitations

These are adapted from [docs/ENGINE_NOTES.md](docs/ENGINE_NOTES.md) and from the deep-learning work.

- **Weak labels.** The mining footprint dataset is public and coarse, and its coverage is patchy.
  Expect label noise.
- **No accuracy claim for new regions.** Scores come from a spatial hold-out inside one area.
  They say nothing about other terrain or other countries.
- **Model selection uses the same hold-out,** so the reported F1 is slightly optimistic.
- **Look-alikes are not handled.** Roads, quarries, farms and burn scars can look like mining.
- **Small patches can be missed** because of cloud masking and the 0.5 ha minimum.
- **The deep-learning models were not trained on real data in this repository.** The
  Open-Pit dataset must be placed by hand (see the Datasets page). Until it is, the Models
  page shows stand-in results only.
- **Demo output is synthetic.** It shows that the pipeline works, not a real detection.

## Data sources, licences and citations

| Data | Use in this tool | Licence | Cite |
| --- | --- | --- | --- |
| Sentinel-2 L2A imagery (AWS Earth Search, Microsoft Planetary Computer) | Satellite images | Copernicus open data | Copernicus Sentinel data, European Space Agency |
| Global Mining Footprint polygons (PANGAEA doi 10.1594/PANGAEA.942325) | Positive labels | CC BY-SA 4.0 | Maus, H. et al. (2022), Global Mining Footprint, PANGAEA |
| Open-Pit Mining Areas Dataset | Training the deep models (not included; place it by hand) | No licence file is included. Check the source before use. | Cite the original authors. Look up the citation on the dataset's own page. |
| Concessions file | Legal status of each alert | Depends on the source (for example RAISG). You supply it. | Name the source in your report. |

## Folder map

```
app/                 the web app (Home, Analyze, Results, Datasets, Models, Review)
app/lib/             app logic that does not depend on Streamlit (testable)
scripts/             setup, run, and dataset download scripts
src/mining_detector/ the detection engine (Terminal A) and hybrid_models/ (deep learning)
tests/               engine and experience tests (run: python -m pytest tests/engine tests/experience)
docs/                contract, engine notes, user guide, requests between terminals
data/                downloaded and cached data (not committed, except small results)
outputs/             one folder per analysis run
```

## Screenshots

<!-- Screenshot placeholders: add images to docs/img/ and link them here. -->

- Home: `docs/img/home.png` (to be added)
- Results: `docs/img/results.png` (to be added)
- Analyze: `docs/img/analyze.png` (to be added)

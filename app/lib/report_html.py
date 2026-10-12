"""Build a one-page HTML summary of a run that a non-expert can open and print.

Pure Python, no Streamlit. The caveats always appear in the output.
"""

from __future__ import annotations

import html

import pandas as pd

from lib.loaders import caveat_lines, summarize

COLOURS = {"legal": "#2e7d32", "disputed": "#ef8c00", "illegal": "#c62828"}


def build_report_html(report: dict, alerts: pd.DataFrame, is_sample: bool) -> str:
    """Return a complete HTML document summarising the run."""
    stats = summarize(alerts)
    esc = html.escape
    rows = "".join(
        f"<tr><td>{esc(str(r.id))}</td>"
        f"<td style='color:{COLOURS[r.legal_status]}'><b>{esc(r.legal_status)}</b></td>"
        f"<td>{r.confidence:.2f}</td><td>{r.area_ha:.2f}</td><td>{esc(str(r.detected_on))}</td></tr>"
        for r in alerts.itertuples()
    )
    cards = "".join(
        f"<div class='card'><div class='label'>{esc(s)}</div>"
        f"<div class='value' style='color:{COLOURS[s]}'>{stats['by_status_ha'][s]:.2f} ha</div>"
        f"<div class='label'>{stats['by_status_count'][s]} alerts</div></div>"
        for s in COLOURS
    )
    caveats = "".join(f"<li>{esc(c)}</li>" for c in caveat_lines(report))
    sample_banner = (
        "<div class='banner'>SAMPLE DATA: invented values for testing, not real detections.</div>"
        if is_sample else ""
    )
    bbox = ", ".join(f"{v:.4f}" for v in report.get("aoi_bbox", []))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Illegal Mining Detector: run summary</title>
<style>
  body {{ font-family: Segoe UI, Arial, sans-serif; margin: 24px; color: #222; }}
  h1 {{ font-size: 22px; margin-bottom: 4px; }}
  .meta {{ color: #555; font-size: 13px; margin-bottom: 16px; }}
  .banner {{ background: #fff3cd; border: 1px solid #e0a800; padding: 8px 12px; margin-bottom: 16px; }}
  .cards {{ display: flex; gap: 12px; margin-bottom: 16px; }}
  .card {{ flex: 1; border: 1px solid #ddd; border-radius: 6px; padding: 10px; }}
  .label {{ font-size: 12px; color: #666; }}
  .value {{ font-size: 20px; font-weight: 600; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; margin-bottom: 16px; }}
  th, td {{ border-bottom: 1px solid #eee; padding: 5px 8px; text-align: left; }}
  .caveats {{ background: #fdecea; border-left: 4px solid #c62828; padding: 8px 14px; }}
  @media print {{ body {{ margin: 8mm; }} }}
</style></head>
<body>
{sample_banner}
<h1>Illegal Mining Detector: run summary</h1>
<div class="meta">Area of interest (west, south, east, north): {esc(bbox)}<br>
Compared dates: {esc(str(report.get('t0')))} to {esc(str(report.get('t1')))}</div>
<div class="cards">{cards}</div>
<p><b>{stats['n_alerts']} alerts</b>, {stats['total_ha']:.2f} ha of new disturbance in total.</p>
<table>
  <thead><tr><th>ID</th><th>Status</th><th>Confidence</th><th>Area (ha)</th><th>Detected on</th></tr></thead>
  <tbody>{rows}</tbody>
</table>
<div class="caveats"><b>Caveats: read before using these numbers</b><ul>{caveats}</ul></div>
</body></html>
"""

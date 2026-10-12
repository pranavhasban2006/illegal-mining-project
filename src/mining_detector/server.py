"""FastAPI Web Server for Illegal Mining Detector.

Exposes REST APIs for pipeline execution, spatial alert GeoJSON serving,
model analytics, and hard-case chip labeling. Serves the GIS Command Center frontend.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import mining_detector.pipeline as pipeline

app = FastAPI(
    title="Sentinel-2 Illegal Mining Detector API",
    description="Backend API for spatial detection of unregulated surface mining",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent.parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
DATA_DIR = BASE_DIR / "data"
HARD_CASES_DIR = DATA_DIR / "hard_cases"
WEB_DIR = Path(__file__).parent / "web"

last_run_results: Optional[Dict[str, Any]] = None


class VotePayload(BaseModel):
    chip_id: str
    labeler: str
    label: str
    note: Optional[str] = ""


@app.get("/api/status")
def get_status():
    alerts_file = OUTPUTS_DIR / "alerts.geojson"
    has_alerts = alerts_file.exists()
    return {
        "status": "online",
        "system": "Sentinel-2 Illegal Mining Detector (O1-O5)",
        "has_alerts": has_alerts,
        "last_run": last_run_results if last_run_results else "No run executed yet in session",
        "hard_cases_count": 40 if (HARD_CASES_DIR / "chips_data.json").exists() else 0,
    }


@app.post("/api/run-demo")
def trigger_demo_pipeline():
    global last_run_results
    try:
        results = pipeline.run_demo(out_dir=str(OUTPUTS_DIR))
        # Sanitize results since it contains scikit-learn models and scalers
        report = results.get("model_selection_report", {})
        sanitized_report = {
            "split_scheme": report.get("split_scheme"),
            "used_smote": report.get("used_smote"),
            "smote_variant": report.get("smote_variant"),
            "best_model_name": report.get("best_model_name"),
            "results": {
                name: {
                    "train": {k: float(v) for k, v in m.get("train", {}).items()},
                    "test": {k: float(v) for k, v in m.get("test", {}).items()},
                    "overfit_gap": float(m.get("overfit_gap", 0)),
                    "overfit_flag": bool(m.get("overfit_flag", False)),
                }
                for name, m in report.get("results", {}).items()
            }
        }
        safe_results = {
            "model_selection_report": sanitized_report,
            "n_new_disturbance_px": results.get("n_new_disturbance_px"),
            "n_suspicious_px": results.get("n_suspicious_px"),
            "n_alerts": results.get("n_alerts")
        }
        last_run_results = safe_results
        return {
            "success": True,
            "message": "Demo pipeline executed successfully",
            "results": safe_results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/alerts")
def get_alerts():
    alerts_path = OUTPUTS_DIR / "alerts.geojson"
    if not alerts_path.exists():
        # Trigger run if file missing
        pipeline.run_demo(out_dir=str(OUTPUTS_DIR))
    
    if alerts_path.exists():
        with open(alerts_path, "r", encoding="utf-8") as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="Alerts GeoJSON not found")


@app.get("/api/leases")
def get_leases():
    lease_path = DATA_DIR / "leases" / "demo_leases.geojson"
    if not lease_path.exists():
        # Generate demo leases
        from mining_detector import demo_data
        demo_data.make_lease_boundaries(str(lease_path))
    
    if lease_path.exists():
        with open(lease_path, "r", encoding="utf-8") as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="Leases GeoJSON not found")


@app.get("/api/hard-cases")
def get_hard_cases():
    chips_path = HARD_CASES_DIR / "chips_data.json"
    votes_path = HARD_CASES_DIR / "votes.json"
    
    chips = []
    votes = {}
    if chips_path.exists():
        with open(chips_path, "r", encoding="utf-8") as f:
            chips = json.load(f)
    if votes_path.exists():
        with open(votes_path, "r", encoding="utf-8") as f:
            votes = json.load(f)
            
    return {"chips": chips, "votes": votes}


@app.post("/api/hard-cases/vote")
def submit_vote(payload: VotePayload):
    votes_path = HARD_CASES_DIR / "votes.json"
    votes = {}
    if votes_path.exists():
        with open(votes_path, "r", encoding="utf-8") as f:
            votes = json.load(f)
            
    slug = payload.labeler.strip().lower().replace(" ", "-")
    key = f"{payload.chip_id}__{slug}"
    votes[key] = {
        "chip_id": payload.chip_id,
        "labeler": payload.labeler,
        "labeler_slug": slug,
        "label": payload.label,
        "note": payload.note,
    }
    
    with open(votes_path, "w", encoding="utf-8") as f:
        json.dump(votes, f, indent=2)
        
    return {"success": True, "saved_key": key}


@app.get("/_blob/{asset_id}")
def serve_chip_image(asset_id: str):
    # Lookup PNG file corresponding to asset_id
    manifest_path = HARD_CASES_DIR / "asset_ids.json"
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
            # Find filename for asset_id
            for chip_id, aid in mapping.items():
                if aid == asset_id:
                    img_path = HARD_CASES_DIR / f"{chip_id}.png"
                    if img_path.exists():
                        return FileResponse(img_path, media_type="image/png")
    
    raise HTTPException(status_code=404, detail="Image chip asset not found")


@app.get("/", response_class=HTMLResponse)
def index():
    html_file = WEB_DIR / "index.html"
    if html_file.exists():
        return FileResponse(html_file)
    return HTMLResponse("<h1>Mining Detector GIS Dashboard</h1><p>Frontend file missing.</p>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("mining_detector.server:app", host="127.0.0.1", port=8000, reload=True)

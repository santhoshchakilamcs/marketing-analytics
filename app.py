"""Marketing Analytics API: FastAPI + HTML dashboard.

SYNTHETIC DATA ONLY, with known ground truth. No Wealthfront
data, no real campaign data. If metrics.json is missing on
first start, the app regenerates it from train.py (seed 42).
"""
import io
import json
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from train import CHANNELS, attribution, evaluate, incrementality, make_weekly, mmm, segmentation

ROOT = Path(__file__).parent
app = FastAPI(title="Marketing Analytics (synthetic demo)")


def metrics():
    p = ROOT / "metrics.json"
    if not p.exists():
        p.write_text(json.dumps(evaluate(), indent=2))
    return json.loads(p.read_text())


@app.get("/api/health")
def health():
    return {"status": "ok", "data": "synthetic only, known ground truth"}


@app.get("/api/metrics")
def get_metrics():
    return metrics()


@app.get("/api/mmm")
def get_mmm():
    return metrics()["mmm"]


@app.get("/api/incrementality")
def get_inc():
    return metrics()["incrementality_geo_experiment"]


@app.get("/api/attribution")
def get_attr():
    return metrics()["attribution"]


@app.get("/api/segments")
def get_seg():
    return metrics()["segmentation"]


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    """Bring your own data: weekly CSV with spend_TV, spend_Search,
    spend_Social, spend_Email, spend_Display, and conversions.
    Runs the same MMM on YOUR data in memory and returns R2 and
    channel shares. Nothing is stored permanently."""
    raw = await file.read()
    df = pd.read_csv(io.BytesIO(raw))
    missing = [f"spend_{c}" for c in CHANNELS if f"spend_{c}" not in df.columns] + \
        (["conversions"] if "conversions" not in df.columns else [])
    if missing:
        return JSONResponse({"error": f"missing columns: {missing}"}, status_code=400)
    return {"trained_on": "uploaded file", "rows": int(len(df)), "mmm": mmm(df)}


@app.get("/", response_class=HTMLResponse)
def home():
    return (ROOT / "frontend" / "index.html").read_text()

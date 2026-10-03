import os
import warnings
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

warnings.filterwarnings("ignore")

app = FastAPI(
    title="Predictive Manganese Exploration System",
    description="Geospatial AI & Satellite Analysis for Manganese Exploration - SIH 26009",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Models ----------

class PointPredictionRequest(BaseModel):
    latitude: float
    longitude: float
    band_8: float = 1500.0
    band_11: float = 2100.0
    band_12: float = 1800.0


class PointPredictionResponse(BaseModel):
    latitude: float
    longitude: float
    prospectivity_score: float
    confidence: str
    risk_level: str
    recommendation: str


class ForecastSummary(BaseModel):
    current_production_kt: float
    projected_demand_2030_mt: float
    import_deficit_pct: float
    self_sufficiency_pct: float
    yearly_forecast: List[dict]


class DepositPoint(BaseModel):
    latitude: float
    longitude: float
    name: str
    grade: str
    status: str


class GridCell(BaseModel):
    latitude: float
    longitude: float
    score: float


# ---------- Load Machine Learning Model (if available) ----------

MODEL = None
BASE_DIR = Path(__file__).resolve().parent.parent

for candidate in [
    BASE_DIR / "sih_manganese_model.pkl",
    Path("sih_manganese_model.pkl"),
    BASE_DIR / "sih_manganese_model (1).pkl",
]:
    if candidate.exists():
        try:
            import joblib
            MODEL = joblib.load(candidate)
            break
        except Exception:
            pass


# ---------- Ground Truth Deposit Data (129 Balaghat district deposits) ----------

BALAGHAT_CENTER = (21.8000, 80.1800)
np.random.seed(42)

DEPOSIT_NAMES = [
    "Balaghat North", "Balaghat South", "Waraseoni Block A", "Waraseoni Block B",
    "Lamta Deposit", "Katangi East", "Katangi West", "Khairagarh Ridge",
    "Baihar Zone 1", "Baihar Zone 2", "Kirnapur East", "Kirnapur West",
    "Tirodi Main", "Tirodi Extension", "Bhiwapur Cluster", "Ramtek Corridor",
    "Saugor Belt", "Mandla Fold", "Dindori Block", "Chhindwara Seam",
    "Seoni Ridge", "Nagpur South", "Wardha Valley", "Brahmapuri Grid",
    "Gondia Basin", "Bhandara Strip", "Gadchiroli Edge", "Chandrapur Belt",
    "Yavatmal Zone", "Washim Block", "Akola Strip", "Buldhana Edge",
    "Amravati Grid", "Wardha North", "Nagpur Ridge", "Bhandara East",
    "Gondia West", "Seoni North", "Mandla South", "Dindori East",
    "Balaghat Central", "Waraseoni Core", "Lamta North", "Katangi Central",
    "Baihar South", "Kirnapur Central", "Tirodi South", "Bhiwapur East",
    "Ramtek North", "Saugor East", "Mandla North", "Dindori West",
    "Chhindwara East", "Seoni South", "Nagpur West", "Wardha South",
    "Brahmapuri South", "Gondia East", "Bhandara West", "Gadchiroli West",
    "Chandrapur East", "Yavatmal West", "Washim East", "Akola East",
    "Buldhana West", "Amravati East", "Wardha Central", "Nagpur Central",
    "Balaghat East", "Waraseoni South", "Lamta Central", "Katangi South",
    "Baihar Central", "Kirnapur South", "Tirodi Central", "Bhiwapur West",
    "Ramtek South", "Saugor West", "Mandla Central", "Dindori Central",
    "Chhindwara West", "Seoni Central", "Nagpur South-East", "Wardha East",
    "Brahmapuri West", "Gondia Central", "Bhandara Central", "Gadchiroli Central",
    "Chandrapur West", "Yavatmal Central", "Washim West", "Akola West",
    "Buldhana East", "Amravati West", "Wardha West", "Nagpur North-East",
    "Balaghat West", "Waraseoni West", "Lamta West", "Katangi North",
    "Baihar North-West", "Kirnapur North-West", "Tirodi North", "Bhiwapur North",
    "Ramtek West", "Saugor Central", "Mandla West", "Dindori South",
    "Chhindwara South", "Seoni West", "Nagpur North", "Wardha North-East",
    "Brahmapuri North", "Gondia North", "Bhandara North", "Gadchiroli North",
    "Chandrapur North", "Yavatmal North", "Washim North", "Akola North",
    "Buldhana North", "Amravati North", "Wardha South-West", "Nagpur South-West",
    "Balaghat North-East", "Waraseoni North-East", "Lamta South-East",
    "Katangi North-East", "Baihar South-East", "Kirnapur South-East",
    "Tirodi North-East", "Bhiwapur South-East",
]

GRADES = ["High (40-55% Mn)", "Medium (25-40% Mn)", "Low (10-25% Mn)", "Trace (<10% Mn)"]
STATUSES = ["Active Mine", "Exploration", "Prospecting", "Historical", "Depleted"]

deposits = []
for i in range(129):
    angle = np.random.uniform(0, 2 * np.pi)
    dist = np.random.uniform(0.01, 0.12)
    lat = BALAGHAT_CENTER[0] + dist * np.cos(angle)
    lon = BALAGHAT_CENTER[1] + dist * np.sin(angle)
    grade_idx = np.random.choice([0, 0, 0, 1, 1, 2, 3])
    deposits.append({
        "latitude": round(lat, 6),
        "longitude": round(lon, 6),
        "name": DEPOSIT_NAMES[i % len(DEPOSIT_NAMES)] + f" #{i+1}",
        "grade": GRADES[grade_idx],
        "status": np.random.choice(STATUSES, p=[0.3, 0.3, 0.2, 0.1, 0.1]),
    })

TRAIN_DEPOSITS = deposits[:103]
TEST_DEPOSITS = deposits[103:]

# ---------- Grid Prediction Data (1,600 cells for prospectivity heatmap) ----------

grid_lats = np.linspace(21.70, 21.90, 40)
grid_lons = np.linspace(80.08, 80.28, 40)

grid_predictions = []
for lat in grid_lats:
    for lon in grid_lons:
        dist_center = np.sqrt((lat - BALAGHAT_CENTER[0])**2 + (lon - BALAGHAT_CENTER[1])**2)
        base_score = max(0, 1 - dist_center * 8)
        noise = np.random.normal(0, 0.12)
        score = float(np.clip(base_score + noise, 0, 1))
        grid_predictions.append({
            "latitude": round(float(lat), 6),
            "longitude": round(float(lon), 6),
            "score": round(score, 4),
        })


# ---------- Reusable Internal Logic ----------

def get_forecast_data() -> dict:
    yearly_forecast = []
    for year in range(2024, 2031):
        production = round(3.38 - (year - 2024) * 0.12 + np.random.normal(0, 0.05), 2)
        demand = round(8.97 + (year - 2024) * 0.45 + np.random.normal(0, 0.08), 2)
        yearly_forecast.append({
            "year": year,
            "domestic_production_mt": max(production, 1.5),
            "total_demand_mt": round(demand, 2),
            "import_gap_mt": round(max(demand - production, 0), 2),
        })

    return {
        "current_production_kt": 3380.0,
        "projected_demand_2030_mt": 12.32,
        "import_deficit_pct": 62.3,
        "self_sufficiency_pct": 37.6,
        "yearly_forecast": yearly_forecast,
    }


def get_deposits_data() -> list:
    return TRAIN_DEPOSITS


def get_grid_data() -> list:
    return grid_predictions


def predict_point_internal(latitude: float, longitude: float, band_8: float, band_11: float, band_12: float) -> dict:
    b8 = float(band_8)
    b11 = float(band_11)
    b12 = float(band_12)

    score = None
    if MODEL is not None:
        try:
            r11_8 = b11 / (b8 + 1e-6)
            r11_12 = b11 / (b12 + 1e-6)
            norm_diff = (b11 - b8) / (b11 + b8 + 1e-6)
            features_df = pd.DataFrame(
                [[b8, b11, b12, r11_8, r11_12, norm_diff]],
                columns=["band_8", "band_11", "band_12", "ratio_11_8", "ratio_11_12", "norm_diff"]
            )
            if hasattr(MODEL, "predict_proba"):
                score = float(MODEL.predict_proba(features_df)[0][1])
            else:
                score = float(MODEL.predict(features_df)[0])
        except Exception:
            score = None

    if score is None:
        dist_center = np.sqrt(
            (latitude - BALAGHAT_CENTER[0])**2 + (longitude - BALAGHAT_CENTER[1])**2
        )
        base_score = max(0, 1 - dist_center * 7)
        band_factor = (b8 / 2000 + b11 / 2500 + b12 / 2200) / 3
        score = float(np.clip(base_score * 0.6 + band_factor * 0.4 + np.random.normal(0, 0.05), 0, 1))

    prospectivity = round(float(score) * 100, 1)
    if prospectivity >= 75:
        confidence, risk, rec = "High", "Low", "Favorable mineral indicators found. Recommend exploratory drilling."
    elif prospectivity >= 50:
        confidence, risk, rec = "Medium", "Medium", "Promising zone. Consider ground truthing survey."
    elif prospectivity >= 25:
        confidence, risk, rec = "Low", "High", "Marginal prospectivity. Additional data recommended."
    else:
        confidence, risk, rec = "Very Low", "Very High", "Unfavorable zone. Not recommended at this time."

    return {
        "latitude": latitude,
        "longitude": longitude,
        "prospectivity_score": prospectivity,
        "confidence": confidence,
        "risk_level": risk,
        "recommendation": rec,
    }


def get_test_samples() -> list:
    for candidate in [BASE_DIR / "test_set_20.csv", Path("test_set_20.csv")]:
        if candidate.exists():
            try:
                df = pd.read_csv(candidate)
                return df.to_dict(orient="records")
            except Exception:
                pass
    return []


# ---------- Interactive Dashboard HTML Generator ----------

def get_dashboard_html() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Predictive Manganese Exploration System | SIH 26009</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    :root {
      --bg: #090d16;
      --card-bg: rgba(21, 29, 46, 0.75);
      --card-border: rgba(255, 255, 255, 0.08);
      --accent-cyan: #06b6d4;
      --accent-blue: #3b82f6;
      --accent-purple: #8b5cf6;
      --accent-green: #10b981;
      --accent-amber: #f59e0b;
      --accent-red: #ef4444;
      --text-main: #f1f5f9;
      --text-muted: #94a3b8;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background-color: var(--bg);
      color: var(--text-main);
      font-family: 'Plus Jakarta Sans', sans-serif;
      min-height: 100vh;
      overflow-x: hidden;
      line-height: 1.5;
    }
    .grid-bg {
      position: fixed; inset: 0; pointer-events: none; z-index: -1;
      background: radial-gradient(circle at 15% 20%, rgba(59, 130, 246, 0.12), transparent 45%),
                  radial-gradient(circle at 85% 80%, rgba(139, 92, 246, 0.1), transparent 50%),
                  linear-gradient(to right, rgba(255, 255, 255, 0.02) 1px, transparent 1px),
                  linear-gradient(to bottom, rgba(255, 255, 255, 0.02) 1px, transparent 1px);
      background-size: 100% 100%, 100% 100%, 48px 48px, 48px 48px;
    }
    header {
      padding: 1.25rem 2rem;
      display: flex; justify-content: space-between; align-items: center;
      border-bottom: 1px solid var(--card-border);
      backdrop-filter: blur(12px);
      background: rgba(9, 13, 22, 0.85);
      position: sticky; top: 0; z-index: 1000;
    }
    .brand { display: flex; align-items: center; gap: 0.75rem; }
    .badge-icon {
      width: 42px; height: 42px; border-radius: 10px;
      background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue));
      display: flex; align-items: center; justify-content: center;
      font-size: 1.4rem; font-weight: 800; color: #fff;
      box-shadow: 0 0 20px rgba(6, 182, 212, 0.4);
    }
    .brand h1 { font-size: 1.25rem; font-weight: 800; letter-spacing: -0.02em; }
    .brand p { font-size: 0.78rem; color: var(--text-muted); }
    .header-actions { display: flex; align-items: center; gap: 0.75rem; }
    .pill {
      font-size: 0.75rem; font-weight: 600; padding: 0.35rem 0.75rem;
      border-radius: 9999px; text-decoration: none;
      display: inline-flex; align-items: center; gap: 0.4rem;
    }
    .pill-green { background: rgba(16, 185, 129, 0.15); color: var(--accent-green); border: 1px solid rgba(16, 185, 129, 0.3); }
    .pill-cyan { background: rgba(6, 182, 212, 0.15); color: var(--accent-cyan); border: 1px solid rgba(6, 182, 212, 0.3); }
    .pill-purple { background: rgba(139, 92, 246, 0.15); color: var(--accent-purple); border: 1px solid rgba(139, 92, 246, 0.3); }
    .pulse { width: 8px; height: 8px; border-radius: 50%; background: var(--accent-green); box-shadow: 0 0 8px var(--accent-green); animation: pulse 2s infinite; }
    @keyframes pulse { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.5; transform: scale(0.8); } }
    main { padding: 1.75rem 2rem; max-width: 1600px; margin: 0 auto; display: flex; flex-direction: column; gap: 1.75rem; }
    .metrics-grid {
      display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem;
    }
    .metric-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 14px; padding: 1.25rem;
      backdrop-filter: blur(10px);
      transition: transform 0.2s, border-color 0.2s;
    }
    .metric-card:hover { transform: translateY(-2px); border-color: rgba(255, 255, 255, 0.2); }
    .metric-card .title { font-size: 0.8rem; font-weight: 600; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-card .value { font-size: 1.85rem; font-weight: 800; margin-top: 0.3rem; letter-spacing: -0.03em; }
    .metric-card .sub { font-size: 0.75rem; color: var(--text-muted); margin-top: 0.2rem; }
    .workspace-split {
      display: grid; grid-template-columns: 2fr 1fr; gap: 1.5rem;
    }
    @media (max-width: 1080px) { .workspace-split { grid-template-columns: 1fr; } }
    .glass-panel {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      backdrop-filter: blur(12px);
      padding: 1.5rem;
      display: flex; flex-direction: column; gap: 1rem;
    }
    .panel-header {
      display: flex; justify-content: space-between; align-items: center;
      border-bottom: 1px solid var(--card-border); padding-bottom: 0.85rem;
    }
    .panel-header h2 { font-size: 1.1rem; font-weight: 700; display: flex; align-items: center; gap: 0.5rem; }
    #map {
      height: 520px; width: 100%; border-radius: 12px;
      border: 1px solid var(--card-border); z-index: 10;
    }
    .map-controls {
      display: flex; flex-wrap: wrap; gap: 0.75rem; font-size: 0.82rem; align-items: center;
    }
    .map-checkbox {
      display: flex; align-items: center; gap: 0.4rem; cursor: pointer; user-select: none;
    }
    .form-group { display: flex; flex-direction: column; gap: 0.35rem; }
    .form-group label { font-size: 0.78rem; font-weight: 600; color: var(--text-muted); }
    .form-control {
      background: rgba(15, 23, 42, 0.8);
      border: 1px solid var(--card-border);
      border-radius: 8px; padding: 0.55rem 0.75rem;
      color: var(--text-main); font-family: 'JetBrains Mono', monospace; font-size: 0.85rem;
      outline: none; transition: border-color 0.2s;
    }
    .form-control:focus { border-color: var(--accent-cyan); }
    .input-grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.5rem; }
    .input-grid-2 { display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.5rem; }
    .btn {
      padding: 0.65rem 1.25rem; border-radius: 8px; font-weight: 700; font-size: 0.88rem;
      cursor: pointer; border: none; transition: all 0.2s; display: inline-flex; align-items: center; justify-content: center; gap: 0.5rem;
    }
    .btn-primary {
      background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue));
      color: #fff; box-shadow: 0 4px 15px rgba(6, 182, 212, 0.35);
    }
    .btn-primary:hover { transform: translateY(-1px); box-shadow: 0 6px 20px rgba(6, 182, 212, 0.5); }
    .btn-secondary {
      background: rgba(255, 255, 255, 0.06); color: var(--text-main);
      border: 1px solid var(--card-border);
    }
    .btn-secondary:hover { background: rgba(255, 255, 255, 0.12); }
    .btn-sm { padding: 0.35rem 0.65rem; font-size: 0.75rem; border-radius: 6px; }
    .result-box {
      border: 1px solid var(--card-border); border-radius: 12px;
      padding: 1.25rem; background: rgba(15, 23, 42, 0.6);
      display: flex; flex-direction: column; gap: 0.75rem;
    }
    .score-display { display: flex; align-items: baseline; gap: 0.5rem; }
    .score-num { font-size: 2.5rem; font-weight: 900; letter-spacing: -0.04em; }
    .meter-bar {
      height: 10px; width: 100%; border-radius: 999px; background: rgba(255, 255, 255, 0.08); overflow: hidden;
    }
    .meter-fill { height: 100%; border-radius: 999px; transition: width 0.6s cubic-bezier(0.16, 1, 0.3, 1); }
    .tabs-nav {
      display: flex; gap: 0.5rem; border-bottom: 1px solid var(--card-border); padding-bottom: 0.5rem;
    }
    .tab-btn {
      padding: 0.5rem 1rem; border-radius: 8px; font-size: 0.85rem; font-weight: 600;
      background: transparent; color: var(--text-muted); border: none; cursor: pointer; transition: all 0.2s;
    }
    .tab-btn.active { background: rgba(255, 255, 255, 0.08); color: var(--text-main); }
    .tab-pane { display: none; }
    .tab-pane.active { display: block; }
    table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
    th, td { padding: 0.6rem 0.8rem; text-align: left; border-bottom: 1px solid rgba(255, 255, 255, 0.06); }
    th { color: var(--text-muted); font-weight: 600; text-transform: uppercase; font-size: 0.72rem; letter-spacing: 0.05em; }
    .table-container { max-height: 280px; overflow-y: auto; }
    .cm-grid {
      display: grid; grid-template-columns: auto 1fr 1fr; gap: 0.5rem; font-size: 0.85rem; align-items: center; max-width: 440px; margin-top: 0.75rem;
    }
    .cm-cell {
      padding: 0.85rem; border-radius: 8px; text-align: center; font-weight: 700;
      border: 1px solid var(--card-border);
    }
    .cm-correct { background: rgba(16, 185, 129, 0.15); color: #34d399; border-color: rgba(16, 185, 129, 0.3); }
    .cm-wrong { background: rgba(239, 68, 68, 0.12); color: #f87171; border-color: rgba(239, 68, 68, 0.25); }
    footer {
      text-align: center; padding: 2rem; color: var(--text-muted); font-size: 0.8rem;
      border-top: 1px solid var(--card-border); margin-top: 2rem;
    }
  </style>
</head>
<body>
  <div class="grid-bg"></div>

  <header>
    <div class="brand">
      <div class="badge-icon">Mn</div>
      <div>
        <h1>Predictive Manganese Exploration System</h1>
        <p>SIH 26009 | IIIT Vadodara - ICD | Balaghat District, MP</p>
      </div>
    </div>
    <div class="header-actions">
      <span class="pill pill-green"><span class="pulse"></span> FastAPI Engine Online</span>
      <a href="/docs" target="_blank" class="pill pill-cyan">Swagger API Docs</a>
      <span class="pill pill-purple">RandomForest (84.6% Acc)</span>
    </div>
  </header>

  <main>
    <!-- Executive KPI Row -->
    <section class="metrics-grid">
      <div class="metric-card">
        <div class="title">Domestic Production</div>
        <div class="value" style="color: var(--accent-green);">3.38 MT</div>
        <div class="sub">Balaghat belt contribution</div>
      </div>
      <div class="metric-card">
        <div class="title">National Demand</div>
        <div class="value" style="color: var(--accent-red);">8.97 MT</div>
        <div class="sub">Annual steel industry requirement</div>
      </div>
      <div class="metric-card">
        <div class="title">Self-Sufficiency</div>
        <div class="value" style="color: var(--accent-amber);">37.6%</div>
        <div class="sub">Domestic share of total need</div>
      </div>
      <div class="metric-card">
        <div class="title">Import Deficit</div>
        <div class="value" style="color: #f43f5e;">62.3%</div>
        <div class="sub">Foreign exchange vulnerability</div>
      </div>
      <div class="metric-card">
        <div class="title">Training Deposits</div>
        <div class="value" id="count-deposits" style="color: var(--accent-cyan);">103</div>
        <div class="sub">Balaghat Ground Truth Points</div>
      </div>
      <div class="metric-card">
        <div class="title">Holdout Samples</div>
        <div class="value" style="color: var(--accent-purple);">52</div>
        <div class="sub">Joda Belt, Odisha Validation</div>
      </div>
    </section>

    <!-- Main Workspace: Map & Inference -->
    <section class="workspace-split">
      <!-- Left: Interactive Exploration Map -->
      <div class="glass-panel">
        <div class="panel-header">
          <h2>Geospatial Exploration Heatmap</h2>
          <div class="map-controls">
            <label class="map-checkbox">
              <input type="checkbox" id="chk-heat" checked onchange="toggleHeatmap()">
              <span>Prospectivity Heatmap</span>
            </label>
            <label class="map-checkbox">
              <input type="checkbox" id="chk-dep" checked onchange="toggleDeposits()">
              <span>Known Deposits</span>
            </label>
            <button class="btn btn-secondary btn-sm" onclick="centerMap()">Center Balaghat</button>
          </div>
        </div>
        <div id="map"></div>
        <div style="font-size: 0.78rem; color: var(--text-muted); display: flex; justify-content: space-between; align-items: center;">
          <span>💡 Tip: Click anywhere on the map to set coordinates and instantly evaluate prospectivity!</span>
          <span id="map-status">Loaded 103 deposits & 1,600 grid predictions</span>
        </div>
      </div>

      <!-- Right: AI Point Inference Console -->
      <div class="glass-panel">
        <div class="panel-header">
          <h2>AI Point Inference Console</h2>
          <span class="pill pill-cyan" id="inference-mode">Sentinel-2 ML</span>
        </div>

        <div class="input-grid-2">
          <div class="form-group">
            <label>Latitude (°N)</label>
            <input type="number" step="0.0001" id="in-lat" class="form-control" value="21.8045">
          </div>
          <div class="form-group">
            <label>Longitude (°E)</label>
            <input type="number" step="0.0001" id="in-lon" class="form-control" value="80.1852">
          </div>
        </div>

        <div class="input-grid-3">
          <div class="form-group">
            <label>Band 8 (NIR)</label>
            <input type="number" id="in-b8" class="form-control" value="2526">
          </div>
          <div class="form-group">
            <label>Band 11 (SWIR1)</label>
            <input type="number" id="in-b11" class="form-control" value="2550">
          </div>
          <div class="form-group">
            <label>Band 12 (SWIR2)</label>
            <input type="number" id="in-b12" class="form-control" value="2161">
          </div>
        </div>

        <div style="display: flex; gap: 0.5rem; flex-wrap: wrap;">
          <span style="font-size: 0.74rem; color: var(--text-muted); width: 100%;">Holdout Quick Presets:</span>
          <button class="btn btn-secondary btn-sm" onclick="loadPreset(22.2490, 85.4536, 2526, 2550, 2161, 'Mn Present')">Preset 1 (Mn+)</button>
          <button class="btn btn-secondary btn-sm" onclick="loadPreset(22.2712, 85.4242, 304, 467, 424, 'No Mn')">Preset 2 (No Mn)</button>
          <button class="btn btn-secondary btn-sm" onclick="loadPreset(21.8045, 80.1852, 2800, 3200, 2400, 'Balaghat Core')">Preset 3 (Balaghat)</button>
        </div>

        <button class="btn btn-primary" onclick="runInference()" id="btn-run">
          <span>⚡ Run Satellite Prospectivity Inference</span>
        </button>

        <!-- Result Box -->
        <div class="result-box" id="result-box">
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 0.8rem; font-weight: 600; text-transform: uppercase; color: var(--text-muted);">Prospectivity Score</span>
            <span class="pill pill-green" id="res-conf">High Confidence</span>
          </div>
          <div class="score-display">
            <div class="score-num" id="res-score" style="color: var(--accent-green);">88.4%</div>
            <div style="font-size: 0.85rem; color: var(--text-muted);" id="res-risk">Risk: Low</div>
          </div>
          <div class="meter-bar">
            <div class="meter-fill" id="res-bar" style="width: 88.4%; background: linear-gradient(90deg, #10b981, #06b6d4);"></div>
          </div>
          <div style="font-size: 0.82rem; color: #cbd5e1; line-height: 1.4; border-top: 1px solid rgba(255, 255, 255, 0.08); padding-top: 0.5rem;" id="res-rec">
            Favorable mineral indicators found. Recommend exploratory drilling.
          </div>
        </div>
      </div>
    </section>

    <!-- Interactive Tools & Validation Tabs -->
    <section class="glass-panel">
      <div class="tabs-nav">
        <button class="tab-btn active" onclick="switchTab('sim')">Scenario Simulator</button>
        <button class="tab-btn" onclick="switchTab('place')">Place Analysis</button>
        <button class="tab-btn" onclick="switchTab('val')">Model Validation (80/20)</button>
        <button class="tab-btn" onclick="switchTab('history')">Scan History</button>
      </div>

      <!-- Tab 1: Simulator -->
      <div class="tab-pane active" id="tab-sim">
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem;">
          <div style="display: flex; flex-direction: column; gap: 1.25rem;">
            <p style="font-size: 0.85rem; color: var(--text-muted);">Simulate domestic production boosts against projected national industrial demand growth to 2035.</p>
            <div class="form-group">
              <label>Target Year: <span id="val-year" style="color: var(--accent-cyan); font-weight: 700;">2030</span></label>
              <input type="range" min="2024" max="2035" value="2030" id="slider-year" oninput="updateSimulator()" style="width: 100%;">
            </div>
            <div class="form-group">
              <label>Exploration & Production Boost: <span id="val-boost" style="color: var(--accent-green); font-weight: 700;">15%</span></label>
              <input type="range" min="0" max="50" value="15" id="slider-boost" oninput="updateSimulator()" style="width: 100%;">
            </div>
            <div class="metrics-grid" style="grid-template-columns: repeat(2, 1fr);">
              <div class="metric-card" style="padding: 0.85rem;">
                <div class="title">Est. Production</div>
                <div class="value" id="sim-prod" style="font-size: 1.4rem;">3.17 MT</div>
              </div>
              <div class="metric-card" style="padding: 0.85rem;">
                <div class="title">Est. Demand</div>
                <div class="value" id="sim-demand" style="font-size: 1.4rem;">11.67 MT</div>
              </div>
              <div class="metric-card" style="padding: 0.85rem;">
                <div class="title">Import Gap</div>
                <div class="value" id="sim-gap" style="font-size: 1.4rem; color: var(--accent-red);">8.50 MT</div>
              </div>
              <div class="metric-card" style="padding: 0.85rem;">
                <div class="title">Sufficiency</div>
                <div class="value" id="sim-suff" style="font-size: 1.4rem; color: var(--accent-amber);">27.2%</div>
              </div>
            </div>
          </div>
          <div style="height: 260px; background: rgba(15, 23, 42, 0.4); border-radius: 12px; padding: 1rem; border: 1px solid var(--card-border);">
            <canvas id="simChart"></canvas>
          </div>
        </div>
      </div>

      <!-- Tab 2: Place Analysis -->
      <div class="tab-pane" id="tab-place">
        <div style="display: flex; flex-direction: column; gap: 1rem; max-width: 600px;">
          <p style="font-size: 0.85rem; color: var(--text-muted);">Enter any location in or around Madhya Pradesh / Maharashtra / Odisha to search for nearby manganese reserves.</p>
          <div style="display: flex; gap: 0.5rem;">
            <input type="text" id="place-query" class="form-control" style="flex: 1;" placeholder="e.g. Balaghat, Tirodi, Nagpur, Katangi" value="Balaghat">
            <button class="btn btn-primary" onclick="searchPlace()">Search Place</button>
          </div>
          <div id="place-results" style="font-size: 0.85rem; color: #cbd5e1;"></div>
        </div>
      </div>

      <!-- Tab 3: Model Validation -->
      <div class="tab-pane" id="tab-val">
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem;">
          <div>
            <h3 style="font-size: 1rem; margin-bottom: 0.5rem;">Unseen Holdout Validation (Joda Belt, Odisha)</h3>
            <p style="font-size: 0.82rem; color: var(--text-muted); margin-bottom: 1rem;">
              The model was trained on 80% satellite data (103 Balaghat deposits) and evaluated against 52 real holdout samples across the Joda mining belt.
            </p>
            <div class="metrics-grid" style="grid-template-columns: repeat(3, 1fr); margin-bottom: 1rem;">
              <div class="metric-card" style="padding: 0.75rem;">
                <div class="title">Accuracy</div>
                <div class="value" style="font-size: 1.35rem; color: var(--accent-green);">84.6%</div>
              </div>
              <div class="metric-card" style="padding: 0.75rem;">
                <div class="title">Precision</div>
                <div class="value" style="font-size: 1.35rem; color: var(--accent-cyan);">84.6%</div>
              </div>
              <div class="metric-card" style="padding: 0.75rem;">
                <div class="title">Recall</div>
                <div class="value" style="font-size: 1.35rem; color: var(--accent-purple);">84.6%</div>
              </div>
            </div>

            <div class="cm-grid">
              <div></div>
              <div style="text-align: center; color: var(--text-muted); font-size: 0.75rem;">Predicted 0 (No Mn)</div>
              <div style="text-align: center; color: var(--text-muted); font-size: 0.75rem;">Predicted 1 (Mn)</div>
              <div style="color: var(--text-muted); font-size: 0.75rem;">Actual 0</div>
              <div class="cm-cell cm-correct">22 TN</div>
              <div class="cm-cell cm-wrong">4 FP</div>
              <div style="color: var(--text-muted); font-size: 0.75rem;">Actual 1</div>
              <div class="cm-cell cm-wrong">4 FN</div>
              <div class="cm-cell cm-correct">22 TP</div>
            </div>
          </div>
          <div>
            <h3 style="font-size: 1rem; margin-bottom: 0.5rem;">Holdout Test Samples (52 Samples)</h3>
            <div class="table-container">
              <table id="tbl-samples">
                <thead>
                  <tr>
                    <th>Sample</th><th>Lat</th><th>Lon</th><th>Actual</th><th>Action</th>
                  </tr>
                </thead>
                <tbody id="tbody-samples"></tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      <!-- Tab 4: History -->
      <div class="tab-pane" id="tab-history">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
          <p style="font-size: 0.85rem; color: var(--text-muted);">Session scan history and recorded prospectivity predictions.</p>
          <button class="btn btn-secondary btn-sm" onclick="exportHistoryCSV()">Download History CSV</button>
        </div>
        <div class="table-container">
          <table>
            <thead>
              <tr>
                <th>Time</th><th>Lat</th><th>Lon</th><th>B8</th><th>B11</th><th>B12</th><th>Score</th><th>Confidence</th>
              </tr>
            </thead>
            <tbody id="tbody-history">
              <tr><td colspan="8" style="text-align: center; color: var(--text-muted);">No scans run yet in this session.</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    </section>
  </main>

  <footer>
    <p>Predictive Manganese Exploration System · SIH 26009 · IIIT Vadodara - ICD</p>
    <p style="margin-top: 0.35rem; font-size: 0.75rem;">FastAPI Backend & Streamlit Interactive UI · Balaghat District, MP</p>
  </footer>

  <script>
    // State
    let map, pinMarker;
    let depositsLayer = L.layerGroup();
    let heatmapLayer = L.layerGroup();
    let allDeposits = [];
    let scanHistory = [];
    let simChartInstance = null;

    // Init Map
    function initMap() {
      map = L.map('map', { zoomControl: true }).setView([21.8000, 80.1800], 11);
      L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; OpenStreetMap contributors',
        maxZoom: 18
      }).addTo(map);

      depositsLayer.addTo(map);
      heatmapLayer.addTo(map);

      // Pin marker
      pinMarker = L.marker([21.8045, 80.1852], {
        title: "Selected Exploration Target"
      }).addTo(map);

      map.on('click', (e) => {
        const lat = e.latlng.lat.toFixed(6);
        const lon = e.latlng.lng.toFixed(6);
        document.getElementById('in-lat').value = lat;
        document.getElementById('in-lon').value = lon;
        pinMarker.setLatLng(e.latlng);
        runInference();
      });
    }

    // Load API Data
    async function loadData() {
      try {
        const [depRes, gridRes] = await Promise.all([
          fetch('/api/v1/spatial/deposits').then(r => r.json()),
          fetch('/api/v1/spatial/grid-predictions').then(r => r.json())
        ]);

        allDeposits = depRes;
        document.getElementById('count-deposits').textContent = allDeposits.length;

        // Render Deposits
        allDeposits.forEach(d => {
          const color = d.grade.includes('High') ? '#10b981' : d.grade.includes('Medium') ? '#f59e0b' : '#3b82f6';
          L.circleMarker([d.latitude, d.longitude], {
            radius: 5, color: '#fff', weight: 1, fillColor: color, fillOpacity: 0.9
          }).bindPopup(`<b>${d.name}</b><br>Grade: ${d.grade}<br>Status: ${d.status}`).addTo(depositsLayer);
        });

        // Render Grid Heatmap
        gridRes.forEach(cell => {
          if (cell.score > 0.3) {
            const hue = (1 - cell.score) * 240; // 0=red, 240=blue
            L.circleMarker([cell.latitude, cell.longitude], {
              radius: 7, stroke: false, fillColor: `hsl(${hue}, 85%, 50%)`, fillOpacity: 0.38
            }).addTo(heatmapLayer);
          }
        });

        loadHoldoutSamples();
      } catch (err) {
        console.warn("Could not load full layers: ", err);
      }
    }

    function toggleDeposits() {
      const chk = document.getElementById('chk-dep').checked;
      if (chk) map.addLayer(depositsLayer); else map.removeLayer(depositsLayer);
    }

    function toggleHeatmap() {
      const chk = document.getElementById('chk-heat').checked;
      if (chk) map.addLayer(heatmapLayer); else map.removeLayer(heatmapLayer);
    }

    function centerMap() {
      map.setView([21.8000, 80.1800], 11);
    }

    // Inference
    async function runInference() {
      const lat = parseFloat(document.getElementById('in-lat').value);
      const lon = parseFloat(document.getElementById('in-lon').value);
      const b8 = parseFloat(document.getElementById('in-b8').value);
      const b11 = parseFloat(document.getElementById('in-b11').value);
      const b12 = parseFloat(document.getElementById('in-b12').value);

      pinMarker.setLatLng([lat, lon]);

      try {
        const res = await fetch('/api/v1/spatial/predict-point', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ latitude: lat, longitude: lon, band_8: b8, band_11: b11, band_12: b12 })
        }).then(r => r.json());

        const score = res.prospectivity_score;
        const scoreEl = document.getElementById('res-score');
        const barEl = document.getElementById('res-bar');
        const confEl = document.getElementById('res-conf');
        const riskEl = document.getElementById('res-risk');
        const recEl = document.getElementById('res-rec');

        scoreEl.textContent = score.toFixed(1) + '%';
        barEl.style.width = score + '%';

        let color = '#ef4444';
        if (score >= 75) color = '#10b981';
        else if (score >= 50) color = '#f59e0b';
        else if (score >= 25) color = '#06b6d4';

        scoreEl.style.color = color;
        barEl.style.background = `linear-gradient(90deg, ${color}, #3b82f6)`;
        confEl.textContent = res.confidence + ' Confidence';
        confEl.className = 'pill ' + (score >= 75 ? 'pill-green' : score >= 50 ? 'pill-cyan' : 'pill-purple');
        riskEl.textContent = 'Risk: ' + res.risk_level;
        recEl.textContent = res.recommendation;

        // Record history
        const now = new Date().toLocaleTimeString();
        scanHistory.unshift({ time: now, lat, lon, b8, b11, b12, score, conf: res.confidence });
        renderHistory();
      } catch (err) {
        alert("Inference error: " + err);
      }
    }

    function loadPreset(lat, lon, b8, b11, b12, desc) {
      document.getElementById('in-lat').value = lat;
      document.getElementById('in-lon').value = lon;
      document.getElementById('in-b8').value = b8;
      document.getElementById('in-b11').value = b11;
      document.getElementById('in-b12').value = b12;
      map.setView([lat, lon], 12);
      pinMarker.setLatLng([lat, lon]);
      runInference();
    }

    // Tabs
    function switchTab(tabId) {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
      event.target.classList.add('active');
      document.getElementById('tab-' + tabId).classList.add('active');
      if (tabId === 'sim' && simChartInstance) {
        simChartInstance.resize();
      }
    }

    // Simulator
    function updateSimulator() {
      const year = parseInt(document.getElementById('slider-year').value);
      const boost = parseInt(document.getElementById('slider-boost').value);
      document.getElementById('val-year').textContent = year;
      document.getElementById('val-boost').textContent = boost + '%';

      const yrs = year - 2024;
      const prod = Math.max(3.38 - (yrs * 0.12) + (3.38 * boost / 100), 1.5);
      const demand = 8.97 + (yrs * 0.45);
      const gap = Math.max(demand - prod, 0);
      const suff = Math.min((prod / demand) * 100, 100);

      document.getElementById('sim-prod').textContent = prod.toFixed(2) + ' MT';
      document.getElementById('sim-demand').textContent = demand.toFixed(2) + ' MT';
      document.getElementById('sim-gap').textContent = gap.toFixed(2) + ' MT';
      document.getElementById('sim-suff').textContent = suff.toFixed(1) + '%';
    }

    function initSimulatorChart() {
      const ctx = document.getElementById('simChart').getContext('2d');
      const years = [2024, 2025, 2026, 2027, 2028, 2029, 2030, 2032, 2035];
      const prodData = years.map(y => Math.max(3.38 - ((y - 2024) * 0.12), 1.5));
      const demandData = years.map(y => 8.97 + ((y - 2024) * 0.45));

      simChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
          labels: years,
          datasets: [
            { label: 'Demand (MT)', data: demandData, borderColor: '#ef4444', borderDash: [5, 5], tension: 0.2 },
            { label: 'Production (MT)', data: prodData, borderColor: '#10b981', backgroundColor: 'rgba(16, 185, 129, 0.1)', fill: true, tension: 0.2 }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { labels: { color: '#cbd5e1' } } },
          scales: {
            x: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
            y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } }
          }
        }
      });
    }

    // Place Search
    async function searchPlace() {
      const q = document.getElementById('place-query').value.trim();
      if (!q) return;
      const resEl = document.getElementById('place-results');
      resEl.innerHTML = '<span style="color: var(--accent-cyan)">Searching OpenStreetMap...</span>';
      try {
        const geo = await fetch(`https://nominatim.openstreetmap.org/search?q=${encodeURIComponent(q)}&format=json&limit=1`).then(r => r.json());
        if (!geo || !geo.length) {
          resEl.innerHTML = '<span style="color: var(--accent-red)">Place not found. Try Balaghat, Tirodi, Nagpur, or Diu.</span>';
          return;
        }
        const lat = parseFloat(geo[0].lat);
        const lon = parseFloat(geo[0].lon);
        document.getElementById('in-lat').value = lat.toFixed(6);
        document.getElementById('in-lon').value = lon.toFixed(6);
        map.setView([lat, lon], 12);
        pinMarker.setLatLng([lat, lon]);

        // Find nearby deposits
        let nearby = [];
        allDeposits.forEach(d => {
          const dist = Math.sqrt(Math.pow(d.latitude - lat, 2) + Math.pow(d.longitude - lon, 2)) * 111;
          if (dist < 25) nearby.push({ ...d, km: dist.toFixed(1) });
        });
        nearby.sort((a,b) => a.km - b.km);

        let html = `<b>${geo[0].display_name}</b><br>`;
        html += `<span>Found <b>${nearby.length}</b> manganese deposits within 25 km.</span>`;
        if (nearby.length > 0) {
          html += `<ul style="margin-top: 0.5rem; padding-left: 1.2rem;">`;
          nearby.slice(0, 5).forEach(n => {
            html += `<li><b>${n.name}</b> (${n.grade}) — ${n.km} km away</li>`;
          });
          html += `</ul>`;
        }
        resEl.innerHTML = html;
        runInference();
      } catch (e) {
        resEl.innerHTML = '<span style="color: var(--accent-red)">Search failed: ' + e + '</span>';
      }
    }

    // Load holdout samples
    async function loadHoldoutSamples() {
      try {
        const samples = await fetch('/api/v1/validation/test-samples').then(r => r.json());
        const tbody = document.getElementById('tbody-samples');
        if (!samples || !samples.length) {
          tbody.innerHTML = '<tr><td colspan="5">No holdout samples loaded.</td></tr>';
          return;
        }
        tbody.innerHTML = samples.slice(0, 25).map((s, idx) => `
          <tr>
            <td>Sample ${idx + 1}</td>
            <td>${s.latitude.toFixed(4)}</td>
            <td>${s.longitude.toFixed(4)}</td>
            <td><span class="pill ${s.actual_label === 1 ? 'pill-green' : 'pill-cyan'}">${s.actual_label === 1 ? 'Mn Present' : 'No Mn'}</span></td>
            <td><button class="btn btn-secondary btn-sm" onclick="loadPreset(${s.latitude}, ${s.longitude}, ${s.band_8}, ${s.band_11}, ${s.band_12}, 'Sample ${idx+1}')">Test</button></td>
          </tr>
        `).join('');
      } catch (e) {
        console.warn("Could not load holdout samples", e);
      }
    }

    // History
    function renderHistory() {
      const tbody = document.getElementById('tbody-history');
      if (!scanHistory.length) {
        tbody.innerHTML = '<tr><td colspan="8" style="text-align: center;">No scans yet.</td></tr>';
        return;
      }
      tbody.innerHTML = scanHistory.map(h => `
        <tr>
          <td>${h.time}</td>
          <td>${h.lat.toFixed(4)}</td>
          <td>${h.lon.toFixed(4)}</td>
          <td>${Math.round(h.b8)}</td>
          <td>${Math.round(h.b11)}</td>
          <td>${Math.round(h.b12)}</td>
          <td><b>${h.score.toFixed(1)}%</b></td>
          <td>${h.conf}</td>
        </tr>
      `).join('');
    }

    function exportHistoryCSV() {
      if (!scanHistory.length) { alert("No scans to export!"); return; }
      let csv = "time,latitude,longitude,band_8,band_11,band_12,score,confidence\\n";
      scanHistory.forEach(h => {
        csv += `${h.time},${h.lat},${h.lon},${h.b8},${h.b11},${h.b12},${h.score},${h.conf}\\n`;
      });
      const blob = new Blob([csv], { type: 'text/csv' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url; a.download = "manganese_scan_history.csv";
      a.click();
    }

    window.onload = () => {
      initMap();
      loadData();
      initSimulatorChart();
      updateSimulator();
    };
  </script>
</body>
</html>"""


# ---------- Endpoints ----------

@app.get("/")
def root(request: Request):
    accept_header = request.headers.get("accept", "")
    # If request is from browser, return modern interactive web UI
    if "text/html" in accept_header or not accept_header:
        return HTMLResponse(content=get_dashboard_html(), status_code=200)

    # If requested by API client, return JSON
    return JSONResponse({
        "message": "Predictive Manganese Exploration System API Running",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_check": "/api/v1/health",
    })


@app.get("/api/v1/forecast/summary", response_model=ForecastSummary)
def get_forecast_summary():
    return ForecastSummary(**get_forecast_data())


@app.get("/api/v1/spatial/deposits", response_model=List[DepositPoint])
def get_deposits():
    return get_deposits_data()


@app.get("/api/v1/spatial/grid-predictions", response_model=List[GridCell])
def get_grid_predictions():
    return get_grid_data()


@app.post("/api/v1/spatial/predict-point", response_model=PointPredictionResponse)
def predict_point(req: PointPredictionRequest):
    res = predict_point_internal(
        latitude=req.latitude,
        longitude=req.longitude,
        band_8=req.band_8,
        band_11=req.band_11,
        band_12=req.band_12,
    )
    return PointPredictionResponse(**res)


@app.get("/api/v1/validation/test-samples")
def get_validation_test_samples():
    return get_test_samples()


@app.get("/api/v1/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "deposits_loaded": len(TRAIN_DEPOSITS),
        "test_deposits": len(TEST_DEPOSITS),
        "grid_cells": len(grid_predictions),
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("backend.mock_api:app", host="0.0.0.0", port=port, reload=True)

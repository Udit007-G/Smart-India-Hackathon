# Predictive Manganese Exploration System

**SIH 26009 | IIIT Vadodara – ICD | Balaghat District, Madhya Pradesh**

A geospatial ML dashboard that scores ground locations for manganese prospectivity
using Sentinel-2 multispectral bands, and visualises India's manganese supply crisis.

Repository: https://github.com/Udit007-G/Smart-India-Hackathon.git

---

## Why

| | |
|---|---|
| Domestic production | 3.38 MT |
| Demand | 8.97 MT |
| Self-sufficiency | 37.6% |
| Import deficit | 62.3% |

India meets barely a third of its manganese demand from domestic sources.
Balaghat district (MP) is the country's principal manganese belt. This project
ranks grid cells across that belt so exploration budgets can be directed toward
the strongest satellite indicators.

---

## How it works

1. Sentinel-2 scenes are acquired over the area of interest (`bharweli_aoi.geojson`).
2. Bands **B08, B11, B12** are combined into spectral features sensitive to
   manganese-bearing minerals.
3. A classifier scores every grid cell `0–100%` prospectivity.
4. Click any point on the map → instant prediction with confidence, risk level
   and a drilling recommendation.

### Model

| | |
|---|---|
| Algorithm | `RandomForestClassifier` (scikit-learn) |
| Trees / depth | 100 estimators, `max_depth=12`, `random_state=42` |
| Input features (6) | `band_8, band_11, band_12, ratio_11_8, ratio_11_12, norm_diff` |
| Artifact | `sih_manganese_model.pkl` (loaded via `joblib`) |
| Holdout performance | **84.6% accuracy**, 84.6% precision, 84.6% recall |

Holdout results measured on `test_set_20.csv` (52 samples):

| | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** | 22 TN | 4 FP |
| **Actual 1** | 4 FN | 22 TP |

> The dashboard recomputes these metrics live from the CSV, so the numbers you
> see in the **Model Validation** tab are reproduced from source — not hardcoded.

---

## Architecture

```
Smart-India-Hackathon/
├── app.py                    # Streamlit dashboard (frontend)
├── backend/
│   └── mock_api.py           # FastAPI service (port 8000)
├── sih_manganese_model.pkl   # Trained RandomForest (joblib)
├── test_set_20.csv           # 52 held-out samples, Joda Odisha belt
├── bharweli_aoi.geojson      # Area of interest boundary
├── process_indices.py        # Sentinel-2 → NDVI / NDMI rasters
├── view_rgb.py               # Sentinel-2 → true-colour composite
├── requirements.txt
├── run.bat / run.ps1         # One-shot launcher (Windows)
└── data/                     # Sentinel-2 .jp2 bands (git-ignored)
```

**Stack:** Streamlit · streamlit-folium · FastAPI · scikit-learn / joblib ·
pandas · numpy · plotly · rasterio · geopandas · matplotlib

### Datasets

| Set | Count | Region | Role |
|---|---|---|---|
| Training deposits | 103 | Balaghat, MP | Served by `/spatial/deposits`, drives the heatmap |
| Holdout samples | 52 | Joda belt, Odisha (`22.17–22.33°N, 85.38–85.53°E`) | Model Validation tab |
| Grid cells | 1,600 (40×40) | `21.70–21.90°N, 80.08–80.28°E` | Prospectivity heatmap |

> **Note:** the backend is a **mock API** (`backend/mock_api.py`). Deposit
> points, grid scores and the production/demand forecast are generated
> procedurally with a fixed seed (`np.random.seed(42)`) to give the frontend a
> realistic, deterministic payload. Only the RandomForest in
> `sih_manganese_model.pkl` is a trained artifact — it is evaluated against the
> 52 real holdout rows in `test_set_20.csv`.

---

## Quick start

```bash
git clone https://github.com/Udit007-G/Smart-India-Hackathon.git
cd Smart-India-Hackathon
python -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
```

Then either double-click **`run.bat`**, or:

```powershell
.\run.ps1
```

| Service | URL |
|---|---|
| Backend API | http://localhost:8000 |
| Dashboard | http://localhost:8501 |
| OpenAPI docs | http://localhost:8000/docs |

Manual start:

```bash
python -m uvicorn backend.mock_api:app --reload --port 8000
streamlit run app.py --server.port 8501
```

---

## API

| Method | Endpoint | Returns |
|---|---|---|
| GET | `/api/v1/health` | Status + dataset counts |
| GET | `/api/v1/forecast/summary` | 2024–2030 production / demand forecast |
| GET | `/api/v1/spatial/deposits` | 103 training deposit points |
| GET | `/api/v1/spatial/grid-predictions` | 1,600 grid-cell prospectivity scores |
| POST | `/api/v1/spatial/predict-point` | Score for a single lat/lon |

```bash
curl -X POST http://localhost:8000/api/v1/spatial/predict-point \
  -H "Content-Type: application/json" \
  -d '{"latitude": 22.249007, "longitude": 85.453637, "band_8": 2526.82, "band_11": 2550.84, "band_12": 2161.84}'
```

```json
{
  "latitude": 22.249007,
  "longitude": 85.453637,
  "prospectivity_score": 81.4,
  "confidence": "High",
  "risk_level": "Low",
  "recommendation": "Favorable mineral indicators found. Recommend exploratory drilling."
}
```

Prospectivity bands: **≥75** High · **50–75** Medium · **25–50** Low · **<25** Very Low.

---

## Dashboard sections

- **India's Manganese Crisis** — production vs demand headline metrics
- **Scenario Simulator** — year + production-boost sliders, forecast chart
- **Place Analysis** — geocode any place (OpenStreetMap Nominatim), count
  deposits within 25 km, breakdown by grade
- **Exploration Map** — prospectivity heatmap, deposit markers, click-to-predict
- **Tabs** — Scan History · Deposits Data · Nearby Deposits · Model Validation
- **Model Validation** — confusion matrix, precision/recall, per-sample explorer
  and a correct/incorrect prediction map over the 52 holdout samples

---

## Satellite processing scripts

Optional — the dashboard runs without them. They need Sentinel-2 bands in
`data/` (not committed; ~520 MB).

```bash
python process_indices.py   # writes outputs/bharweli_NDVI.tif, bharweli_NDMI.tif
python view_rgb.py          # true-colour composite of the AOI
```

`process_indices.py` reads `*_B04_10m.jp2`, `*_B08_10m.jp2`, `*_B11_20m.jp2`
and `*_SCL_20m.jp2`, masks them to the AOI, and plots NDVI / NDMI side by side.

> `process_indices.py:15` hardcodes an absolute local path to `data/`. Point it
> at your own directory before running.

---

## Not in this repository

| Path | Why |
|---|---|
| `data/` | Sentinel-2 `.jp2` bands, ~520 MB — exceeds GitHub's 100 MB file limit |
| `outputs/` | Generated `.tif` rasters; regenerable via `process_indices.py` |
| `.venv/` | Virtual environment |
| `Skyminers.pptx` | Slide deck |

Download imagery from the
[Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/) using AOI
`bharweli_aoi.geojson` (Bharweli, Balaghat) and drop the bands into `data/`.

---

## Known limitations

- The FastAPI backend serves **mock** spatial and forecast data (see above).
- Holdout set is 52 samples from a single belt; metrics are indicative, not a
  field-validated result.
- `sih_manganese_model.pkl` was written with scikit-learn 1.6.1. Newer versions
  emit an `InconsistentVersionWarning` on load — harmless, but retrain if you
  hit a real incompatibility.
- The model file is picked up by `glob("*.pkl")` in `app.py:260`, so keep a
  single `.pkl` in the project root.

---

## License

For hackathon evaluation purposes.

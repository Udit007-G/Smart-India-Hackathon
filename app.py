import os
import sys
import glob as glib
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import requests

# ---------- 1. Expose Top-Level FastAPI Instance ----------
# When deployed to cloud environments expecting FastAPI (e.g., Hugging Face Spaces, Render, Koyeb,
# Railway, or ASGI runners like Uvicorn), `from app import app` provides the valid FastAPI instance.
from backend.mock_api import (
    app,
    get_forecast_data,
    get_deposits_data,
    get_grid_data,
    predict_point_internal,
    get_test_samples,
    TRAIN_DEPOSITS,
)

warnings.filterwarnings("ignore")

# ---------- 2. Environment & Runner Detection ----------
def is_streamlit_running() -> bool:
    """Detect whether script is currently executed inside a Streamlit runner."""
    try:
        from streamlit.runtime import exists
        if not exists():
            return False
    except Exception:
        return False

    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return True



# ---------- 3. Streamlit Application Implementation ----------
def run_streamlit_app():
    import streamlit as st
    import folium
    from streamlit_folium import st_folium
    import plotly.graph_objects as go
    from branca.colormap import LinearColormap

    st.set_page_config(page_title="Manganese Explorer | SIH 26009", layout="wide")

    BACKEND = os.environ.get("BACKEND_URL", "http://localhost:8000")

    # Resilient Data Loaders with Automatic Direct Fallback
    @st.cache_data(ttl=300)
    def load_forecast():
        try:
            res = requests.get(f"{BACKEND}/api/v1/forecast/summary", timeout=1.5)
            if res.ok:
                return res.json()
        except Exception:
            pass
        return get_forecast_data()

    @st.cache_data(ttl=300)
    def load_deposits():
        try:
            res = requests.get(f"{BACKEND}/api/v1/spatial/deposits", timeout=1.5)
            if res.ok:
                return res.json()
        except Exception:
            pass
        return get_deposits_data()

    @st.cache_data(ttl=300)
    def load_grid():
        try:
            res = requests.get(f"{BACKEND}/api/v1/spatial/grid-predictions", timeout=1.5)
            if res.ok:
                return res.json()
        except Exception:
            pass
        return get_grid_data()

    @st.cache_data(ttl=300)
    def load_test_csv():
        for candidate in ["test_set_20.csv", os.path.join(os.path.dirname(__file__), "test_set_20.csv")]:
            if os.path.exists(candidate):
                return pd.read_csv(candidate)
        return pd.DataFrame(get_test_samples())

    forecast = load_forecast()
    deposits = load_deposits()
    grid = load_grid()

    HISTORY_FILE = "scan_history.csv"

    if "history" not in st.session_state:
        if os.path.exists(HISTORY_FILE):
            try:
                st.session_state.history = pd.read_csv(HISTORY_FILE).to_dict("records")
            except Exception:
                st.session_state.history = []
        else:
            st.session_state.history = []

    if "lat" not in st.session_state:
        st.session_state.lat = 21.8045
    if "lon" not in st.session_state:
        st.session_state.lon = 80.1852
    if "score" not in st.session_state:
        st.session_state.score = None

    # Title & Branding
    st.title("Predictive Manganese Exploration System")
    st.write("SIH 26009 | IIIT Vadodara - ICD | Balaghat District, MP")

    # Crisis Executive Metrics
    st.header("India's Manganese Crisis")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Production", "3.38 MT")
    c2.metric("Demand", "8.97 MT")
    c3.metric("Self-Sufficiency", "37.6%")
    c4.metric("Import Deficit", "62.3%")

    c5, c6 = st.columns(2)
    c5.metric("Training Deposits", f"{len(deposits)}")
    c6.metric("Test Samples", "52")

    # Sidebar Controls & Analysis
    with st.sidebar:
        st.title("Controls")

        st.subheader("Scenario Simulator")
        if forecast:
            year = st.slider("Target Year", 2024, 2035, 2030)
            boost = st.slider("Production Boost %", 0, 50, 0)

            yrs = year - 2024
            prod = max(3.38 - (yrs * 0.12) + (3.38 * boost / 100), 1.5)
            demand = 8.97 + (yrs * 0.45)
            gap = max(demand - prod, 0)
            suff = min((prod / demand) * 100, 100)

            st.metric("Production", f"{prod:.2f} MT")
            st.metric("Demand", f"{demand:.2f} MT")
            st.metric("Self-Sufficiency", f"{suff:.1f}%")
            st.metric("Import Gap", f"{gap:.2f} MT")

            if boost > 0:
                st.write(f"Could save **{gap*boost/100:.1f} MT** imports by {year}")

            df_fc = pd.DataFrame(forecast["yearly_forecast"])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_fc["year"], y=df_fc["domestic_production_mt"], name="Production", line=dict(color="green")))
            fig.add_trace(go.Scatter(x=df_fc["year"], y=df_fc["total_demand_mt"], name="Demand", line=dict(color="red", dash="dash")))
            fig.update_layout(height=200, margin=dict(l=0, r=0, t=0, b=0), legend=dict(orientation="h", y=-0.3, x=0.5, xanchor="center"))
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Place Analysis")
        place_name = st.text_input("Enter a place name", placeholder="e.g. Diu, Nagpur, Balaghat")
        if st.button("Analyze Place"):
            if place_name:
                try:
                    geo = requests.get(f"https://nominatim.openstreetmap.org/search?q={place_name}&format=json&limit=1", timeout=5).json()
                    if geo:
                        p_lat, p_lon = float(geo[0]["lat"]), float(geo[0]["lon"])
                        display_name = geo[0].get("display_name", place_name)
                        st.success(f"Found: {display_name[:50]}")
                        st.session_state.lat = p_lat
                        st.session_state.lon = p_lon
                        st.session_state.score = None

                        nearby = []
                        for d in deposits:
                            dist = np.sqrt((d["latitude"] - p_lat)**2 + (d["longitude"] - p_lon)**2) * 111
                            if dist < 25:
                                nearby.append({"name": d["name"], "grade": d["grade"], "km": round(dist, 1)})

                        st.metric("Deposits within 25 km", len(nearby))
                        if nearby:
                            grades = {}
                            for n in nearby:
                                g = n["grade"].split(" ")[0]
                                grades[g] = grades.get(g, 0) + 1
                            st.write(", ".join(f"{k}: {v}" for k, v in sorted(grades.items(), key=lambda x: -x[1])))
                            closest = min(nearby, key=lambda x: x["km"])
                            st.write(f"Closest: **{closest['name']}** at {closest['km']} km")
                        else:
                            st.info("No known deposits found within 25 km of this location.")
                    else:
                        st.warning("Place not found. Try a different name.")
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.warning("Please enter a place name.")

        st.divider()
        st.subheader("How It Works")
        st.write("1. Sentinel-2 captures multispectral imagery")
        st.write("2. Bands 8, 11, 12 isolate manganese mineral signatures")
        st.write("3. ML model scores each location 0–100% prospectivity")
        st.write("4. Click map to obtain instant AI predictions")

    # Exploration Map & Point Inference
    st.header("Exploration Map")
    col1, col2 = st.columns([3, 1])

    with col1:
        show_heat = st.checkbox("Show Prospectivity Heatmap", value=True)
        show_dep = st.checkbox("Show Deposits", value=True)

        m = folium.Map(location=[21.8, 80.18], zoom_start=11, tiles="OpenStreetMap")

        if show_heat and grid:
            cm = LinearColormap(["blue", "cyan", "yellow", "orange", "red"], 0, 1)
            for cell in grid:
                if cell["score"] > 0.3:
                    folium.CircleMarker(
                        [cell["latitude"], cell["longitude"]], radius=8,
                        fill=True, fill_color=cm(cell["score"]), fill_opacity=0.45,
                        color=None).add_to(m)
            cm.add_to(m)

        if show_dep and deposits:
            for d in deposits:
                col = "green" if "High" in d["grade"] else "orange" if "Medium" in d["grade"] else "blue"
                folium.Marker([d["latitude"], d["longitude"]], tooltip=d["name"],
                              icon=folium.Icon(color=col, prefix="fa", icon="circle")).add_to(m)

        if st.session_state.score is not None:
            mcol = "green" if st.session_state.score >= 75 else "orange" if st.session_state.score >= 50 else "red"
        else:
            mcol = "blue"
        folium.Marker([st.session_state.lat, st.session_state.lon],
                      icon=folium.Icon(color=mcol, icon="crosshairs", prefix="fa")).add_to(m)

        map_data = st_folium(m, height=450, key="map", returned_objects=["last_clicked"])

    with col2:
        st.subheader("Point Inference")

        lat = st.number_input("Latitude", value=st.session_state.lat, format="%.6f")
        lon = st.number_input("Longitude", value=st.session_state.lon, format="%.6f")

        if st.button("Set on Map"):
            st.session_state.lat = lat
            st.session_state.lon = lon
            st.session_state.score = None
            st.rerun()

        if map_data and map_data.get("last_clicked"):
            lc = map_data["last_clicked"]
            if st.button("Pick Location from Map"):
                st.session_state.lat = lc["lat"]
                st.session_state.lon = lc["lng"]
                st.session_state.score = None
                st.rerun()

        if "band_8" not in st.session_state:
            st.session_state.band_8 = 1500
        if "band_11" not in st.session_state:
            st.session_state.band_11 = 2100
        if "band_12" not in st.session_state:
            st.session_state.band_12 = 1800

        st.write(f"**Current:** Lat {st.session_state.lat:.4f}, Lon {st.session_state.lon:.4f}")
        st.write(f"**Bands:** B8={st.session_state.band_8} | B11={st.session_state.band_11} | B12={st.session_state.band_12}")

        st.divider()

        st.write("**Band Values** (type or load from test samples)")
        ib1, ib2, ib3 = st.columns(3)
        nb8_text = ib1.text_input("Band 8 (NIR)", value=str(st.session_state.band_8))
        nb11_text = ib2.text_input("Band 11 (SWIR1)", value=str(st.session_state.band_11))
        nb12_text = ib3.text_input("Band 12 (SWIR2)", value=str(st.session_state.band_12))
        try:
            nb8 = float(nb8_text)
            nb11 = float(nb11_text)
            nb12 = float(nb12_text)
        except Exception:
            nb8, nb11, nb12 = float(st.session_state.band_8), float(st.session_state.band_11), float(st.session_state.band_12)

        if st.button("Run Analysis", type="primary"):
            try:
                # Resilient point prediction: try HTTP, fallback to internal engine
                try:
                    r = requests.post(
                        f"{BACKEND}/api/v1/spatial/predict-point",
                        json={"latitude": lat, "longitude": lon, "band_8": nb8, "band_11": nb11, "band_12": nb12},
                        timeout=3
                    ).json()
                except Exception:
                    r = predict_point_internal(lat, lon, nb8, nb11, nb12)

                st.session_state.score = r["prospectivity_score"]
                st.session_state.lat = lat
                st.session_state.lon = lon
                st.session_state.band_8 = int(nb8)
                st.session_state.band_11 = int(nb11)
                st.session_state.band_12 = int(nb12)

                sc = r["prospectivity_score"]
                if sc >= 75:
                    st.success(f"Score: {sc}% - {r['recommendation']}")
                elif sc >= 50:
                    st.warning(f"Score: {sc}% - {r['recommendation']}")
                else:
                    st.error(f"Score: {sc}% - {r['recommendation']}")

                st.write(f"Confidence: {r['confidence']} | Risk: {r['risk_level']}")

                st.session_state.history.append({
                    "time": pd.Timestamp.now().strftime("%H:%M:%S"),
                    "lat": lat, "lon": lon, "band_8": int(nb8), "band_11": int(nb11), "band_12": int(nb12),
                    "score": sc, "conf": r["confidence"]
                })

                try:
                    pd.DataFrame(st.session_state.history).to_csv(HISTORY_FILE, index=False)
                except Exception:
                    pass
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    # ---------- Model Loading (prioritizing primary artifact) ----------
    MODEL = None
    MODEL_TYPE = None
    model_path = None

    candidates = ["sih_manganese_model.pkl"] + glib.glob("*.pkl") + glib.glob("*.joblib")
    seen = set()
    ordered_candidates = [c for c in candidates if not (c in seen or seen.add(c))]

    for cand in ordered_candidates:
        if os.path.exists(cand):
            try:
                import joblib
                MODEL = joblib.load(cand)
                MODEL_TYPE = "sklearn"
                model_path = cand
                break
            except Exception:
                continue

    # Exploration & Validation Tabs
    st.divider()
    t1, t2, t3, t4 = st.tabs(["Scan History", "Deposits Data", "Nearby Deposits", "Model Validation"])

    with t1:
        if st.session_state.history:
            df = pd.DataFrame(st.session_state.history)
            st.dataframe(df, use_container_width=True)
            col_d1, col_d2 = st.columns(2)
            col_d1.download_button("Download CSV", df.to_csv(index=False), "scans.csv")
            if col_d2.button("Clear History"):
                st.session_state.history = []
                if os.path.exists(HISTORY_FILE):
                    try:
                        os.remove(HISTORY_FILE)
                    except Exception:
                        pass
                st.rerun()
        else:
            st.info("No scans recorded yet in this session.")

    with t2:
        if deposits:
            df = pd.DataFrame(deposits)
            st.dataframe(df, use_container_width=True)
            st.download_button("Download CSV", df.to_csv(index=False), "deposits.csv")

    with t3:
        if st.session_state.score is not None and deposits:
            nearby = []
            for d in deposits:
                dist = np.sqrt((d["latitude"] - st.session_state.lat)**2 + (d["longitude"] - st.session_state.lon)**2) * 111
                if dist < 15:
                    nearby.append({"name": d["name"], "grade": d["grade"], "km": round(dist, 1), "status": d["status"]})
            nearby.sort(key=lambda x: x["km"])
            if nearby:
                st.write("Deposits within 15 km:")
                st.dataframe(pd.DataFrame(nearby), use_container_width=True)
            else:
                st.info("No nearby deposits. Unexplored zone.")
        else:
            st.info("Run a scan or inference first to inspect surrounding deposits.")

    with t4:
        st.subheader("Model Validation — 80/20 Train-Test Split")

        test_df = load_test_csv()
        if test_df.empty:
            st.error("test_set_20.csv not found in project folder.")
        else:
            st.info(f"**{len(deposits)} deposits** used for training (80%, Balaghat MP) | **{len(test_df)} samples** held out for testing (20%, Joda Odisha)")

            feature_cols = ["band_8", "band_11", "band_12", "ratio_11_8", "ratio_11_12", "norm_diff"]

            if MODEL is not None:
                st.success(f"Loaded model: {model_path} ({MODEL_TYPE})")
                try:
                    X_df = test_df[feature_cols]
                    if MODEL_TYPE == "sklearn":
                        preds = MODEL.predict(X_df)
                        if hasattr(MODEL, "predict_proba"):
                            probs = MODEL.predict_proba(X_df)[:, 1]
                        else:
                            probs = preds.astype(float)
                    else:
                        preds = MODEL.predict(X_df.values)
                        probs = preds.astype(float)

                    test_df["predicted"] = preds
                    test_df["probability"] = probs
                except Exception as e:
                    st.error(f"Model prediction failed: {e}")
                    st.info("Falling back to rule-based classifier.")
                    MODEL = None

            if MODEL is None:
                st.info("Using rule-based classifier")
                def rule_predict(row):
                    score = 0
                    if row["ratio_11_8"] > 1.0: score += 1
                    if row["ratio_11_12"] > 1.2: score += 1
                    if row["norm_diff"] > 0.05: score += 1
                    if row["band_8"] > 1000: score += 1
                    return 1 if score >= 3 else 0

                test_df["predicted"] = test_df.apply(rule_predict, axis=1)
                test_df["probability"] = test_df.apply(
                    lambda row: min(1.0, (1 if row["ratio_11_8"] > 1.0 else 0) +
                        (1 if row["ratio_11_12"] > 1.2 else 0) +
                        (1 if row["norm_diff"] > 0.05 else 0) +
                        (1 if row["band_8"] > 1000 else 0)) / 4, axis=1)

            actual = test_df["actual_label"].values
            predicted = test_df["predicted"].values

            tp = int(((predicted == 1) & (actual == 1)).sum())
            tn = int(((predicted == 0) & (actual == 0)).sum())
            fp = int(((predicted == 1) & (actual == 0)).sum())
            fn = int(((predicted == 0) & (actual == 1)).sum())
            accuracy = (tp + tn) / len(actual) * 100
            precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0

            c1, c2, c3 = st.columns(3)
            c1.metric("Accuracy", f"{accuracy:.1f}%")
            c2.metric("Precision", f"{precision:.1f}%")
            c3.metric("Recall", f"{recall:.1f}%")

            st.caption("Model was trained on 80% satellite imagery. These metrics show performance on the unseen 20% holdout set.")

            st.write("**Confusion Matrix**")
            cm_df = pd.DataFrame([[tn, fp], [fn, tp]], index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])
            st.dataframe(cm_df.style.background_gradient(cmap="Blues"), use_container_width=False)

            st.divider()
            st.subheader("Explore Test Samples")

            sample_labels = []
            for i, row in test_df.iterrows():
                label = "Mn Present" if int(row["actual_label"]) == 1 else "No Mn"
                sample_labels.append(f"Sample {i+1}: ({row['latitude']:.4f}, {row['longitude']:.4f}) - {label}")

            selected = st.selectbox("Select a test sample", sample_labels, key="test_sample_select")
            sel_idx = sample_labels.index(selected)
            sel_row = test_df.iloc[sel_idx]

            st.session_state.lat = float(sel_row['latitude'])
            st.session_state.lon = float(sel_row['longitude'])
            st.session_state.band_8 = int(float(sel_row['band_8']))
            st.session_state.band_11 = int(float(sel_row['band_11']))
            st.session_state.band_12 = int(float(sel_row['band_12']))

            sc1, sc2 = st.columns(2)
            with sc1:
                st.write(f"**Latitude:** {sel_row['latitude']:.6f}")
                st.write(f"**Longitude:** {sel_row['longitude']:.6f}")
                st.write(f"**Actual Label:** {'Mn Present' if int(sel_row['actual_label']) == 1 else 'No Mn'}")
                st.write(f"**Predicted:** {'Mn Present' if int(sel_row['predicted']) == 1 else 'No Mn'}")
                correct = int(sel_row['actual_label']) == int(sel_row['predicted'])
                if correct:
                    st.success("Correct prediction")
                else:
                    st.error("Wrong prediction")
            with sc2:
                st.write(f"**Band 8:** {sel_row['band_8']:.2f}")
                st.write(f"**Band 11:** {sel_row['band_11']:.2f}")
                st.write(f"**Band 12:** {sel_row['band_12']:.2f}")
                st.write(f"**Ratio 11/8:** {sel_row['ratio_11_8']:.4f}")
                st.write(f"**Ratio 11/12:** {sel_row['ratio_11_12']:.4f}")
                st.write(f"**Norm Diff:** {sel_row['norm_diff']:.6f}")

            st.divider()
            st.write("**All Predictions on Map**")
            val_map = folium.Map(location=[22.25, 85.45], zoom_start=11, tiles="OpenStreetMap")
            for _, row in test_df.iterrows():
                correct = row["actual_label"] == row["predicted"]
                color = "green" if correct else "red"
                popup_text = f"Lat: {row['latitude']:.4f}<br>Lon: {row['longitude']:.4f}<br>Actual: {int(row['actual_label'])}<br>Predicted: {int(row['predicted'])}"
                folium.CircleMarker(
                    [row["latitude"], row["longitude"]], radius=6,
                    fill=True, fill_color=color, fill_opacity=0.7,
                    color=None, popup=popup_text
                ).add_to(val_map)

            folium.Marker(
                [sel_row["latitude"], sel_row["longitude"]],
                icon=folium.Icon(color="white", icon="info-sign", prefix="glyphicon"),
                popup=f"Selected: Sample {sel_idx+1}"
            ).add_to(val_map)

            st_folium(val_map, height=350, key="val_map", returned_objects=[])

            st.download_button("Download Predictions", test_df.to_csv(index=False), "predictions.csv")

    st.divider()
    st.caption("SIH 26009 | IIIT Vadodara - ICD | FastAPI + Streamlit + Leaflet")


# ---------- 4. Execution Entry Point ----------
if is_streamlit_running():
    run_streamlit_app()
elif __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting Manganese Exploration System API on port {port}...")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)

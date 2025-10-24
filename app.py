# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
import shap
import time
import requests
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ----------------------
# Page config
# ----------------------
st.set_page_config(page_title="Traffic Flow Predictor", layout="wide")
st.title("🚦 Traffic Flow Prediction Dashboard")
st.markdown("Explore actual vs predicted traffic, suggested green times, SHAP explanations, and real-time simulation.")

# ----------------------
# Load model & data
# ----------------------
@st.cache_resource
def load_model(path="xgb_traffic_model.pkl"):
    return joblib.load(path)

@st.cache_data
def load_predictions(path="predictions (1).csv"):
    df = pd.read_csv(path)
    for c in df.columns:
        if "date" in c.lower() or "time" in c.lower():
            try: df[c] = pd.to_datetime(df[c])
            except: pass
    if 'date_time' in df.columns:
        df = df.set_index('date_time')
    if 'route' not in df.columns:
        df['route'] = 'Main Road'
    return df

# ----------------------
# Load files
# ----------------------
st.sidebar.header("Files")
try:
    model = load_model("xgb_traffic_model.pkl")
except Exception as e:
    st.sidebar.error(f"Failed to load model: {e}")
    st.stop()

try:
    data = load_predictions("predictions (1).csv")
except Exception as e:
    st.sidebar.error(f"Failed to load predictions.csv: {e}")
    st.stop()

st.sidebar.success("Model & data loaded ✅")

# ----------------------
# Tabs for better layout
# ----------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Data & Metrics", "Actual vs Predicted", "Green-Time & SHAP", "Real-Time Simulation", "ORS Route Fetch"])

# ----------------------
# TAB 1: Data & Metrics
# ----------------------
with tab1:
    st.subheader("Data Preview")
    st.dataframe(data.head(10))

    if 'predicted_volume' in data.columns and 'traffic_volume' in data.columns:
        y_true = data['traffic_volume']
        y_pred = data['predicted_volume']
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100
        col1, col2, col3 = st.columns(3)
        col1.metric("MAE", f"{mae:.2f}")
        col2.metric("RMSE", f"{rmse:.2f}")
        col3.metric("MAPE", f"{mape:.2f}%")
    else:
        st.info("`predicted_volume` and/or `traffic_volume` not found in data.")

# ----------------------
# TAB 2: Actual vs Predicted
# ----------------------
with tab2:
    st.subheader("Interactive Actual vs Predicted")
    n_points = st.slider("Number of latest samples to display", min_value=5, max_value=min(1000,len(data)), value=50, step=5)
    plot_df = data.tail(n_points).copy()
    if isinstance(plot_df.index, pd.DatetimeIndex):
        plot_df = plot_df.reset_index()
    if 'predicted_volume' in plot_df.columns and 'traffic_volume' in plot_df.columns:
        fig = px.line(plot_df, x=plot_df.columns[0], y=['traffic_volume', 'predicted_volume'],
                      labels={'value':'Traffic Volume', plot_df.columns[0]:'Time'},
                      title=f"Actual vs Predicted (last {n_points} samples)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        numeric_cols = plot_df.select_dtypes([np.number]).columns.tolist()
        st.line_chart(plot_df[numeric_cols])

# ----------------------
# TAB 3: Green-Time & SHAP
# ----------------------
with tab3:
    st.subheader("Suggested Green-Time (Rule-Based)")
    threshold = st.slider("Congestion threshold (vehicles/hour)", min_value=100, max_value=5000,
                          value=int(data['traffic_volume'].quantile(0.85)) if 'traffic_volume' in data.columns else 1000)
    
    def suggest_green(predicted_volume, base_green=30, threshold=threshold):
        if predicted_volume > threshold:
            factor = (predicted_volume - threshold)/(threshold+1e-9)
            add_seconds = int(np.clip(5 + 25*factor,5,30))
            return base_green + add_seconds
        elif predicted_volume < 0.5*threshold:
            return max(10, base_green-10)
        else:
            return base_green

    slice_df = data.head(10).copy()
    if 'predicted_volume' in slice_df.columns:
        slice_df['suggested_green'] = slice_df['predicted_volume'].apply(suggest_green)
        st.dataframe(slice_df[['predicted_volume','suggested_green']])
    
    st.subheader("SHAP Feature Importance")
    feature_names = getattr(model, 'feature_names_in_', None)
    if feature_names is None or len(feature_names) == 0:
        excluded = ['traffic_volume','predicted_volume']
        feature_names = [c for c in data.select_dtypes([np.number]).columns if c not in excluded]
    st.write("Features used:", feature_names)
    
    X_shap = data[feature_names].fillna(0).astype(float)
    
    @st.cache_data
    def compute_shap(_model, X):
        try:
            explainer = shap.Explainer(_model, X)
            return explainer(X)
        except:
            explainer = shap.TreeExplainer(_model)
            return explainer.shap_values(X)
    
    with st.spinner("Computing SHAP values..."):
        shap_values = compute_shap(model, X_shap)
        st.success("SHAP computed ✅")
    
    st.write("Global feature importance (bar chart)")
    fig, ax = plt.subplots(figsize=(8,4))
    try:
        shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False)
        st.pyplot(fig)
        plt.clf()
    except: st.write("Cannot plot SHAP summary.")

# ----------------------
# TAB 4: Real-Time Simulation
# ----------------------
with tab4:
    st.subheader("🚦 Real-Time Traffic Flow Simulation (Simulated)")
    route_selected = st.selectbox("Select Route", data['route'].unique())
    route_df = data[data['route']==route_selected].reset_index()
    num_rows = st.slider("Number of time steps", 5, min(50,len(route_df)), 10)
    chart_placeholder = st.empty()
    metrics_placeholder = st.empty()
    
    for i in range(num_rows):
        slice_df = route_df.iloc[:i+1]
        
        # Metrics update
        if 'predicted_volume' in slice_df.columns and 'traffic_volume' in slice_df.columns:
            y_true = slice_df['traffic_volume']
            y_pred = slice_df['predicted_volume']
            mae = np.mean(np.abs(y_true - y_pred))
            metrics_placeholder.metric("MAE", f"{mae:.2f}")
        
        plt.figure(figsize=(8,4))
        plt.plot(slice_df.index, slice_df['traffic_volume'], color='green', marker='o', label='Actual')
        plt.plot(slice_df.index, slice_df['predicted_volume'], color='red', marker='x', label='Predicted')
        
        for idx, val in enumerate(slice_df['predicted_volume']):
            if val > slice_df['predicted_volume'].quantile(0.85):
                plt.scatter(slice_df.index[idx], val, color='red', s=100)
            elif val > slice_df['predicted_volume'].quantile(0.6):
                plt.scatter(slice_df.index[idx], val, color='orange', s=80)
            else:
                plt.scatter(slice_df.index[idx], val, color='green', s=50)
        
        plt.xlabel("Time Index")
        plt.ylabel("Traffic Volume")
        plt.title(f"Traffic Flow Simulation - {route_selected}")
        plt.legend()
        plt.grid(True)
        chart_placeholder.pyplot(plt.gcf())
        plt.clf()
        time.sleep(0.5)
    
    st.success("Simulation complete ✅")

# ----------------------
# TAB 5: ORS Route Fetch Example
# ----------------------
with tab5:
    st.subheader("Fetch Route using OpenRouteService API")
    start_coords = st.text_input("Start coordinates (lon,lat)", "-122.42,37.78")
    end_coords = st.text_input("End coordinates (lon,lat)", "-122.45,37.91")
    fetch_button = st.button("Fetch Route")

    ORS_API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6IjNiNmU3MzY5YTU1ZTQ4OWU5Y2FiNWJlYTIxZTVhM2YxIiwiaCI6Im11cm11cjY0In0="

    if fetch_button:
        start = [float(c) for c in start_coords.split(",")]
        end = [float(c) for c in end_coords.split(",")]
        url = "https://api.openrouteservice.org/v2/directions/driving-car/geojson"
        headers = {"Authorization": ORS_API_KEY, "Content-Type": "application/json"}
        body = {
            "coordinates": [start, end]
        }
        response = requests.post(url, json=body, headers=headers)
        if response.status_code == 200:
            route_data = response.json()
            st.write("Route fetched successfully ✅")
            st.json(route_data)
        else:
            st.error(f"Failed to fetch route: {response.status_code} - {response.text}")

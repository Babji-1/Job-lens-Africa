import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

# PAGE CONFIG
st.set_page_config(
    page_title="AfriWork",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AfriWork")
st.subheader("African Labour Market Intelligence & Employment Forecasting")

st.markdown("---")

# PATH
MODEL_PATH = Path("models")

# LOADER
@st.cache_resource
def load_model(name):
    return joblib.load(MODEL_PATH / f"{name}.pkl")


@st.cache_data
def load_data(name):
    df = pd.read_csv(MODEL_PATH / f"{name}.csv")
    # Leave df["ds"] as raw numbers/strings for clean plotting conversion
    return df


# MODEL LISTS
broad_models = {
    "Agriculture":"Broad_Agriculture",
    "Industry":"Broad_Industry",
    "Services":"Broad_Services"
}

aggregate_models = {
    "Manufacturing":"Aggregate_Manufacturing",
    "Construction":"Aggregate_Construction",
    "Mining & Utilities": "Aggregate_Mining_and_quarrying_Electricity_gas_and_water_supply",
    "Trade, Transport & Business": "Aggregate_Trade_Transportation_Accommodation_and_Food_and_Business_and_Administrative_Services",
    "Public & Community Services": "Aggregate_Public_Administration_Community_Social_and_other_Services_and_Activities"
}

# Automatically discover every ISIC model
isic_models = {}
for file in MODEL_PATH.glob("*.pkl"):
    name = file.stem
    if name.startswith(("A._","B._","C._","D._","E._","F._","G._","H._",
                        "I._","J._","K._","L._","M._","N._","O._","P._",
                        "Q._","R._","S._","T._")):
        display = name.replace("_"," ")
        isic_models[display] = name

# SIDEBAR
st.sidebar.header("Forecast Options")

level = st.sidebar.selectbox(
    "Forecast Category",
    ["Broad Sector", "Aggregate Sector", "ISIC Rev.4 Activity"]
)

if level == "Broad Sector":
    display_name = st.sidebar.selectbox("Choose Sector", list(broad_models.keys()))
    model_name = broad_models[display_name]
elif level == "Aggregate Sector":
    display_name = st.sidebar.selectbox("Choose Sector", list(aggregate_models.keys()))
    model_name = aggregate_models[display_name]
else:
    display_name = st.sidebar.selectbox("Choose Activity", sorted(isic_models.keys()))
    model_name = isic_models[display_name]

forecast_years = st.sidebar.slider("Forecast Horizon", 1, 5, 5)
predict = st.sidebar.button("Predict")

# FORECAST
if predict:
    model = load_model(model_name)
    history = load_data(model_name)

    # -----------------------------------------------------------------
    # FIX: ROBUST ALGEBRAIC FORECAST Bypasses Internal Statsmodels Indexing
    # Extracting coefficients directly mathematically projects future values
    # -----------------------------------------------------------------
    # Get parameters: [const, ar.L1, ar.L2, ar.L3]
    params = model.params
    const = params.get('const', 0.0)
    ar_coefs = [params.get('ar.L1', 0.0), params.get('ar.L2', 0.0), params.get('ar.L3', 0.0)]
    
    # Work with historical differences since your model is integrated ARIMA(3, 1, 0)
    history_y = history["y"].tolist()
    diffs = np.diff(history_y).tolist()
    
    # Calculate predictions step-by-step
    forecast_diffs = []
    for step in range(forecast_years):
        # Gather the 3 most recent differences
        lag_1 = forecast_diffs[-1] if len(forecast_diffs) >= 1 else (diffs[-1] if len(diffs) >= 1 else 0)
        lag_2 = forecast_diffs[-2] if len(forecast_diffs) >= 2 else (diffs[-2] if len(diffs) >= 2 else 0)
        lag_3 = forecast_diffs[-3] if len(forecast_diffs) >= 3 else (diffs[-3] if len(diffs) >= 3 else 0)
        
        # Calculate new integrated variance element
        next_diff = const + ar_coefs[0]*lag_1 + ar_coefs[1]*lag_2 + ar_coefs[2]*lag_3
        forecast_diffs.append(next_diff)
    
    # Reconstruct original scales (undifferencing)
    forecast = []
    current_value = history_y[-1]
    for d in forecast_diffs:
        current_value += d
        forecast.append(current_value)
    # -----------------------------------------------------------------

    # Align clean calendar timeline
    last_historical_year = int(float(history["ds"].iloc[-1]))
    future_years_list = list(range(last_historical_year + 1, last_historical_year + forecast_years + 1))
    
    future_dates = pd.to_datetime([f"{y}-01-01" for y in future_years_list])
    history_dates = pd.to_datetime([f"{int(float(y))}-01-01" for y in history["ds"]])

    forecast_x = [history_dates.iloc[-1]] + list(future_dates)
    forecast_y = [history_y[-1]] + forecast

    fig = go.Figure()

    # Historical
    fig.add_trace(
        go.Scatter(
            x=history_dates,
            y=history_y,
            mode="lines+markers",
            name="Historical",
            line=dict(color="#2E8B57", width=3)
        )
    )

    # Forecast
    fig.add_trace(
        go.Scatter(
            x=forecast_x,
            y=forecast_y,
            mode="lines+markers",
            name="Forecast",
            line=dict(color="#D4A017", width=3, dash="dash")
        )
    )
    
    # Highlight Forecast Region
    fig.add_vrect(
        x0=future_dates[0],
        x1=future_dates[-1],
        fillcolor="#FFD54F",
        opacity=0.15,
        line_width=0
    )

    fig.update_layout(
        title=f"{display_name} Employment Forecast",
        template="plotly_white",
        height=650,
        font=dict(family="Arial", size=15, color="#1F3B63"),
        xaxis_title="Year",
        yaxis_title="Employees (Thousands)",
        legend_title="Legend"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.success(f"Forecast generated for {display_name} ({forecast_years} years).")

    st.dataframe(
        pd.DataFrame({
            "Year": future_years_list,
            "Forecast Employment": [round(val, 2) for val in forecast]
        }),
        use_container_width=True
    )

# FOOTER
st.markdown("---")
st.caption("AfriWork | African Labour Market Intelligence Platform | ARIMA Forecasting")

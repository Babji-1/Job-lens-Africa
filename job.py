import streamlit as st
import pandas as pd
import joblib
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
    df["ds"] = pd.to_datetime(df["ds"])
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
    # FIX: PRODUCTION-SAFE INDEX PREDICTION
    # Using row-count indices instead of .forecast(steps=...) 
    # to avoid system pointer errors in cloud deployments.
    # -----------------------------------------------------------------
    start_idx = len(history)
    end_idx = start_idx + forecast_years - 1
    
    # Run prediction safely across different OS/environments
    forecast_series = model.predict(start=start_idx, end=end_idx)
    forecast = list(forecast_series)
    # -----------------------------------------------------------------

    future_dates = pd.date_range(
        start=history["ds"].max() + pd.DateOffset(years=1),
        periods=forecast_years,
        freq="YS"
    )

    forecast_x = [history["ds"].iloc[-1]] + list(future_dates)
    forecast_y = [history["y"].iloc[-1]] + forecast

    fig = go.Figure()

    # Historical
    fig.add_trace(
        go.Scatter(
            x=history["ds"],
            y=history["y"],
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
            "Year": future_dates.year,
            "Forecast Employment": [round(val, 2) for val in forecast]
        }),
        use_container_width=True
    )

# FOOTER
st.markdown("---")
st.caption("AfriWork | African Labour Market Intelligence Platform | ARIMA Forecasting")

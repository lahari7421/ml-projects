# app_house.py
# Run: streamlit run app_house.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ── Train Model ───────────────────────────────────────────
@st.cache_resource
def train_model():
    housing = fetch_california_housing()
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = housing.target * 100000

    features = ['MedInc','HouseAge','AveRooms','AveBedrms','Population']
    X = X[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2   = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return model, scaler, features, r2, rmse, y_test, y_pred

model, scaler, features, r2, rmse, y_test, y_pred = train_model()

# ── UI ────────────────────────────────────────────────────
st.set_page_config(page_title="House Price Predictor", page_icon="🏠")
st.title("🏠 House Price Predictor")
st.markdown("Predict California house prices using Linear Regression.")
st.divider()

col1, col2 = st.columns(2)
with col1:
    st.subheader("🏘️ House Details")
    med_inc   = st.slider("Median Income (×$10k)", 0.5, 15.0, 5.0, 0.1)
    house_age = st.slider("House Age (years)",      1,   52,   20)
    ave_rooms = st.slider("Avg Rooms",              1.0, 10.0, 5.0, 0.1)

with col2:
    st.subheader("📊 Area Info")
    ave_bedrms = st.slider("Avg Bedrooms",   0.5, 5.0, 1.0, 0.1)
    population = st.number_input("Area Population",
                                  min_value=100,
                                  max_value=40000,
                                  value=1500, step=100)

st.divider()

if st.button("💰 Predict Price", use_container_width=True, type="primary"):
    inp = np.array([[med_inc, house_age, ave_rooms, ave_bedrms, population]])
    inp_scaled  = scaler.transform(inp)
    prediction  = model.predict(inp_scaled)[0]

    st.success(f"### 💰 Estimated Price: ${prediction:,.0f}")

    if prediction > 300000:
        st.info("🏆 Premium Property")
    elif prediction > 180000:
        st.info("✅ Mid-range Property")
    else:
        st.info("💡 Affordable Property")

    # Price gauge bar
    max_price = 500000
    pct = min(prediction / max_price, 1.0)
    st.progress(pct, text=f"Price Range: ${prediction:,.0f} / ${max_price:,.0f}")

st.divider()

# ── Model Performance ─────────────────────────────────────
with st.expander("📊 View Model Performance"):
    col_a, col_b = st.columns(2)
    col_a.metric("R² Score", f"{r2:.4f}")
    col_b.metric("RMSE",     f"${rmse:,.0f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].scatter(y_test, y_pred, alpha=0.4, color='steelblue')
    axes[0].plot([y_test.min(), y_test.max()],
                 [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0].set_xlabel("Actual"); axes[0].set_ylabel("Predicted")
    axes[0].set_title("Predicted vs Actual")

    residuals = y_test - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.4, color='purple')
    axes[1].axhline(0, color='red', linestyle='--')
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Residual")
    axes[1].set_title("Residual Plot")

    plt.tight_layout()
    st.pyplot(fig)

st.caption("Built with Python · Scikit-learn · Streamlit | ML & AI Course — Mini Project 3")
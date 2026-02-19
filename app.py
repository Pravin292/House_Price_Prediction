import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------- Load model artifacts ----------------
model = joblib.load("xgboost_ames_model.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------- Feature config ----------------
FEATURE_CONFIG = {
    "GrLivArea": {
        "label": "Above Ground Living Area",
        "unit": "sq ft",
        "min": 300,
        "max": 6000,
        "step": 50,
        "help": "Finished living area above ground"
    },
    "FullBath": {
        "label": "Full Bathrooms",
        "unit": "count",
        "min": 0,
        "max": 5,
        "step": 1,
        "help": "Bathrooms with tub or shower"
    },
    "TotalBsmtSF": {
        "label": "Basement Area",
        "unit": "sq ft",
        "min": 0,
        "max": 4000,
        "step": 50,
        "help": "Total basement square footage"
    },
    "GarageCars": {
        "label": "Garage Capacity",
        "unit": "cars",
        "min": 0,
        "max": 5,
        "step": 1,
        "help": "Number of cars the garage can hold"
    },
    "YearBuilt": {
        "label": "Year Built",
        "unit": "year",
        "min": 1870,
        "max": 2025,
        "step": 1,
        "help": "Construction year"
    },
    "OverallQual": {
        "label": "Overall Quality",
        "unit": "rating (1–10)",
        "min": 1,
        "max": 10,
        "step": 1,
        "help": "Material and finish quality"
    }
}

FEATURE_ORDER = list(FEATURE_CONFIG.keys())

# ---------------- Page config ----------------
st.set_page_config(
    page_title="Ames Housing Price Prediction",
    layout="centered"
)

# ---------------- Minimal styling ----------------
st.markdown("""
<style>
    .stApp { background-color: #0f1115; color: #e6e6e6; }
    h1, h2, h3 { color: #f1f1f1; font-weight: 600; }
    .stButton > button {
        background-color: #1f2937;
        color: #ffffff;
        border-radius: 8px;
        padding: 0.5rem 1.4rem;
        border: 1px solid #374151;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- Header ----------------
st.title("Ames Housing Price Prediction")
st.caption("XGBoost-based house price estimation")

st.divider()

# ---------------- Input section ----------------
st.subheader("House Details")

user_input = {}
for feature in FEATURE_ORDER:
    cfg = FEATURE_CONFIG[feature]
    user_input[feature] = st.number_input(
        label=f"{cfg['label']} ({cfg['unit']})",
        min_value=float(cfg["min"]),
        max_value=float(cfg["max"]),
        step=float(cfg["step"]),
        help=cfg["help"]
    )

st.divider()

# ---------------- Prediction ----------------
if st.button("Predict Price"):
    input_df = pd.DataFrame([user_input])[FEATURE_ORDER]
    input_scaled = scaler.transform(input_df)

    log_pred = model.predict(input_scaled)[0]

    # 🔒 Clamp log prediction (prevents inf)
    log_pred = np.clip(log_pred, 8, 14)

    # 🔒 Correct inverse for log1p
    prediction = np.expm1(log_pred)

    # 🔒 Final sanity bounds
    prediction = np.clip(prediction, 50000, 1000000)

    st.markdown(
        f"""
        <div style="padding:1rem;
                    border-radius:10px;
                    background:#111827;
                    border:1px solid #374151;
                    font-size:1.2rem;">
            Estimated Sale Price<br>
            <strong>$ {prediction:,.0f}</strong>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.caption("Prediction based on historical Ames Housing data")
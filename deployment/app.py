import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


import pandas as pd
import streamlit as st
from joblib import load
import yaml
from src.data_preprocessing import preprocess_and_engineer


# Config
def load_config(path="config/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


@st.cache_resource
def load_model(cfg):
    model_path = os.path.join(
        cfg["paths"]["models_dir"], cfg["serialization"]["model_filename"]
    )
    return load(model_path)


def preprocess_for_inference(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:

    return df


st.set_page_config(
    page_title=" Telecommunication Customer Churn Predictor",
    page_icon="📡",
    layout="wide",
)
st.title("Telecommunication Customer Churn Prediction")

cfg = load_config()
try:
    model = load_model(cfg)
    st.success(f"Model loaded: {cfg['serialization']['model_filename']}")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

tab1, tab2 = st.tabs(["Single customer", "Batch prediction"])

with tab1:
    st.subheader("Single customer input")
    gender = st.selectbox("Gender", ["Male", "Female"])
    SeniorCitizen = st.selectbox("SeniorCitizen", ["No", "Yes"])
    Partner = st.selectbox("Partner", ["No", "Yes"])
    Dependents = st.selectbox("Dependents", ["No", "Yes"])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=120, value=12)
    PhoneService = st.selectbox("PhoneService", ["No", "Yes"])
    MultipleLines = st.selectbox("MultipleLines", ["No", "Yes"])
    InternetService = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox(
        "OnlineSecurity", ["No", "Yes", "No internet service"]
    )
    OnlineBackup = st.selectbox("OnlineBackup", ["No", "Yes", "No internet service"])
    DeviceProtection = st.selectbox(
        "DeviceProtection", ["No", "Yes", "No internet service"]
    )
    TechSupport = st.selectbox("TechSupport", ["No", "Yes", "No internet service"])
    StreamingTV = st.selectbox("StreamingTV", ["No", "Yes", "No internet service"])
    StreamingMovies = st.selectbox(
        "StreamingMovies", ["No", "Yes", "No internet service"]
    )
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("PaperlessBilling", ["No", "Yes"])
    PaymentMethod = st.selectbox(
        "PaymentMethod",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )
    MonthlyCharges = st.number_input(
        "MonthlyCharges", min_value=0.0, value=00.0, step=0.1
    )
    TotalCharges = st.number_input("TotalCharges", min_value=0.0, value=000.0, step=0.1)

    if st.button("Predict churn"):
        row = {
            "gender": gender,
            "SeniorCitizen": SeniorCitizen,
            "Partner": Partner,
            "Dependents": Dependents,
            "tenure": tenure,
            "PhoneService": PhoneService,
            "MultipleLines": MultipleLines,
            "InternetService": InternetService,
            "OnlineSecurity": OnlineSecurity,
            "OnlineBackup": OnlineBackup,
            "DeviceProtection": DeviceProtection,
            "TechSupport": TechSupport,
            "StreamingTV": StreamingTV,
            "StreamingMovies": StreamingMovies,
            "Contract": Contract,
            "PaperlessBilling": PaperlessBilling,
            "PaymentMethod": PaymentMethod,
            "MonthlyCharges": MonthlyCharges,
            "TotalCharges": TotalCharges,
        }
        churn = pd.DataFrame([row])
        churn_proc = preprocess_and_engineer(churn, cfg)
        y_pred = model.predict(churn_proc)

        # probability
        proba_fn = getattr(model, "predict_proba", None)
        prob = None
        if proba_fn:
            proba = proba_fn(churn_proc)
            prob = float(proba[0][1]) if proba is not None else None

        label = "Yes" if int(y_pred[0]) == 1 else "No"
        st.metric("Churn prediction", label)
        if prob is not None:
            st.write(f"Churn probability: {prob:.3f}")
            st.progress(min(max(prob, 0.0), 1.0))

with tab2:
    st.subheader("Batch prediction via CSV")
    st.caption("Upload a CSV with the same columns used during training.")
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file is not None:
        churn = pd.read_csv(file)
        st.write("Preview:", churn.head())

        # drop the customerid
        if "customerID" in churn.columns:
            churn_id = churn["customerID"]
            churn = churn.drop(columns=["customerID"])
        else:
            churn_id = None

        # apply preprocessing and feature engineering

        churn_proc = preprocess_and_engineer(churn, cfg)
        st.write("prrocessed columns", churn_proc.columns.tolist())
        y_pred = model.predict(churn_proc)
        # Try probabilities
        proba_fn = getattr(model, "predict_proba", None)
        churn_proba = None
        if proba_fn:
            try:
                probs = proba_fn(churn_proc)
                churn_proba = [p[1] for p in probs]
            except Exception:
                pass
        #  output

        out = churn.copy()
        out["churn_pred"] = y_pred
        if churn_proba is not None:
            out["churn_proba"] = churn_proba
        out["Churn"] = out["churn_pred"].map({0: "No", 1: "Yes"})

        st.write("Results:", out.head())
        csv = out.to_csv(index=False).encode("utf-8")
        st.download_button("Download predicted CSV", csv, "predictions.csv", "text/csv")

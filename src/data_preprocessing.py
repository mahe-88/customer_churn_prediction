
import pandas as pd

def preprocess_and_engineer(churn: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Apply preprocessing and feature engineering steps to raw churn dataset.
    Mirrors your notebook logic but simplified for clarity.
    """

    # 1. Fix TotalCharges dtype and fill missing values
    churn["TotalCharges"] = pd.to_numeric(churn["TotalCharges"], errors="coerce")
    churn["TotalCharges"] = churn["TotalCharges"].fillna(churn["tenure"] * churn["MonthlyCharges"])

    # 2. SeniorCitizen mapping (0 -> No, 1 -> Yes)
    churn["SeniorCitizen"] = churn["SeniorCitizen"].map({0: "No", 1: "Yes"})

    # 3. Replace "No internet service"/"No phone service" with "No"
    for col in cfg["feature_engineering"]["replace_no_service_cols"]:
        churn[col] = churn[col].replace({"No internet service": "No", "No phone service": "No"})

    # 4. Tenure group (binning)
    churn["tenure_group"] = pd.cut(
        churn["tenure"],
        bins=cfg["feature_engineering"]["tenure_bins"],
        labels=cfg["feature_engineering"]["tenure_labels"]
    )

    # 5. Fiber optic flag
    churn["FiberOptic"] = (churn["InternetService"] == "Fiber optic").astype(int)

    # 6. Family flag (Partner or Dependents)
    churn["Family"] = ((churn["Partner"] == "Yes") | (churn["Dependents"] == "Yes")).astype(int)

    # 7. Protective services count
    service_cols = cfg["feature_engineering"]["service_cols"]
    churn["ProtectiveServices"] = (churn[service_cols] == "Yes").sum(axis=1)

    # 8. Streaming addict count
    streaming_cols = cfg["feature_engineering"]["streaming_cols"]
    churn["StreamingAddict"] = (churn[streaming_cols] == "Yes").sum(axis=1)

    # 9. Contract score (ordinal mapping)
    churn["ContractScore"] = churn["Contract"].map(cfg["feature_engineering"]["contract_score_map"])

    # 10. Payment risk score (ordinal mapping)
    churn["PaymentRisk"] = churn["PaymentMethod"].map(cfg["feature_engineering"]["payment_risk_map"])

    # 11. High charges flag
    churn["HighCharges"] = (churn["MonthlyCharges"] > cfg["feature_engineering"]["high_charges_threshold"]).astype(int)

    # 12. Avg monthly spend
    churn["AvgMonthlySpend"] = churn["TotalCharges"] / (churn["tenure"] + 1)

    # 13. New customer flag
    churn["Newcustomer"] = (churn["tenure"] <= cfg["feature_engineering"]["new_customer_max_tenure"]).astype(int)

    return churn

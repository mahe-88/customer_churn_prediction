import sys
import os

# project root always on the sys path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
from config import load_config
from src.model import load_model
from src.data_preprocessing import preprocess_and_engineer


def main():
    # 1. Loading config
    cfg = load_config()

    # pd.set_option('display.max_columns',None)

    # 2. Load raw new customer data
    new_data_path = os.path.join(cfg["paths"]["data_dir"], "new_customers.csv")
    churn_new_raw = pd.read_csv(new_data_path)
    print(f"[predict] Raw new data shape: {churn_new_raw.shape}")

    # 3.  preprocessing and feature engineering
    churn_new = preprocess_and_engineer(churn_new_raw, cfg)
    print(f"[predict] After preprocessing: {churn_new.shape}")

    # 4. Drop target column
    target = cfg["data"]["target"]
    if target in churn_new.columns:
        churn_new = churn_new.drop(columns=[target])

    # 5. Load trained model
    pipe = load_model(cfg)
    print("[predict] Model loaded successfully")

    # 6. Predict churn probabilities and classes
    y_proba = pipe.predict_proba(churn_new)[:, 1]
    y_pred = pipe.predict(churn_new)

    # 7.  predictions
    churn_new_raw = churn_new_raw.drop(columns=["customerID"], errors="ignore")

    churn_new_raw["churn_proba"] = y_proba
    churn_new_raw["churn_pred"] = y_pred

    # 8.readable formate  Churn column
    churn_new_raw["Churn"] = churn_new_raw["churn_pred"].map({0: "No", 1: "Yes"})

    # 9. Save predictions to the csv file
    output_path = os.path.join(cfg["paths"]["models_dir"], "predictions.csv")
    churn_new_raw.to_csv(output_path, index=False)
    print(f"[predict] Predictions saved to {output_path}")

    # 10.  sample predictions of unseen data
    print(churn_new_raw.head())


if __name__ == "__main__":
    main()

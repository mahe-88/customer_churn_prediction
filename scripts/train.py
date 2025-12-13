import sys
import os

# project root is always on sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
from sklearn.model_selection import train_test_split

from config import load_config
from src.data_loading import load_processed_data
from src.model import build_model, train_model, evaluate_model, save_model


def main():
    # 1. Load configuration
    cfg = load_config()

    # 2. Load processed data
    churn = load_processed_data(cfg)
    print(f"[run_train] Processed data shape: {churn.shape}")

    # 3. Split data into train/test
    target = cfg["data"]["target"]
    X = churn.drop(columns=[target])

    #  Mapping target 'No'/'Yes' to 0/1 for metrics

    y_raw = churn[target]
    label_map = {"No": 0, "Yes": 1}
    y = y_raw.map(label_map)
    if y.isna().any():
        raise ValueError(
            f"Unexpected target values found: {set(y_raw.unique())}. Expected only 'No'/'Yes'."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,y, test_size=cfg["training"]["test_size"],
        random_state=cfg["project"]["seed"],
        stratify=y if cfg["training"]["stratify"] else None,
    )
    print(f"[run_train] Train: {X_train.shape}, Test: {X_test.shape}")

    # 4. Build and train model
    pipe = build_model(cfg)
    pipe = train_model(pipe, X_train, y_train)

    # 5. Evaluate model (prints report and plots)
    metrics = evaluate_model(pipe, X_test, y_test, cfg)
    print(f"[run_train] Metrics: {metrics}")

    # 6. Save model
    save_model(pipe, cfg)


if __name__ == "__main__":
    main()

import sys
import os
import pandas as pd

# project root is always on sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import load_config
from src.data_loading import load_processed_data
from src.data_preprocessing import preprocess_and_engineer
from src.model import build_model, load_model


def test_full_pipeline():
    # 1. Load config
    cfg = load_config()
    print("\n[TEST] Config loaded successfully")

    # 2. Load processed training data
    df = load_processed_data(cfg)
    assert not df.empty
    print(f"[TEST] Processed data shape: {df.shape}")

    # 3. Apply preprocessing and feature engineering on a  sample
    df_sample = df.sample(3, random_state=42).drop(columns=[cfg["data"]["target"]])
    df_proc = preprocess_and_engineer(df_sample, cfg)
    print(f"[TEST] After preprocessing: {df_proc.shape}")

    # 4. Build model pipeline
    pipe = build_model(cfg)
    assert hasattr(pipe, "predict")
    print("[TEST] Model pipeline built successfully")

    # 5. Load trained model
    trained_pipe = load_model(cfg)
    print("[TEST] Trained model loaded successfully")

    # 6. Predict on sample
    y_pred = trained_pipe.predict(df_proc)
    print(f"[TEST] Predictions: {y_pred}")
    print(
        "Test Prediction for Readable format:",
        ["Yes" if val == 1 else "No" for val in y_pred],
    )

    assert y_pred.shape[0] == df_proc.shape[0]


test_full_pipeline()

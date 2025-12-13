
import sys
import os


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_config
import pandas as pd


def load_raw_data(cfg: dict) -> pd.DataFrame:
    """
    Load the raw IBM churn dataset from CSV.
    """
    raw_path = os.path.join(cfg["paths"]["data_raw"], cfg["data"]["raw_file"])
    churn = pd.read_csv(raw_path)
    return churn

def load_processed_data(cfg: dict) -> pd.DataFrame:
    """
    Load the processed churn dataset after the feature engineering.
    """
    processed_path = os.path.join(cfg["paths"]["data_processed"], cfg["data"]["processed_file"])
    churn = pd.read_csv(processed_path)
    return churn

def save_processed_data(cfg: dict, churn: pd.DataFrame) -> None:
    """
    Save processed dataset to CSV.
    """
    processed_path = os.path.join(cfg["paths"]["data_processed"], cfg["data"]["processed_file"])
    os.makedirs(cfg["paths"]["data_processed"], exist_ok=True)
    churn.to_csv(processed_path, index=False)
    print(f"Processed data saved to {processed_path}")
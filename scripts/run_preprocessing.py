import sys
import os

#  project root is on sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_config
from src.data_loading import load_raw_data, save_processed_data
from src.data_preprocessing import preprocess_and_engineer


def main():
    # 1. Load configuration
    cfg = load_config()

    # 2. Load raw IBM dataset
    churn_raw = load_raw_data(cfg)
    print(f"[run_preprocessing] Raw shape: {churn_raw.shape}")

    # 3.  preprocessing and feature engineering
    churn_proc = preprocess_and_engineer(churn_raw, cfg)
    print(f"[run_preprocessing] Processed shape: {churn_proc.shape}")

    # 4. Save processed dataset
    save_processed_data(cfg, churn_proc)


if __name__ == "__main__":
    main()

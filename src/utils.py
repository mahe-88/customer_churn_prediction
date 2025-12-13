
import yaml
import os
import sys
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path='config/config.yaml'):
    """Load configuration from yaml file."""
    abs_path = os.path.abspath(config_path)
    with open(abs_path) as f:
        config = yaml.safe_load(f)
    logging.info(f"Config loaded from {abs_path}")
    return config

def binary_map(series):
    """Convert Yes/No to 1/0 for binary columns."""
    return series.map({
        'Yes': 1,
        'No': 0,
        'No phone service': 0,
        'No internet service': 0
    }).fillna(0)
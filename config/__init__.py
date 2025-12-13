import os
import yaml

def load_config(config_path: str = None) -> dict:
    """
    Load YAML configuration as a Python dict.
    """
    if config_path is None:
        config_path = os.path.join(os.getcwd(), "config", "config.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return cfg



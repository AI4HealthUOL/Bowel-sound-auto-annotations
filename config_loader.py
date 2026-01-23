"""
Centralized YAML configuration loader.

This module guarantees:
- Single source of truth for parameters
- Early failure on invalid config
"""

from pathlib import Path
import yaml

def load_config(path: str) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("Invalid YAML structure: root must be a dictionary")

    return cfg

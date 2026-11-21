"""Configuration loading and project-relative path resolution."""

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "system.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load platform YAML configuration and reject an empty or malformed file."""
    selected_path = path or os.environ.get("AURUMMIND_CONFIG")
    config_path = Path(selected_path) if selected_path else DEFAULT_CONFIG
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {config_path}")
    return config


def resolve_path(value: str | Path) -> Path:
    """Resolve a configured path relative to the repository root."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path

"""Load project settings from config.yaml."""

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "config.yaml"
DEFAULT_DATABASE_URL = "postgresql+psycopg://backtester:backtester@localhost:5434/backtester"


def load_config(path: str | Path | None = None) -> dict:
    """Read config.yaml and return it as a dict."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path) as f:
        return yaml.safe_load(f)


def database_url() -> str:
    """Postgres URL. Set DATABASE_URL to override the docker-compose default."""
    return os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)

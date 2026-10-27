"""Shared building blocks for the Streamlit dashboard.

Bootstrap: put <repo>/src on sys.path so `import backtester` works without
`pip install -e .`. Streamlit Community Cloud runs `streamlit run
dashboard/app.py` from the repo root and only installs requirements.txt, so
the package isn't installed there. Every page imports `components` before
`backtester`, so this one spot covers the whole app.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

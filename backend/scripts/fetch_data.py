"""
CLI: rebuild data/raw/nifty100_{features,prices,volumes}.csv with yfinance.

Run from the backend/ folder:
    python -m scripts.fetch_data
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.data.fetch import fetch_all  # noqa: E402

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    summary = fetch_all()
    print(f"Wrote {summary['stocks']} stocks to {summary['out_dir']}")
    for d in summary["dropped"]:
        print(f"  dropped {d['ticker']}: {d['reason']}")

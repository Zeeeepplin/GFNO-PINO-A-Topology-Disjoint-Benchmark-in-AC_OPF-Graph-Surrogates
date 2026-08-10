"""Regenerate manuscript figures and tables from the complete revision runs."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aggregate_revision_results import main  # noqa: E402

if __name__ == "__main__":
    main(
        [
            "--results-root",
            str(ROOT / "results" / "revision"),
            "--runs-root",
            str(ROOT / "runs" / "revision"),
            "--data-root",
            str(ROOT / "data" / "pilot"),
            "--output-dir",
            str(ROOT / "results" / "revision" / "aggregate"),
            "--figure-dir",
            str(ROOT / "manuscript" / "figures"),
            "--generated-tex-dir",
            str(ROOT / "manuscript" / "generated"),
        ]
    )

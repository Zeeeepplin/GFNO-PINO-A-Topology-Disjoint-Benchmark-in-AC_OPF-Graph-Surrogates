"""Build a clean, deterministic submission/reproducibility ZIP archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections.abc import Iterable
from pathlib import Path

ROOT_FILES = (
    ".gitignore",
    "README.md",
    "DESIGN.md",
    "environment.yml",
    "requirements.txt",
    "requirements-lock.txt",
    "pyproject.toml",
    "train.py",
    "evaluate.py",
    "run_revision_benchmark.py",
    "aggregate_revision_results.py",
    "aggregate_results.py",
    "analyze_active_bounds.py",
    "reproduce_submission.py",
    "build_artifact_manifest.py",
    "build_submission_archive.py",
)
ROOT_DIRS = (
    "configs",
    "data/pilot",
    "data_gen",
    "julia",
    "losses",
    "models",
    "notebooks",
    "results/revision",
    "runs/revision",
    "tests",
    "utils",
    "manuscript",
)
EXCLUDED_PARTS = {
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "tensorboard",
}
EXCLUDED_SUFFIXES = {
    ".aux",
    ".bbl",
    ".blg",
    ".fdb_latexmk",
    ".fls",
    ".log",
    ".synctex.gz",
}


def _files(root: Path) -> list[Path]:
    candidates = [root / item for item in ROOT_FILES]
    for item in ROOT_DIRS:
        candidates.extend(path for path in (root / item).rglob("*") if path.is_file())
    selected = []
    for path in candidates:
        relative = path.relative_to(root)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if any(path.name.endswith(suffix) for suffix in EXCLUDED_SUFFIXES):
            continue
        selected.append(path)
    missing = [path for path in selected if not path.exists()]
    if missing:
        raise FileNotFoundError(missing[0])
    return sorted(set(selected), key=lambda value: value.relative_to(root).as_posix())


def build(root: Path, output: Path) -> dict[str, object]:
    """Create an archive with fixed timestamps and normalized POSIX paths."""
    output.parent.mkdir(parents=True, exist_ok=True)
    files = _files(root)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            relative = path.relative_to(root).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(2026, 7, 28, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compresslevel=6)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    summary = {
        "archive": str(output),
        "file_count": len(files),
        "bytes": output.stat().st_size,
        "sha256": digest,
    }
    output.with_suffix(".sha256.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/submission/topology_conditioned_gfno_pino_submission.zip"),
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(build(args.root.resolve(), args.output.resolve()), indent=2))

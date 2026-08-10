"""Write a deterministic SHA-256 inventory for the reproducibility package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

DEFAULT_PATHS = (
    "configs",
    "data/pilot",
    "runs/revision",
    "results/revision",
    "manuscript/generated",
    "manuscript/figures",
    "manuscript/main.tex",
    "manuscript/main.pdf",
    "manuscript/references.bib",
    "manuscript/cover_letter.tex",
    "manuscript/cover_letter.pdf",
    "manuscript/framing_revision_summary.md",
    "manuscript/final_revision_audit.md",
    "manuscript/response_to_reviewers.tex",
    "manuscript/response_to_reviewers.pdf",
    "tests",
    "data_gen",
    "losses",
    "models",
    "utils",
    "julia/Project.toml",
    "julia/Manifest.toml",
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
    "analyze_active_bounds.py",
    "reproduce_submission.py",
    "build_artifact_manifest.py",
    "build_submission_archive.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(root: Path, paths: Iterable[str], output: Path) -> list[dict[str, object]]:
    """Hash every regular file below ``paths`` except the manifest itself."""
    output_resolved = output.resolve()
    files: set[Path] = set()
    for item in paths:
        candidate = root / item
        if candidate.is_file():
            files.add(candidate)
        elif candidate.is_dir():
            files.update(path for path in candidate.rglob("*") if path.is_file())
        else:
            raise FileNotFoundError(candidate)
    records: list[dict[str, object]] = []
    for path in sorted(files, key=lambda value: value.as_posix()):
        if path.resolve() == output_resolved or path.name == "artifact_manifest.json":
            continue
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "algorithm": "SHA-256",
        "root": ".",
        "file_count": len(records),
        "total_bytes": sum(int(record["bytes"]) for record in records),
        "csv": output.relative_to(root).as_posix(),
        "csv_sha256": sha256(output),
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return records


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/revision/aggregate/artifact_manifest.csv"),
    )
    parser.add_argument("--paths", nargs="+", default=list(DEFAULT_PATHS))
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    records = build_manifest(args.root.resolve(), args.paths, args.output.resolve())
    print(json.dumps({"files_hashed": len(records), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()

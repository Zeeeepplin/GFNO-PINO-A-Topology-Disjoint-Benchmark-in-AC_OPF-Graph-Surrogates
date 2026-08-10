"""Reproduce tests, five-seed benchmark, paper artifacts, and hash inventory."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from pathlib import Path


def run(command: list[str], *, cwd: Path, report: Path | None = None) -> None:
    """Run one stage and optionally persist its combined terminal output."""
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE if report else None,
        stderr=subprocess.STDOUT if report else None,
    )
    if report is not None:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(completed.stdout, encoding="utf-8")
        print(completed.stdout, end="")
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, command)


def reproduce(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parent
    python = sys.executable
    aggregate = root / "results" / "revision" / "aggregate"
    run(
        [python, "-m", "pytest"],
        cwd=root,
        report=aggregate / "test_report.txt",
    )
    run(
        [python, "-m", "data_gen.refresh_topology_artifacts", "--config-root", "configs/pilot"],
        cwd=root,
    )
    benchmark = [python, "run_revision_benchmark.py", "--device", args.device]
    if args.force:
        benchmark.append("--force")
    run(benchmark, cwd=root)
    run([python, "aggregate_revision_results.py"], cwd=root)
    run([python, "analyze_active_bounds.py", "--device", args.device], cwd=root)
    if not args.skip_pdf:
        latexmk = shutil.which("latexmk")
        tectonic = shutil.which("tectonic")
        if latexmk:
            command = [
                latexmk,
                "-pdf",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "main.tex",
            ]
        elif tectonic:
            command = [tectonic, "--keep-logs", "--keep-intermediates", "main.tex"]
        else:
            raise FileNotFoundError(
                "neither latexmk nor tectonic is on PATH; rerun with --skip-pdf "
                "or install a TeX engine"
            )
        run(command, cwd=root / "manuscript")
        run([*command[:-1], "response_to_reviewers.tex"], cwd=root / "manuscript")
        # Compile the cover letter in an isolated directory so a stale
        # auxiliary file from a different document class cannot break a clean
        # reproduction.
        with tempfile.TemporaryDirectory(prefix="gfno_cover_") as temporary:
            temporary_path = Path(temporary)
            shutil.copy2(
                root / "manuscript" / "cover_letter.tex",
                temporary_path / "cover_letter.tex",
            )
            run([*command[:-1], "cover_letter.tex"], cwd=temporary_path)
            shutil.copy2(
                temporary_path / "cover_letter.pdf",
                root / "manuscript" / "cover_letter.pdf",
            )
        pdf_output = root / "output" / "pdf"
        pdf_output.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            root / "manuscript" / "main.pdf",
            pdf_output / "topology_conditioned_gfno_pino_submission.pdf",
        )
        shutil.copy2(
            root / "manuscript" / "response_to_reviewers.pdf",
            pdf_output / "response_to_technical_review.pdf",
        )
        shutil.copy2(
            root / "manuscript" / "cover_letter.pdf",
            pdf_output / "cover_letter.pdf",
        )
    run([python, "build_artifact_manifest.py"], cwd=root)
    run([python, "build_submission_archive.py"], cwd=root)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true", help="retrain completed runs")
    parser.add_argument("--skip-pdf", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    reproduce(parse_args())

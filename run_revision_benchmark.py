"""Run the reviewer-requested multi-seed benchmark and all evaluations."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
import yaml

from evaluate import evaluate_checkpoint
from train import train

DEFAULT_CASES = ("case30", "case57", "case118", "case300")
DEFAULT_MODELS = (
    "fixed_pinn",
    "topology_blind_gfno",
    "gnn",
    "data_only_gfno",
    "pino",
)
DEFAULT_SEEDS = (2026, 2027, 2028, 2029, 2030)


def run(args: argparse.Namespace) -> Path:
    """Execute or resume every requested case/model/seed combination."""
    args.run_root.mkdir(parents=True, exist_ok=True)
    args.result_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for case in args.cases:
        config_path = args.config_root / f"{case}.yaml"
        base_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        for model in args.models:
            for seed in args.seeds:
                config = json.loads(json.dumps(base_config))
                config["benchmark_tier"] = "review-revision-multiseed"
                config["training"]["seed"] = int(seed)
                run_dir = args.run_root / case / model / f"seed_{seed}"
                result_dir = args.result_root / case / model / f"seed_{seed}"
                checkpoint = run_dir / "best.pt"
                started = time.perf_counter()
                if args.force or not checkpoint.exists():
                    checkpoint = train(config, model, run_dir, args.device)
                if args.force or not (result_dir / "per_sample_metrics.csv").exists():
                    evaluate_checkpoint(
                        checkpoint,
                        result_dir,
                        args.device,
                        warmup=args.warmup,
                    )
                records.append(
                    {
                        "case": case,
                        "model": model,
                        "seed": seed,
                        "checkpoint": checkpoint.as_posix(),
                        "result_dir": result_dir.as_posix(),
                        "elapsed_s": time.perf_counter() - started,
                    }
                )
                print(json.dumps(records[-1]), flush=True)
                if args.device.type == "cuda":
                    torch.cuda.empty_cache()
    manifest_path = args.result_root / "benchmark_run_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return manifest_path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-root", type=Path, default=Path("configs/pilot"))
    parser.add_argument("--run-root", type=Path, default=Path("runs/revision"))
    parser.add_argument("--result-root", type=Path, default=Path("results/revision"))
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--device",
        type=torch.device,
        default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = run(args)
    print(json.dumps({"manifest": str(manifest)}, indent=2))


if __name__ == "__main__":
    main()

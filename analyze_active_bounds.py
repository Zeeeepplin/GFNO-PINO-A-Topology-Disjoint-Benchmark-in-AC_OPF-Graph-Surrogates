"""Stratify GFNO-PINO label error by active voltage/generator bounds.

A label is treated as near a bound when it is within 1e-4 p.u. of either
endpoint. The script reproduces ``results/revision/aggregate/
active_bound_analysis.csv`` from the archived test set and five checkpoints.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from train import build_model, move_batch, predict
from utils.dataset import OPFDataset

CASES = ("case30", "case57", "case118", "case300")
QUANTITIES = ("V", "P", "Q")


def analyze(root: Path, tolerance: float, device: torch.device) -> pd.DataFrame:
    """Return case-level active/inactive MAEs pooled across seeds and devices."""
    rows: list[dict[str, float | str]] = []
    for case in CASES:
        errors: dict[str, dict[str, list[float]]] = {
            key: defaultdict(list) for key in QUANTITIES
        }
        active_counts = dict.fromkeys(QUANTITIES, 0)
        total_counts = dict.fromkeys(QUANTITIES, 0)
        dataset = OPFDataset(root / "data" / "pilot" / case, "test")
        for seed in range(2026, 2031):
            checkpoint_path = (
                root / "runs" / "revision" / case / "pino" / f"seed_{seed}" / "best.pt"
            )
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )
            model = build_model("pino", checkpoint["config"], dataset[0]).to(device)
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
            with torch.no_grad():
                for raw_batch in DataLoader(dataset, batch_size=1, shuffle=False):
                    batch = move_batch(raw_batch, device)
                    prediction = predict(model, batch)
                    fields = (
                        (
                            "V",
                            prediction.vm,
                            batch["target_vm"],
                            batch["v_min"],
                            batch["v_max"],
                        ),
                        (
                            "P",
                            prediction.pg_pu,
                            batch["target_pg_pu"],
                            batch["p_min_pu"],
                            batch["p_max_pu"],
                        ),
                        (
                            "Q",
                            prediction.qg_pu,
                            batch["target_qg_pu"],
                            batch["q_min_pu"],
                            batch["q_max_pu"],
                        ),
                    )
                    for key, predicted, target, lower, upper in fields:
                        active = ((target - lower).abs() <= tolerance) | (
                            (target - upper).abs() <= tolerance
                        )
                        absolute_error = (predicted - target).abs()
                        active_counts[key] += int(active.sum())
                        total_counts[key] += active.numel()
                        errors[key]["active"].extend(
                            absolute_error[active].detach().cpu().tolist()
                        )
                        errors[key]["inactive"].extend(
                            absolute_error[~active].detach().cpu().tolist()
                        )
        row: dict[str, float | str] = {"case": case}
        for key in QUANTITIES:
            row[f"{key}_reference_near_bound_percent"] = (
                100.0 * active_counts[key] / total_counts[key]
            )
            row[f"{key}_active_mae_pu"] = float(np.mean(errors[key]["active"]))
            row[f"{key}_inactive_mae_pu"] = float(np.mean(errors[key]["inactive"]))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    output = args.root / "results" / "revision" / "aggregate" / "active_bound_analysis.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    analyze(args.root, args.tolerance, torch.device(args.device)).to_csv(output, index=False)
    print(output)


if __name__ == "__main__":
    main()

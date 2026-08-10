"""Evaluate neural AC-OPF models on held-out topologies and create paper tables."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from losses.physics_losses import (
    angle_difference_violation,
    branch_power_flows,
    economic_cost,
    generator_box_violation,
    power_balance_residual,
    thermal_violation,
    voltage_violation,
)
from train import build_model, move_batch, predict
from utils.dataset import OPFDataset

REFERENCE_COST_SCALE_CURRENCY_PER_HOUR = 1.0
POWER_BALANCE_TOLERANCE_PU = 1e-3
INEQUALITY_TOLERANCE = 1e-6


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def evaluate_checkpoint(
    checkpoint_path: Path,
    output_dir: Path,
    device: torch.device,
    warmup: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    dataset_root = Path(config["output_dir"]) / config["case"]
    dataset = OPFDataset(dataset_root, "test")
    model = build_model(checkpoint["model_kind"], config, dataset[0]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    first = move_batch(next(iter(loader)), device)
    for _ in range(warmup):
        predict(model, first)
    _synchronize(device)

    rows: list[dict[str, Any]] = []
    p_balance_by_sample: list[np.ndarray] = []
    q_balance_by_sample: list[np.ndarray] = []
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        _synchronize(device)
        started = time.perf_counter_ns()
        prediction = predict(model, batch)
        _synchronize(device)
        latency_ms = (time.perf_counter_ns() - started) / 1e6

        p_res, q_res = power_balance_residual(
            prediction.vm,
            prediction.va,
            prediction.pg_pu,
            prediction.qg_pu,
            batch["p_load_pu"],
            batch["q_load_pu"],
            batch["p_renewable_pu"],
            batch["ybus_pu"],
            batch["gen_bus"],
        )
        sf, st = branch_power_flows(
            prediction.vm,
            prediction.va,
            batch["y_from_pu"],
            batch["y_to_pu"],
            batch["branch_from"],
            batch["branch_to"],
        )
        thermal = thermal_violation(sf, st, batch["rate_pu"], batch["branch_mask"])
        voltage = voltage_violation(prediction.vm, batch["v_min"], batch["v_max"])
        generator = generator_box_violation(
            prediction.pg_pu,
            prediction.qg_pu,
            batch["p_min_pu"],
            batch["p_max_pu"],
            batch["q_min_pu"],
            batch["q_max_pu"],
        )
        angle = angle_difference_violation(
            prediction.va,
            batch["branch_from"],
            batch["branch_to"],
            batch["branch_angle_min_rad"],
            batch["branch_angle_max_rad"],
            batch["branch_status"],
        )
        predicted_cost = economic_cost(
            prediction.pg_pu * batch["base_mva"].unsqueeze(-1),
            batch["cost_coefficients"],
        )
        reference_cost = batch["target_objective"].item()
        reference_difference = (
            100.0
            * (predicted_cost.item() - reference_cost)
            / max(abs(reference_cost), REFERENCE_COST_SCALE_CURRENCY_PER_HOUR)
        )
        # Per-bus complex-power-balance mismatch magnitude. This is power in
        # per unit, not an electrical-current residual.
        balance = torch.sqrt(p_res.square() + q_res.square())
        p_balance_by_sample.append(p_res[0].detach().cpu().numpy())
        q_balance_by_sample.append(q_res[0].detach().cpu().numpy())
        voltage_rate = (voltage > INEQUALITY_TOLERANCE).float().mean().item()
        thermal_rate = (
            ((thermal > INEQUALITY_TOLERANCE).float() * batch["branch_mask"]).sum()
            / batch["branch_mask"].sum().clamp_min(1)
        ).item()
        generator_rate = (generator > INEQUALITY_TOLERANCE).float().mean().item()
        angle_rate = (
            ((angle > INEQUALITY_TOLERANCE).float() * batch["branch_status"]).sum()
            / batch["branch_status"].sum().clamp_min(1)
        ).item()
        balance_rate = (balance > POWER_BALANCE_TOLERANCE_PU).float().mean().item()
        ac_feasible_proxy = bool(
            balance.max().item() <= POWER_BALANCE_TOLERANCE_PU
            and voltage.max().item() <= INEQUALITY_TOLERANCE
            and thermal.max().item() <= INEQUALITY_TOLERANCE
            and generator.max().item() <= INEQUALITY_TOLERANCE
            and angle.max().item() <= INEQUALITY_TOLERANCE
        )
        rows.append(
            {
                "sample_id": batch["sample_id"][0],
                "topology_id": batch["topology_id"][0],
                "case": config["case"],
                "model_kind": checkpoint["model_kind"],
                "benchmark_tier": config.get("benchmark_tier", "publication"),
                "solver_backend": config.get("solver", {}).get("backend", "pandapower"),
                "n_bus": prediction.vm.shape[-1],
                "signed_reference_cost_difference_percent": reference_difference,
                "absolute_reference_cost_difference_percent": abs(reference_difference),
                "feasible_reference_cost_difference_percent": (
                    reference_difference if ac_feasible_proxy else float("nan")
                ),
                "predicted_cost": predicted_cost.item(),
                "reference_cost": reference_cost,
                "power_balance_mean_pu": balance.mean().item(),
                "power_balance_max_pu": balance.max().item(),
                "power_balance_violation_rate": balance_rate,
                "p_balance_mean_pu": p_res.abs().mean().item(),
                "p_balance_max_pu": p_res.abs().max().item(),
                "q_balance_mean_pu": q_res.abs().mean().item(),
                "q_balance_max_pu": q_res.abs().max().item(),
                "voltage_violation_rate": voltage_rate,
                "voltage_violation_max_pu": voltage.max().item(),
                "thermal_violation_rate": thermal_rate,
                "thermal_violation_max_pu": thermal.max().item(),
                "generator_violation_rate": generator_rate,
                "generator_violation_max_pu": generator.max().item(),
                "angle_violation_rate": angle_rate,
                "angle_violation_max_rad": angle.max().item(),
                "ac_feasible_proxy": float(ac_feasible_proxy),
                "inference_latency_ms": latency_ms,
                "classical_ac_latency_ms": 1000.0 * batch["solver_runtime_s"].item(),
                "classical_dc_latency_ms": 1000.0 * batch["dc_opf_runtime_s"].item(),
                "dc_reference_cost_difference_percent": 100.0
                * (batch["dc_opf_objective"].item() - reference_cost)
                / max(abs(reference_cost), REFERENCE_COST_SCALE_CURRENCY_PER_HOUR),
                "vm_mae": (prediction.vm - batch["target_vm"]).abs().mean().item(),
                "va_mae_rad": (prediction.va - batch["target_va"]).abs().mean().item(),
                "pg_mae_pu": (prediction.pg_pu - batch["target_pg_pu"]).abs().mean().item(),
                "qg_mae_pu": (prediction.qg_pu - batch["target_qg_pu"]).abs().mean().item(),
            }
        )

    per_sample = pd.DataFrame(rows)
    metric_columns = list(per_sample.select_dtypes(include=[np.number]).columns)
    summary_records: list[dict[str, Any]] = []
    for scope, frame in [
        ("all_unseen_topologies", per_sample),
        *[(f"topology:{name}", group) for name, group in per_sample.groupby("topology_id")],
    ]:
        record: dict[str, Any] = {
            "scope": scope,
            "case": config["case"],
            "model_kind": checkpoint["model_kind"],
            "benchmark_tier": config.get("benchmark_tier", "publication"),
            "solver_backend": config.get("solver", {}).get("backend", "pandapower"),
            "n_samples": len(frame),
            "n_topologies": frame["topology_id"].nunique(),
        }
        for column in metric_columns:
            record[f"{column}_mean"] = frame[column].mean()
            record[f"{column}_p95"] = frame[column].quantile(0.95)
            record[f"{column}_max"] = frame[column].max()
        summary_records.append(record)
    summary = pd.DataFrame(summary_records)

    output_dir.mkdir(parents=True, exist_ok=True)
    per_sample.to_csv(output_dir / "per_sample_metrics.csv", index=False)
    summary.to_csv(output_dir / "summary_metrics.csv", index=False)
    (output_dir / "evaluation_protocol.json").write_text(
        json.dumps(
            {
                "reference_cost_scale_currency_per_hour": (
                    REFERENCE_COST_SCALE_CURRENCY_PER_HOUR
                ),
                "power_balance_tolerance_pu": POWER_BALANCE_TOLERANCE_PU,
                "voltage_tolerance_pu": INEQUALITY_TOLERANCE,
                "thermal_tolerance_pu": INEQUALITY_TOLERANCE,
                "generator_tolerance_pu": INEQUALITY_TOLERANCE,
                "angle_tolerance_rad": INEQUALITY_TOLERANCE,
                "mean_bus_mismatch": (
                    "mean over buses within each sample, then mean over the "
                    "36 held-out samples within a seed"
                ),
                "reported_seed_aggregation": (
                    "compute each statistic within a seed, then report the "
                    "arithmetic mean and sample standard deviation across five seeds"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    np.savez_compressed(
        output_dir / "per_bus_power_balance_residuals.npz",
        sample_id=per_sample["sample_id"].to_numpy(dtype=str),
        topology_id=per_sample["topology_id"].to_numpy(dtype=str),
        p_balance_pu=np.stack(p_balance_by_sample),
        q_balance_pu=np.stack(q_balance_by_sample),
    )
    _plot_latency(
        per_sample,
        output_dir / "latency_comparison.png",
        checkpoint["model_kind"],
    )
    _plot_topology_generalization(
        per_sample, output_dir / "unseen_topology_power_balance.png"
    )
    return per_sample, summary


def _plot_latency(frame: pd.DataFrame, path: Path, model_kind: str) -> None:
    fig, axis = plt.subplots(figsize=(5.5, 3.5), constrained_layout=True)
    values = [
        frame["inference_latency_ms"].values,
        frame["classical_dc_latency_ms"].values,
        frame["classical_ac_latency_ms"].values,
    ]
    axis.boxplot(
        values,
        tick_labels=[model_kind.replace("_", " "), "DC-OPF", "Classical AC-OPF"],
        showfliers=False,
    )
    axis.set_yscale("log")
    axis.set_ylabel("Wall-clock latency (ms, log scale)")
    axis.grid(axis="y", alpha=0.25)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _plot_topology_generalization(frame: pd.DataFrame, path: Path) -> None:
    grouped = frame.groupby("topology_id", as_index=False)[
        "power_balance_max_pu"
    ].mean()
    grouped = grouped.sort_values("power_balance_max_pu")
    fig, axis = plt.subplots(figsize=(7.0, 3.5), constrained_layout=True)
    axis.plot(
        np.arange(len(grouped)),
        grouped["power_balance_max_pu"],
        marker=".",
        linewidth=1,
    )
    axis.set_xlabel("Held-out contingency (sorted)")
    axis.set_ylabel("Mean maximum power-balance mismatch (p.u.)")
    axis.grid(alpha=0.25)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def combine_scalability(results: list[Path], output_path: Path) -> pd.DataFrame:
    """Combine case30..300 summaries into the requested scalability curve."""
    frames = [pd.read_csv(path) for path in results]
    overall = pd.concat(
        [frame.loc[frame["scope"] == "all_unseen_topologies"] for frame in frames],
        ignore_index=True,
    ).sort_values("n_bus_mean")
    fig, latency_axis = plt.subplots(figsize=(6.0, 3.8), constrained_layout=True)
    residual_axis = latency_axis.twinx()
    latency_axis.plot(
        overall["n_bus_mean"],
        overall["inference_latency_ms_mean"],
        "o-",
        label="Inference latency",
    )
    residual_axis.plot(
        overall["n_bus_mean"],
        overall["power_balance_max_pu_mean"],
        "s--",
        color="tab:red",
        label="Power-balance mismatch",
    )
    latency_axis.set_xlabel("Bus count")
    latency_axis.set_ylabel("Inference latency (ms)")
    residual_axis.set_ylabel("Maximum power-balance mismatch (p.u.)")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return overall


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    per_sample, summary = evaluate_checkpoint(
        args.checkpoint, args.output_dir, torch.device(args.device), args.warmup
    )
    overall = summary.loc[summary["scope"] == "all_unseen_topologies"].iloc[0]
    print(
        json.dumps(
            {
                "n_samples": int(len(per_sample)),
                "n_topologies": int(per_sample.topology_id.nunique()),
                "mean_signed_reference_cost_difference_percent": overall[
                    "signed_reference_cost_difference_percent_mean"
                ],
                "mean_power_balance_max_pu": overall[
                    "power_balance_max_pu_mean"
                ],
                "mean_inference_latency_ms": overall["inference_latency_ms_mean"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

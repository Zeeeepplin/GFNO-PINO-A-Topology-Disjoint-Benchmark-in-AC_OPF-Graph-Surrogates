"""Train GFNO-PINO and ablations with supervised-to-physics curriculum."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from losses.physics_losses import AdaptiveLossBalancer, physics_informed_loss
from models.baselines import FixedTopologyPINN, PlainGNN, TopologyBlindGFNO
from models.pino_model import OPFPrediction, TopologyConditionedPINO
from utils.dataset import OPFDataset

MODEL_INPUTS = (
    "bus_features",
    "edge_features",
    "edge_index",
    "v_min",
    "v_max",
    "slack_mask",
    "gen_bus",
    "p_min_pu",
    "p_max_pu",
    "q_min_pu",
    "q_max_pu",
    "v_setpoint",
    "cost_coefficients",
    "base_mva",
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def predict(model: nn.Module, batch: Mapping[str, Any]) -> OPFPrediction:
    return model(**{name: batch[name] for name in MODEL_INPUTS})


def compute_losses(
    prediction: OPFPrediction,
    batch: Mapping[str, Tensor],
    include_supervised: bool,
) -> dict[str, Tensor]:
    targets = None
    if include_supervised:
        targets = {
            "vm": batch["target_vm"],
            "va": batch["target_va"],
            "pg_pu": batch["target_pg_pu"],
            "qg_pu": batch["target_qg_pu"],
        }
    output = physics_informed_loss(
        vm=prediction.vm,
        va=prediction.va,
        pg_pu=prediction.pg_pu,
        qg_pu=prediction.qg_pu,
        p_load_pu=batch["p_load_pu"],
        q_load_pu=batch["q_load_pu"],
        p_renewable_pu=batch["p_renewable_pu"],
        ybus_pu=batch["ybus_pu"],
        gen_bus=batch["gen_bus"],
        y_from_pu=batch["y_from_pu"],
        y_to_pu=batch["y_to_pu"],
        branch_from=batch["branch_from"],
        branch_to=batch["branch_to"],
        rate_pu=batch["rate_pu"],
        v_min=batch["v_min"],
        v_max=batch["v_max"],
        p_min_pu=batch["p_min_pu"],
        p_max_pu=batch["p_max_pu"],
        q_min_pu=batch["q_min_pu"],
        q_max_pu=batch["q_max_pu"],
        cost_coefficients=batch["cost_coefficients"],
        base_mva=batch["base_mva"],
        reference_objective=batch["target_objective"],
        economic_cost_scale=1.0,
        branch_angle_min_rad=batch["branch_angle_min_rad"],
        branch_angle_max_rad=batch["branch_angle_max_rad"],
        branch_status=batch["branch_status"],
        branch_mask=batch["branch_mask"],
        targets=targets,
    )
    return output.as_dict()


def gradnorm_shared_parameter(model: nn.Module) -> tuple[Tensor, str]:
    """Return the exact first shared lifting matrix used for GradNorm.

    GFNO and spatial-GNN variants share ``lift.weight`` across all downstream
    bus and generator outputs. The fixed-topology MLP uses
    ``network.0.weight``. Naming this tensor explicitly avoids a
    parameter-order-dependent GradNorm implementation.
    """
    if hasattr(model, "lift") and isinstance(model.lift, nn.Linear):
        return model.lift.weight, "lift.weight"
    if (
        hasattr(model, "network")
        and isinstance(model.network, nn.Sequential)
        and isinstance(model.network[0], nn.Linear)
    ):
        return model.network[0].weight, "network.0.weight"
    raise TypeError(f"no declared GradNorm shared layer for {type(model).__name__}")


def supervised_loss(prediction: OPFPrediction, batch: Mapping[str, Tensor]) -> Tensor:
    return torch.stack(
        [
            torch.mean((prediction.vm - batch["target_vm"]) ** 2),
            torch.mean((prediction.va - batch["target_va"]) ** 2),
            torch.mean((prediction.pg_pu - batch["target_pg_pu"]) ** 2),
            torch.mean((prediction.qg_pu - batch["target_qg_pu"]) ** 2),
        ]
    ).mean()


def build_model(kind: str, config: Mapping[str, Any], example: Mapping[str, Tensor]) -> nn.Module:
    model_config = dict(config["model"])
    if kind in {"pino", "data_only_gfno"}:
        return TopologyConditionedPINO(**model_config)
    if kind == "topology_blind_gfno":
        return TopologyBlindGFNO(
            fixed_edge_features=example["edge_features"],
            **model_config,
        )
    if kind == "gnn":
        return PlainGNN(**model_config)
    if kind == "fixed_pinn":
        return FixedTopologyPINN(
            n_bus=example["bus_features"].shape[-2],
            n_gen=example["gen_bus"].shape[-1],
            bus_channels=model_config.get("bus_channels", 8),
            width=model_config.get("width", 512),
        )
    raise ValueError(f"unknown model kind {kind!r}")


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        losses = compute_losses(predict(model, batch), batch, include_supervised=True)
        batch_size = batch["bus_features"].shape[0]
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + float(value) * batch_size
        count += batch_size
    return {name: total / count for name, total in totals.items()}


def train(config: dict[str, Any], model_kind: str, run_dir: Path, device: torch.device) -> Path:
    training = config["training"]
    seed_everything(int(training["seed"]))
    dataset_root = Path(config["output_dir"]) / config["case"]
    if model_kind == "fixed_pinn":
        # A fixed-topology PINN is fitted only on base-grid scenarios. Its
        # held-out contingency evaluation therefore measures the retraining
        # limitation rather than accidentally training across outages.
        base_data = OPFDataset(dataset_root, "train", topology_ids={"base"})
        validation_count = max(1, int(0.2 * len(base_data)))
        train_count = len(base_data) - validation_count
        if train_count < 1:
            raise ValueError("fixed_pinn requires at least two base-topology scenarios")
        train_data, validation_data = random_split(
            base_data,
            [train_count, validation_count],
            generator=torch.Generator().manual_seed(int(training["seed"])),
        )
    else:
        train_data = OPFDataset(dataset_root, "train")
        validation_data = OPFDataset(dataset_root, "val")
    loader_options = {
        "batch_size": int(training["batch_size"]),
        "num_workers": int(training.get("num_workers", 0)),
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_data, shuffle=True, **loader_options)
    validation_loader = DataLoader(validation_data, shuffle=False, **loader_options)
    example = train_data[0]
    model = build_model(model_kind, config, example).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(training["warm_start_epochs"]) + int(training["physics_epochs"]),
    )
    # Voltage and generator boxes are exact output reparameterizations and are
    # diagnostics only. Keeping an identically zero term in GradNorm would make
    # the relative training-rate definition ill-posed.
    loss_names = ("economic", "powerflow", "thermal", "angle", "supervised")
    all_loss_names = (
        "economic",
        "powerflow",
        "thermal",
        "angle",
        "voltage",
        "generator",
        "supervised",
    )
    balancer = AdaptiveLossBalancer(loss_names).to(device)
    balance_optimizer = torch.optim.Adam([balancer.log_weights], lr=0.01)
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(run_dir / "tensorboard")
    best_validation = float("inf")
    checkpoint_path = run_dir / "best.pt"
    global_step = 0
    total_epochs = int(training["warm_start_epochs"]) + int(training["physics_epochs"])
    training_started = time.perf_counter()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    shared, shared_parameter_name = gradnorm_shared_parameter(model)
    history: list[dict[str, Any]] = []

    for epoch in range(total_epochs):
        epoch_started = time.perf_counter()
        epoch_sums = {name: 0.0 for name in all_loss_names}
        epoch_total = 0.0
        epoch_samples = 0
        model.train()
        warm_start = epoch < int(training["warm_start_epochs"])
        physics_progress = (
            0.0
            if warm_start
            else (epoch - int(training["warm_start_epochs"]) + 1)
            / max(int(training["physics_epochs"]), 1)
        )
        powerflow_scale = 1.0 + 9.0 * physics_progress
        thermal_scale = 1.0 + 4.0 * physics_progress
        for raw_batch in train_loader:
            batch = move_batch(raw_batch, device)
            prediction = predict(model, batch)
            if warm_start or model_kind == "data_only_gfno":
                losses = compute_losses(prediction, batch, include_supervised=True)
                total = losses["supervised"]
            else:
                losses = compute_losses(prediction, batch, include_supervised=True)
                objective_losses = dict(losses)
                objective_losses["powerflow"] = losses["powerflow"] * powerflow_scale
                objective_losses["thermal"] = losses["thermal"] * thermal_scale
                balance_optimizer.zero_grad(set_to_none=True)
                gradnorm = balancer.gradnorm_objective(objective_losses, shared)
                gradnorm.backward(retain_graph=True)
                balance_optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                total = balancer.weighted_total(objective_losses)

            optimizer.zero_grad(set_to_none=True)
            total.backward()
            clip_grad_norm_(model.parameters(), float(training["grad_clip"]))
            optimizer.step()
            batch_size = int(batch["bus_features"].shape[0])
            epoch_total += float(total.detach()) * batch_size
            for name in all_loss_names:
                epoch_sums[name] += float(losses[name].detach()) * batch_size
            epoch_samples += batch_size
            writer.add_scalar("train/total", float(total.detach()), global_step)
            for name, value in losses.items():
                writer.add_scalar(f"train/{name}", float(value.detach()), global_step)
            if not warm_start and model_kind != "data_only_gfno":
                writer.add_scalar("annealing/powerflow_scale", powerflow_scale, global_step)
                writer.add_scalar("annealing/thermal_scale", thermal_scale, global_step)
                for name, weight in zip(
                    loss_names, balancer.positive_weights().detach(), strict=True
                ):
                    writer.add_scalar(f"weights/{name}", float(weight), global_step)
            global_step += 1
        scheduler.step()
        validation = validate(model, validation_loader, device)
        # Solver-warm-start accuracy prevents soft-constraint training from
        # selecting an apparently low-residual but economically implausible
        # dispatch. Power balance and thermal feasibility remain explicit in the score.
        validation_score = (
            5.0 * validation["supervised"]
            + validation["powerflow"]
            + validation["thermal"]
            + validation["angle"]
        )
        for name, value in validation.items():
            writer.add_scalar(f"validation/{name}", value, epoch)
        weights = {
            name: float(weight)
            for name, weight in zip(
                loss_names, balancer.positive_weights().detach().cpu(), strict=True
            )
        }
        history.append(
            {
                "epoch": epoch,
                "phase": "warm" if warm_start else "physics",
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train_total": epoch_total / max(epoch_samples, 1),
                **{
                    f"train_{name}": epoch_sums[name] / max(epoch_samples, 1)
                    for name in all_loss_names
                },
                **{f"validation_{name}": validation[name] for name in all_loss_names},
                **{f"weight_{name}": weights[name] for name in loss_names},
                "powerflow_scale": powerflow_scale,
                "thermal_scale": thermal_scale,
                "effective_weight_powerflow": weights["powerflow"] * powerflow_scale,
                "effective_weight_thermal": weights["thermal"] * thermal_scale,
                "epoch_duration_s": time.perf_counter() - epoch_started,
            }
        )
        if validation_score < best_validation:
            best_validation = validation_score
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "balancer_state": balancer.state_dict(),
                    "model_kind": model_kind,
                    "config": config,
                    "epoch": epoch,
                    "validation": validation,
                },
                checkpoint_path,
            )
        print(
            f"epoch={epoch:04d} phase={'warm' if warm_start else 'physics'} "
            f"val_supervised={validation['supervised']:.6g} "
            f"val_pf={validation['powerflow']:.6g}"
        )
    writer.close()
    history_path = run_dir / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as stream:
        writer_csv = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer_csv.writeheader()
        writer_csv.writerows(history)
    metadata = {
        "case": config["case"],
        "model_kind": model_kind,
        "benchmark_tier": config.get("benchmark_tier", "publication"),
        "solver_backend": config.get("solver", {}).get("backend", "pandapower"),
        "seed": int(training["seed"]),
        "device": str(device),
        "cuda_device": (torch.cuda.get_device_name(device) if device.type == "cuda" else None),
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "parameter_count": parameter_count,
        "gradnorm_shared_parameter": shared_parameter_name,
        "epochs": total_epochs,
        "best_validation_score": best_validation,
        "training_runtime_s": time.perf_counter() - training_started,
        "checkpoint": str(checkpoint_path),
        "history": str(history_path),
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return checkpoint_path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--model",
        choices=[
            "pino",
            "data_only_gfno",
            "gnn",
            "topology_blind_gfno",
            "fixed_pinn",
        ],
        default="pino",
    )
    parser.add_argument("--run-dir", type=Path, default=Path("runs"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    run_dir = args.run_dir / config["case"] / args.model
    checkpoint = train(config, args.model, run_dir, torch.device(args.device))
    print(json.dumps({"checkpoint": str(checkpoint), "device": args.device}))


if __name__ == "__main__":
    main()

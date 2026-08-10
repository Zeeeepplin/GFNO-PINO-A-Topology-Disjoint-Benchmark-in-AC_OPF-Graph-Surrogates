"""PyTorch dataset for separated topology and scenario artifacts."""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


def _tensor(array: np.ndarray, *, integer: bool = False) -> Tensor:
    if integer:
        return torch.from_numpy(np.asarray(array, dtype=np.int64))
    return torch.from_numpy(np.asarray(array, dtype=np.float32))


def _directional_edge_features(
    y_from: Tensor,
    y_to: Tensor,
    branch_from: Tensor,
    branch_to: Tensor,
    status: Tensor,
    rate_pu: Tensor,
) -> Tensor:
    """Extract ``Yff,Yft,Ytt,Ytf,status,rate`` from exact terminal operators."""
    row = torch.arange(branch_from.numel())
    yff = y_from[row, branch_from]
    yft = y_from[row, branch_to]
    ytt = y_to[row, branch_to]
    ytf = y_to[row, branch_from]
    return torch.stack(
        [
            yff.real,
            yff.imag,
            yft.real,
            yft.imag,
            ytt.real,
            ytt.imag,
            ytf.real,
            ytf.imag,
            status,
            rate_pu,
        ],
        dim=-1,
    )


@lru_cache(maxsize=128)
def _load_topology(path: str) -> dict[str, Tensor]:
    with np.load(path, allow_pickle=False) as payload:
        ybus = torch.complex(_tensor(payload["ybus_real"]), _tensor(payload["ybus_imag"]))
        y_from = torch.complex(_tensor(payload["y_from_real"]), _tensor(payload["y_from_imag"]))
        y_to = torch.complex(_tensor(payload["y_to_real"]), _tensor(payload["y_to_imag"]))
        branch_from = _tensor(payload["branch_from"], integer=True)
        branch_to = _tensor(payload["branch_to"], integer=True)
        bus_type_onehot = _tensor(payload["bus_type_onehot"])
        bus_type = bus_type_onehot.argmax(dim=-1)
        status = _tensor(payload["branch_status"])
        rate_pu = _tensor(payload["branch_rate_pu"])
        stored_edges = _tensor(payload["edge_features"])
        edge_features = (
            stored_edges
            if stored_edges.shape[-1] == 10
            else _directional_edge_features(
                y_from,
                y_to,
                branch_from,
                branch_to,
                status,
                rate_pu,
            )
        )
        default_angle = torch.full_like(status, 2.0 * torch.pi)
        return {
            "base_mva": _tensor(payload["base_mva"]),
            "ybus_pu": ybus,
            "y_from_pu": y_from,
            "y_to_pu": y_to,
            "branch_from": branch_from,
            "branch_to": branch_to,
            "edge_index": torch.stack([branch_from, branch_to]),
            "edge_features": edge_features,
            "rate_pu": rate_pu,
            "branch_status": status,
            # MATPOWER RATE_A == 0 means unconstrained, not a zero-MVA line.
            "branch_mask": _tensor(payload["branch_status"] * (payload["branch_rate_pu"] > 0)),
            "branch_angle_min_rad": (
                _tensor(payload["branch_angle_min_rad"])
                if "branch_angle_min_rad" in payload
                else -default_angle
            ),
            "branch_angle_max_rad": (
                _tensor(payload["branch_angle_max_rad"])
                if "branch_angle_max_rad" in payload
                else default_angle
            ),
            "branch_component_type": (
                _tensor(payload["branch_component_type"], integer=True)
                if "branch_component_type" in payload
                else torch.full_like(branch_from, 2)
            ),
            "physical_line_index": (
                _tensor(payload["physical_line_index"], integer=True)
                if "physical_line_index" in payload
                else torch.full_like(branch_from, -1)
            ),
            "v_min": _tensor(payload["v_min"]),
            "v_max": _tensor(payload["v_max"]),
            "slack_mask": bus_type.eq(3),
            "gen_bus": _tensor(payload["gen_bus"], integer=True),
            "p_min_pu": _tensor(payload["p_min_pu"]),
            "p_max_pu": _tensor(payload["p_max_pu"]),
            "q_min_pu": _tensor(payload["q_min_pu"]),
            "q_max_pu": _tensor(payload["q_max_pu"]),
            "v_setpoint": _tensor(payload["v_setpoint"]),
            "cost_coefficients": _tensor(payload["cost_coefficients"]),
        }


class OPFDataset(Dataset[dict[str, Any]]):
    """Dataset whose split membership is defined by entire topology IDs."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        topology_ids: set[str] | None = None,
    ) -> None:
        self.root = Path(root)
        manifest = self.root / "manifest.csv"
        if not manifest.exists():
            raise FileNotFoundError(f"dataset manifest does not exist: {manifest}")
        with manifest.open(newline="", encoding="utf-8") as stream:
            self.rows = [
                row
                for row in csv.DictReader(stream)
                if row["split"] == split
                and (topology_ids is None or row["topology_id"] in topology_ids)
            ]
        if not self.rows:
            raise ValueError(f"split {split!r} contains no samples in {manifest}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        topology_id = row["topology_id"]
        topology = _load_topology(str(self.root / "topologies" / f"{topology_id}.npz"))
        with np.load(self.root / row["relative_path"], allow_pickle=False) as sample:
            result: dict[str, Any] = {
                "bus_features": _tensor(sample["bus_features"]),
                "p_load_pu": _tensor(sample["p_load_pu"]),
                "q_load_pu": _tensor(sample["q_load_pu"]),
                "p_renewable_pu": _tensor(sample["p_renewable_pu"]),
                "target_vm": _tensor(sample["target_vm"]),
                "target_va": _tensor(sample["target_va"]),
                "target_pg_pu": _tensor(sample["target_pg_pu"]),
                "target_qg_pu": _tensor(sample["target_qg_pu"]),
                "target_objective": _tensor(sample["target_objective"]),
                "solver_runtime_s": _tensor(sample["solver_runtime_s"]),
                "dc_opf_runtime_s": _tensor(sample["dc_opf_runtime_s"]),
                "dc_opf_objective": _tensor(sample["dc_opf_objective"]),
                "dc_opf_converged": _tensor(sample["dc_opf_converged"]),
                "sample_id": row["sample_id"],
                "topology_id": topology_id,
            }
        result.update(topology)
        return result

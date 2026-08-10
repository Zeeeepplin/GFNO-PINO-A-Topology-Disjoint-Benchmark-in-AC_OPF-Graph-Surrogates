"""Fixed-dimensional PINN baseline that cannot transfer across grid topology/size."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from models.pino_model import OPFPrediction


class FixedTopologyPINN(nn.Module):
    """MLP PINN baseline for one fixed bus/generator configuration.

    It intentionally has no edge or Ybus input. A contingency therefore
    requires retraining (or produces a distribution shift), matching the
    limitation the topology-conditioned surrogate is designed to address.
    """

    def __init__(self, n_bus: int, n_gen: int, bus_channels: int = 8, width: int = 512) -> None:
        super().__init__()
        self.n_bus = n_bus
        self.n_gen = n_gen
        output_size = 2 * n_bus + 2 * n_gen
        self.network = nn.Sequential(
            nn.Linear(n_bus * bus_channels, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, output_size),
        )

    def forward(self, **batch: Tensor) -> OPFPrediction:
        features = batch["bus_features"]
        if features.shape[1] != self.n_bus:
            raise ValueError("FixedTopologyPINN cannot accept a different bus count")
        raw = self.network(features.flatten(start_dim=1))
        vm_raw, va_raw, pg_raw, qg_raw = torch.split(
            raw, [self.n_bus, self.n_bus, self.n_gen, self.n_gen], dim=-1
        )
        vm = batch["v_min"] + (batch["v_max"] - batch["v_min"]) * torch.sigmoid(vm_raw)
        slack_index = batch["slack_mask"].long().argmax(dim=-1, keepdim=True)
        va = va_raw - va_raw.gather(-1, slack_index)
        pg = batch["p_min_pu"] + (batch["p_max_pu"] - batch["p_min_pu"]) * torch.sigmoid(pg_raw)
        qg = batch["q_min_pu"] + (batch["q_max_pu"] - batch["q_min_pu"]) * torch.sigmoid(qg_raw)
        return OPFPrediction(vm, va, pg, qg)

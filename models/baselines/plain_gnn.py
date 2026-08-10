"""Spatial message-passing GNN ablation without a spectral polynomial filter."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from models.gfno_layers import EdgeConditioner
from models.pino_model import (
    HardConstraintHead,
    OPFPrediction,
    normalized_generator_features,
)


class SpatialMessageBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.self_linear = nn.Linear(width, width)
        self.neighbor_linear = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)

    def forward(self, features: Tensor, edge_index: Tensor, edge_status: Tensor) -> Tensor:
        if edge_index.ndim == 2:
            edge_index = edge_index.unsqueeze(0).expand(features.shape[0], -1, -1)
        source, target = edge_index[:, 0], edge_index[:, 1]
        source_index = source.unsqueeze(-1).expand(-1, -1, features.shape[-1])
        target_index = target.unsqueeze(-1).expand(-1, -1, features.shape[-1])
        source_features = features.gather(1, source_index) * edge_status.unsqueeze(-1)
        target_features = features.gather(1, target_index) * edge_status.unsqueeze(-1)
        messages = torch.zeros_like(features)
        messages.scatter_add_(1, target_index, source_features)
        messages.scatter_add_(1, source_index, target_features)
        degree = torch.zeros_like(features[..., :1])
        degree.scatter_add_(1, target.unsqueeze(-1), edge_status.unsqueeze(-1))
        degree.scatter_add_(1, source.unsqueeze(-1), edge_status.unsqueeze(-1))
        update = F.gelu(
            self.self_linear(features) + self.neighbor_linear(messages / degree.clamp_min(1.0))
        )
        return self.norm(features + update)


class PlainGNN(nn.Module):
    """Topology-conditioned spatial GNN with the same constrained output head."""

    def __init__(
        self,
        bus_channels: int = 8,
        edge_channels: int = 10,
        width: int = 128,
        depth: int = 6,
        **_: object,
    ) -> None:
        super().__init__()
        self.edge_conditioner = EdgeConditioner(edge_channels, width)
        self.lift = nn.Linear(bus_channels + width, width)
        self.blocks = nn.ModuleList([SpatialMessageBlock(width) for _ in range(depth)])
        self.head = HardConstraintHead(width)

    def forward(self, **batch: Tensor) -> OPFPrediction:
        bus_features = batch["bus_features"]
        edge_features = batch["edge_features"]
        edge_index = batch["edge_index"]
        latent = self.lift(
            torch.cat(
                [
                    bus_features,
                    self.edge_conditioner(edge_features, edge_index, bus_features.shape[1]),
                ],
                dim=-1,
            )
        )
        for block in self.blocks:
            latent = block(latent, edge_index, edge_features[..., -2])
        generator_features = normalized_generator_features(
            p_min_pu=batch["p_min_pu"],
            p_max_pu=batch["p_max_pu"],
            q_min_pu=batch["q_min_pu"],
            q_max_pu=batch["q_max_pu"],
            v_setpoint=batch["v_setpoint"],
            cost_coefficients=batch["cost_coefficients"],
            base_mva=batch["base_mva"],
        )
        return self.head(
            latent,
            v_min=batch["v_min"],
            v_max=batch["v_max"],
            slack_mask=batch["slack_mask"],
            gen_bus=batch["gen_bus"],
            gen_features=generator_features,
            p_min_pu=batch["p_min_pu"],
            p_max_pu=batch["p_max_pu"],
            q_min_pu=batch["q_min_pu"],
            q_max_pu=batch["q_max_pu"],
        )

"""Topology-conditioned physics-informed Chebyshev graph surrogate.

``GFNO-PINO`` is retained as the experiment identifier. Separate checkpoints
are trained per grid size, so the implementation does not claim
discretization-transfer solely from this architecture.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from models.gfno_layers import EdgeConditioner, GFNOBlock, normalized_laplacian


@dataclass(frozen=True)
class OPFPrediction:
    vm: Tensor
    va: Tensor
    pg_pu: Tensor
    qg_pu: Tensor


def normalized_generator_features(
    *,
    p_min_pu: Tensor,
    p_max_pu: Tensor,
    q_min_pu: Tensor,
    q_max_pu: Tensor,
    v_setpoint: Tensor,
    cost_coefficients: Tensor,
    base_mva: Tensor,
) -> Tensor:
    """Build dimensionless generator features, including economics.

    The MW polynomial is converted to coefficients for a per-unit active-power
    variable and divided by one positive case-level cost scale. A normalized
    generator ordinal distinguishes otherwise identical co-located units.
    """
    base = base_mva.to(p_min_pu.dtype).reshape(-1, 1)
    c2, c1, c0 = cost_coefficients.unbind(dim=-1)
    per_unit_cost = torch.stack(
        [c2 * base.square(), c1 * base, c0],
        dim=-1,
    )
    cost_scale = per_unit_cost.abs().sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
    normalized_cost = per_unit_cost / cost_scale
    n_gen = p_min_pu.shape[-1]
    ordinal = (
        torch.arange(n_gen, device=p_min_pu.device, dtype=p_min_pu.dtype)
        .div(max(n_gen - 1, 1))
        .reshape(1, n_gen, 1)
        .expand(p_min_pu.shape[0], -1, -1)
    )
    return torch.cat(
        [
            torch.stack(
                [p_min_pu, p_max_pu, q_min_pu, q_max_pu, v_setpoint],
                dim=-1,
            ),
            normalized_cost,
            ordinal,
        ],
        dim=-1,
    )


class HardConstraintHead(nn.Module):
    """Enforce independent box limits and the slack-angle reference exactly.

    These transformations cannot enforce coupled nodal-balance or MVA constraints.
    Calling their output "AC feasible" without residual checks would therefore
    be incorrect; the evaluator always reports those remaining violations.
    """

    def __init__(self, width: int, generator_features: int = 9) -> None:
        super().__init__()
        self.bus_head = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 2))
        self.gen_head = nn.Sequential(
            nn.Linear(width + generator_features, width),
            nn.GELU(),
            nn.Linear(width, 2),
        )

    def forward(
        self,
        latent: Tensor,
        *,
        v_min: Tensor,
        v_max: Tensor,
        slack_mask: Tensor,
        gen_bus: Tensor,
        gen_features: Tensor,
        p_min_pu: Tensor,
        p_max_pu: Tensor,
        q_min_pu: Tensor,
        q_max_pu: Tensor,
    ) -> OPFPrediction:
        raw_bus = self.bus_head(latent)
        vm = v_min + (v_max - v_min) * torch.sigmoid(raw_bus[..., 0])
        raw_angle = raw_bus[..., 1]
        slack_index = slack_mask.to(torch.long).argmax(dim=-1, keepdim=True)
        slack_angle = raw_angle.gather(-1, slack_index)
        va = raw_angle - slack_angle

        gather_index = gen_bus.unsqueeze(-1).expand(-1, -1, latent.shape[-1])
        generator_latent = latent.gather(1, gather_index)
        raw_gen = self.gen_head(torch.cat([generator_latent, gen_features], dim=-1))
        pg = p_min_pu + (p_max_pu - p_min_pu) * torch.sigmoid(raw_gen[..., 0])
        qg = q_min_pu + (q_max_pu - q_min_pu) * torch.sigmoid(raw_gen[..., 1])
        return OPFPrediction(vm=vm, va=va, pg_pu=pg, qg_pu=qg)


class TopologyConditionedPINO(nn.Module):
    """GFNO mapping bus/branch fields to bounded AC-OPF decision variables."""

    def __init__(
        self,
        bus_channels: int = 8,
        edge_channels: int = 10,
        width: int = 128,
        depth: int = 6,
        chebyshev_order: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.edge_conditioner = EdgeConditioner(edge_channels, width)
        self.lift = nn.Linear(bus_channels + width, width)
        self.blocks = nn.ModuleList(
            [GFNOBlock(width, chebyshev_order, dropout) for _ in range(depth)]
        )
        self.head = HardConstraintHead(width)

    def forward(
        self,
        *,
        bus_features: Tensor,
        edge_features: Tensor,
        edge_index: Tensor,
        v_min: Tensor,
        v_max: Tensor,
        slack_mask: Tensor,
        gen_bus: Tensor,
        p_min_pu: Tensor,
        p_max_pu: Tensor,
        q_min_pu: Tensor,
        q_max_pu: Tensor,
        v_setpoint: Tensor,
        cost_coefficients: Tensor,
        base_mva: Tensor,
        node_mask: Tensor | None = None,
    ) -> OPFPrediction:
        n_nodes = bus_features.shape[1]
        edge_context = self.edge_conditioner(edge_features, edge_index, n_nodes)
        latent = self.lift(torch.cat([bus_features, edge_context], dim=-1))
        physical_weight = (
            0.5
            * (
                torch.linalg.vector_norm(edge_features[..., 2:4], dim=-1)
                + torch.linalg.vector_norm(edge_features[..., 6:8], dim=-1)
            )
            * edge_features[..., -2]
        )
        laplacian = normalized_laplacian(edge_index, physical_weight, n_nodes, node_mask)
        for block in self.blocks:
            latent = block(latent, laplacian, node_mask)
        generator_features = normalized_generator_features(
            p_min_pu=p_min_pu,
            p_max_pu=p_max_pu,
            q_min_pu=q_min_pu,
            q_max_pu=q_max_pu,
            v_setpoint=v_setpoint,
            cost_coefficients=cost_coefficients,
            base_mva=base_mva,
        )
        return self.head(
            latent,
            v_min=v_min,
            v_max=v_max,
            slack_mask=slack_mask,
            gen_bus=gen_bus,
            gen_features=generator_features,
            p_min_pu=p_min_pu,
            p_max_pu=p_max_pu,
            q_min_pu=q_min_pu,
            q_max_pu=q_max_pu,
        )

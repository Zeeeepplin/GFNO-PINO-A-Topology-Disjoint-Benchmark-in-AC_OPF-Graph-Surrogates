"""Matched GFNO baseline that receives no sample-specific topology."""

from __future__ import annotations

from torch import Tensor

from models.pino_model import OPFPrediction, TopologyConditionedPINO


class TopologyBlindGFNO(TopologyConditionedPINO):
    """Use one fixed graph/edge field while training on all topology rows.

    The train/validation rows, architecture, optimizer, and physics losses
    match GFNO-PINO. Only the sample-specific topology input is removed.
    """

    def __init__(self, *, fixed_edge_features: Tensor, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.register_buffer(
            "fixed_edge_features",
            fixed_edge_features.detach().clone(),
            persistent=True,
        )

    def forward(self, **batch: Tensor) -> OPFPrediction:
        fixed = self.fixed_edge_features
        if fixed.ndim == 2:
            fixed = fixed.unsqueeze(0)
        fixed = fixed.expand(batch["bus_features"].shape[0], -1, -1)
        return super().forward(**{**batch, "edge_features": fixed})

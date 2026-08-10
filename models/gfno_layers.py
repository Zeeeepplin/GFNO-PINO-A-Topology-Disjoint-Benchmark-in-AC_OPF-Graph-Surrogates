"""Chebyshev graph-spectral layers for topology-conditioned prediction.

Chebyshev filters approximate a spectral multiplier without explicitly
computing eigenvectors. This is essential here: N-1 outages change Laplacian
eigenvectors, whose sign/rotation ambiguity would otherwise make Fourier
coefficients inconsistent across topology samples. The layer is structurally
a localized ChebNet convolution; it does not by itself establish
discretization-transfer or a neural-operator property.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def normalized_laplacian(
    edge_index: Tensor,
    edge_weight: Tensor,
    n_nodes: int,
    node_mask: Tensor | None = None,
) -> Tensor:
    """Construct a batched symmetric normalized graph Laplacian.

    Args:
        edge_index: ``(2, E)`` or ``(B, 2, E)`` integer terminal indices.
        edge_weight: Non-negative active edge weights, ``(E,)`` or ``(B,E)``.
        n_nodes: Padded node dimension.
        node_mask: Optional ``(B,N)`` mask. Padded rows/columns remain zero.

    The physical weight is normally ``status * abs(G + jB)``. Thus the
    contingency is an explicit input that changes the polynomial operator.
    """
    if edge_weight.ndim == 1:
        edge_weight = edge_weight.unsqueeze(0)
    batch_size, n_edges = edge_weight.shape
    if edge_index.ndim == 2:
        edge_index = edge_index.unsqueeze(0).expand(batch_size, -1, -1)
    if edge_index.shape != (batch_size, 2, n_edges):
        raise ValueError(
            f"edge_index must have shape ({batch_size}, 2, {n_edges}), got {edge_index.shape}"
        )
    adjacency = torch.zeros(
        batch_size,
        n_nodes,
        n_nodes,
        dtype=edge_weight.dtype,
        device=edge_weight.device,
    )
    sources, targets = edge_index[:, 0], edge_index[:, 1]
    batch = torch.arange(batch_size, device=edge_weight.device)[:, None]
    adjacency.index_put_((batch, sources, targets), edge_weight, accumulate=True)
    adjacency.index_put_((batch, targets, sources), edge_weight, accumulate=True)
    degree = adjacency.sum(dim=-1)
    inverse_sqrt_degree = degree.clamp_min(1e-12).rsqrt()
    normalized_adjacency = (
        inverse_sqrt_degree.unsqueeze(-1) * adjacency * inverse_sqrt_degree.unsqueeze(-2)
    )
    identity = torch.eye(n_nodes, dtype=adjacency.dtype, device=adjacency.device)
    laplacian = identity.unsqueeze(0) - normalized_adjacency
    if node_mask is not None:
        valid = node_mask.to(laplacian.dtype)
        laplacian = laplacian * valid.unsqueeze(-1) * valid.unsqueeze(-2)
    return laplacian


class ChebyshevSpectralConv(nn.Module):
    """Learned K-order polynomial spectral graph convolution."""

    def __init__(self, in_channels: int, out_channels: int, order: int) -> None:
        super().__init__()
        if order < 1:
            raise ValueError("Chebyshev order must be at least one")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.order = order
        self.weight = nn.Parameter(torch.empty(order, in_channels, out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features: Tensor, laplacian: Tensor) -> Tensor:
        """Apply the spectral filter to ``(B,N,C)`` features.

        The normalized Laplacian spectrum is in ``[0,2]``. Therefore
        ``L_tilde = L - I`` maps it to ``[-1,1]`` without a per-topology
        eigenvalue computation.
        """
        n_nodes = features.shape[-2]
        identity = torch.eye(n_nodes, dtype=laplacian.dtype, device=laplacian.device).unsqueeze(0)
        scaled_laplacian = laplacian - identity
        terms = [features]
        if self.order > 1:
            terms.append(torch.matmul(scaled_laplacian, features))
        for _ in range(2, self.order):
            terms.append(2.0 * torch.matmul(scaled_laplacian, terms[-1]) - terms[-2])
        stacked = torch.stack(terms, dim=1)
        return torch.einsum("bkni,kio->bno", stacked, self.weight) + self.bias


class GFNOBlock(nn.Module):
    """Chebyshev spectral path plus pointwise path and GELU."""

    def __init__(
        self,
        width: int,
        chebyshev_order: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.spectral = ChebyshevSpectralConv(width, width, chebyshev_order)
        self.pointwise = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, features: Tensor, laplacian: Tensor, node_mask: Tensor | None = None
    ) -> Tensor:
        update = F.gelu(self.spectral(features, laplacian) + self.pointwise(features))
        output = self.norm(features + self.dropout(update))
        if node_mask is not None:
            output = output * node_mask.unsqueeze(-1).to(output.dtype)
        return output


class EdgeConditioner(nn.Module):
    """Encode directional branch physics with endpoint-specific messages.

    ``edge_features`` contains ``[Yff, Yft, Ytt, Ytf]`` as real/imaginary
    pairs followed by status and rating. Separate from/to encoders prevent a
    tap-changing or phase-shifting transformer from being treated as an
    undirected line.
    """

    def __init__(self, edge_channels: int, width: int) -> None:
        super().__init__()
        self.from_encoder = nn.Sequential(
            nn.Linear(edge_channels, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.to_encoder = nn.Sequential(
            nn.Linear(edge_channels, width),
            nn.GELU(),
            nn.Linear(width, width),
        )

    def forward(
        self,
        edge_features: Tensor,
        edge_index: Tensor,
        n_nodes: int,
    ) -> Tensor:
        if edge_index.ndim == 2:
            edge_index = edge_index.unsqueeze(0).expand(edge_features.shape[0], -1, -1)
        # Zero status must remove *all* learned edge influence, including MLP
        # biases; otherwise the contingency is not faithfully represented.
        status = edge_features[..., -2:-1]
        from_encoded = self.from_encoder(edge_features) * status
        to_encoded = self.to_encoder(edge_features) * status
        output = torch.zeros(
            edge_features.shape[0],
            n_nodes,
            from_encoded.shape[-1],
            dtype=from_encoded.dtype,
            device=from_encoded.device,
        )
        source = edge_index[:, 0].unsqueeze(-1).expand_as(from_encoded)
        target = edge_index[:, 1].unsqueeze(-1).expand_as(to_encoded)
        output.scatter_add_(1, source, from_encoded)
        output.scatter_add_(1, target, to_encoded)
        degree = torch.zeros(
            edge_features.shape[0],
            n_nodes,
            1,
            dtype=from_encoded.dtype,
            device=from_encoded.device,
        )
        degree.scatter_add_(1, edge_index[:, 0].unsqueeze(-1), status)
        degree.scatter_add_(1, edge_index[:, 1].unsqueeze(-1), status)
        return output / degree.clamp_min(1.0)

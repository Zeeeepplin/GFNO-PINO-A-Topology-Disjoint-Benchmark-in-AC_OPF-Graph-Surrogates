from __future__ import annotations

import torch
from torch import nn

from losses.physics_losses import branch_power_flows
from models.gfno_layers import ChebyshevSpectralConv, EdgeConditioner, normalized_laplacian
from models.pino_model import TopologyConditionedPINO


def test_laplacian_changes_when_line_is_outaged() -> None:
    edge_index = torch.tensor([[0, 1, 0], [1, 2, 2]])
    weights = torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 1.0]])
    laplacian = normalized_laplacian(edge_index, weights, n_nodes=3)
    assert not torch.allclose(laplacian[0], laplacian[1])
    assert laplacian[1, 1, 2] == 0


def test_edge_conditioner_divides_by_active_degree() -> None:
    class Ones(nn.Module):
        def forward(self, features: torch.Tensor) -> torch.Tensor:
            return torch.ones((*features.shape[:-1], 1), dtype=features.dtype)

    conditioner = EdgeConditioner(edge_channels=10, width=1)
    conditioner.from_encoder = Ones()
    conditioner.to_encoder = Ones()
    edge_features = torch.zeros(1, 2, 10)
    edge_features[0, :, -2] = torch.tensor([1.0, 0.0])
    output = conditioner(
        edge_features,
        torch.tensor([[0, 0], [1, 2]]),
        n_nodes=3,
    )
    torch.testing.assert_close(
        output,
        torch.tensor([[[1.0], [1.0], [0.0]]]),
    )


def test_chebyshev_conv_is_differentiable() -> None:
    features = torch.randn(2, 4, 3, requires_grad=True)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    laplacian = normalized_laplacian(edge_index, torch.ones(2, 3), n_nodes=4)
    layer = ChebyshevSpectralConv(3, 5, order=4)
    layer(features, laplacian).square().mean().backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()


def test_hard_head_satisfies_boxes_and_slack_reference() -> None:
    batch, n_bus, n_gen = 2, 5, 2
    model = TopologyConditionedPINO(width=16, depth=2, chebyshev_order=3)
    bus_features = torch.randn(batch, n_bus, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    edge_features = torch.randn(batch, 4, 10)
    edge_features[..., -2] = 1.0
    v_min = torch.full((batch, n_bus), 0.9)
    v_max = torch.full((batch, n_bus), 1.1)
    gen_bus = torch.tensor([[0, 3], [0, 3]])
    p_min = torch.zeros(batch, n_gen)
    p_max = torch.ones(batch, n_gen)
    q_min = -torch.ones(batch, n_gen)
    q_max = torch.ones(batch, n_gen)
    slack = torch.zeros(batch, n_bus, dtype=torch.bool)
    slack[:, 0] = True
    prediction = model(
        bus_features=bus_features,
        edge_features=edge_features,
        edge_index=edge_index,
        v_min=v_min,
        v_max=v_max,
        slack_mask=slack,
        gen_bus=gen_bus,
        p_min_pu=p_min,
        p_max_pu=p_max,
        q_min_pu=q_min,
        q_max_pu=q_max,
        v_setpoint=torch.ones(batch, n_gen),
        cost_coefficients=torch.ones(batch, n_gen, 3),
        base_mva=torch.full((batch,), 100.0),
    )
    assert torch.all((prediction.vm >= v_min) & (prediction.vm <= v_max))
    assert torch.all((prediction.pg_pu >= p_min) & (prediction.pg_pu <= p_max))
    assert torch.all((prediction.qg_pu >= q_min) & (prediction.qg_pu <= q_max))
    torch.testing.assert_close(prediction.va[:, 0], torch.zeros(batch))


def _transformer_features(phase_rad: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    y = torch.tensor(2.0 - 6.0j)
    tap = 1.05 * torch.exp(torch.tensor(1j * phase_rad))
    yff = y / tap.abs().square()
    yft = -y / tap.conj()
    ytf = -y / tap
    ytt = y
    edge = torch.tensor(
        [
            yff.real,
            yff.imag,
            yft.real,
            yft.imag,
            ytt.real,
            ytt.imag,
            ytf.real,
            ytf.imag,
            1.0,
            2.0,
        ]
    ).reshape(1, 1, 10)
    y_from = torch.stack([torch.stack([yff, yft])]).to(torch.complex64)
    y_to = torch.stack([torch.stack([ytf, ytt])]).to(torch.complex64)
    return edge, y_from, y_to


def test_reversing_phase_shift_changes_prediction_and_terminal_residual() -> None:
    torch.manual_seed(19)
    model = TopologyConditionedPINO(width=12, depth=2, chebyshev_order=2).eval()
    common = {
        "bus_features": torch.zeros(1, 2, 8),
        "edge_index": torch.tensor([[0], [1]]),
        "v_min": torch.full((1, 2), 0.9),
        "v_max": torch.full((1, 2), 1.1),
        "slack_mask": torch.tensor([[True, False]]),
        "gen_bus": torch.tensor([[0]]),
        "p_min_pu": torch.zeros(1, 1),
        "p_max_pu": torch.ones(1, 1),
        "q_min_pu": -torch.ones(1, 1),
        "q_max_pu": torch.ones(1, 1),
        "v_setpoint": torch.ones(1, 1),
        "cost_coefficients": torch.tensor([[[0.1, 1.0, 0.0]]]),
        "base_mva": torch.tensor([100.0]),
    }
    positive_edge, positive_yf, positive_yt = _transformer_features(0.2)
    negative_edge, negative_yf, negative_yt = _transformer_features(-0.2)
    positive = model(edge_features=positive_edge, **common)
    negative = model(edge_features=negative_edge, **common)
    assert not torch.allclose(positive.va, negative.va)
    vm = torch.tensor([[1.02, 0.98]])
    va = torch.tensor([[0.0, -0.05]])
    positive_flow = branch_power_flows(
        vm, va, positive_yf, positive_yt, torch.tensor([0]), torch.tensor([1])
    )
    negative_flow = branch_power_flows(
        vm, va, negative_yf, negative_yt, torch.tensor([0]), torch.tensor([1])
    )
    assert not torch.allclose(positive_flow[0], negative_flow[0])

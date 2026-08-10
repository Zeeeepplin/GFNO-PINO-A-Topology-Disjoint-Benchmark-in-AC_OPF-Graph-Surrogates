"""Numerical validation of differentiable AC equations.

These tests intentionally use an independently evaluated explicit G/B formula
and a two-bus analytic branch model. They do not merely compare a function to
itself through another wrapper.
"""

from __future__ import annotations

import math

import pytest
import torch

from losses.physics_losses import (
    AdaptiveLossBalancer,
    ac_power_injections,
    angle_difference_violation,
    branch_power_flows,
    economic_cost_per_sample,
    physics_informed_loss,
    power_balance_residual,
    samplewise_economic_regret,
    thermal_violation,
)

DTYPE = torch.float64
CDTYPE = torch.complex128


def test_ac_injection_matches_explicit_gb_equations() -> None:
    ybus = torch.tensor(
        [
            [6.25 - 18.695j, -5.0 + 15.0j, -1.25 + 3.75j],
            [-5.0 + 15.0j, 6.6666666667 - 19.95j, -1.6666666667 + 5.0j],
            [-1.25 + 3.75j, -1.6666666667 + 5.0j, 2.9166666667 - 8.705j],
        ],
        dtype=CDTYPE,
    )
    vm = torch.tensor([1.04, 1.01, 0.98], dtype=DTYPE)
    va = torch.tensor([0.0, -0.06, -0.11], dtype=DTYPE)
    p_actual, q_actual = ac_power_injections(vm, va, ybus)

    angle = va[:, None] - va[None, :]
    g, b = ybus.real, ybus.imag
    p_expected = vm * torch.sum(vm[None, :] * (g * angle.cos() + b * angle.sin()), dim=1)
    q_expected = vm * torch.sum(vm[None, :] * (g * angle.sin() - b * angle.cos()), dim=1)
    torch.testing.assert_close(p_actual, p_expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(q_actual, q_expected, rtol=1e-12, atol=1e-12)


def test_known_two_bus_powerflow_has_zero_kcl_residual() -> None:
    # Lossless line x=0.1 p.u.; V0=V1=1 and delta=asin(0.5) transfers 0.5 p.u.
    series_y = -10j
    ybus = torch.tensor([[series_y, -series_y], [-series_y, series_y]], dtype=CDTYPE)
    delta = math.asin(0.5)
    vm = torch.ones(2, dtype=DTYPE)
    va = torch.tensor([0.0, -delta], dtype=DTYPE)
    p_network, q_network = ac_power_injections(vm, va, ybus)

    pg = p_network[:1].clone()
    qg = q_network[:1].clone()
    p_load = torch.tensor([0.0, -p_network[1].item()], dtype=DTYPE)
    q_load = torch.tensor([0.0, -q_network[1].item()], dtype=DTYPE)
    p_res, q_res = power_balance_residual(
        vm,
        va,
        pg,
        qg,
        p_load,
        q_load,
        torch.zeros(2, dtype=DTYPE),
        ybus,
        torch.tensor([0]),
    )
    torch.testing.assert_close(p_res, torch.zeros_like(p_res), atol=1e-12, rtol=0)
    torch.testing.assert_close(q_res, torch.zeros_like(q_res), atol=1e-12, rtol=0)


def test_branch_flows_include_both_terminal_voltages() -> None:
    series_y = torch.tensor(2.0 - 4.0j, dtype=CDTYPE)
    y_from = torch.tensor([[series_y, -series_y]], dtype=CDTYPE)
    y_to = torch.tensor([[-series_y, series_y]], dtype=CDTYPE)
    vm = torch.tensor([1.02, 0.97], dtype=DTYPE)
    va = torch.tensor([0.05, -0.03], dtype=DTYPE)
    sf, st = branch_power_flows(
        vm,
        va,
        y_from,
        y_to,
        torch.tensor([0]),
        torch.tensor([1]),
    )
    voltage = torch.polar(vm, va)
    sf_expected = voltage[0] * (series_y * (voltage[0] - voltage[1])).conj()
    st_expected = voltage[1] * (series_y * (voltage[1] - voltage[0])).conj()
    torch.testing.assert_close(sf[0], sf_expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(st[0], st_expected, rtol=1e-12, atol=1e-12)


def test_thermal_violation_and_autograd() -> None:
    sf = torch.tensor([0.5 + 0.1j, 1.2 + 0.0j], dtype=CDTYPE, requires_grad=True)
    st = torch.tensor([0.4 + 0.2j, 0.9 + 0.0j], dtype=CDTYPE)
    violation = thermal_violation(sf, st, torch.tensor([1.0, 1.0], dtype=DTYPE))
    torch.testing.assert_close(violation, torch.tensor([0.0, 0.2], dtype=DTYPE))
    violation.sum().backward()
    assert sf.grad is not None
    assert torch.isfinite(sf.grad).all()


def test_rejects_real_ybus() -> None:
    with pytest.raises(TypeError):
        ac_power_injections(torch.ones(2), torch.zeros(2), torch.eye(2))


def test_ieee_case30_solution_matches_pandapower_sbus() -> None:
    """Integration check against an independently solved IEEE power flow."""
    pp = pytest.importorskip("pandapower")
    networks = pytest.importorskip("pandapower.networks")
    from pandapower.pypower.idx_bus import VA, VM
    from pandapower.pypower.makeSbus import makeSbus
    from pandapower.pypower.makeYbus import makeYbus

    net = networks.case30()
    pp.runpp(net, calculate_voltage_angles=True, numba=False)
    ppc = net["_ppc"]
    bus = ppc["bus"]
    ybus, _, _ = makeYbus(ppc["baseMVA"], bus, ppc["branch"])
    expected = makeSbus(ppc["baseMVA"], bus, ppc["gen"])
    p_actual, q_actual = ac_power_injections(
        torch.from_numpy(bus[:, VM]).to(DTYPE),
        torch.deg2rad(torch.from_numpy(bus[:, VA]).to(DTYPE)),
        torch.from_numpy(ybus.toarray()).to(CDTYPE),
    )
    torch.testing.assert_close(
        torch.complex(p_actual, q_actual),
        torch.from_numpy(expected).to(CDTYPE),
        rtol=1e-7,
        atol=1e-8,
    )


def test_adaptive_balancer_captures_initial_loss_ratios_before_gradnorm() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    losses = {
        "a": parameter.square().sum(),
        "b": (parameter - 1.0).square().sum(),
    }
    balancer = AdaptiveLossBalancer(("a", "b"))
    objective = balancer.gradnorm_objective(losses, parameter)
    torch.testing.assert_close(
        balancer.initial_losses,
        torch.stack([losses["a"], losses["b"]]).detach(),
    )
    objective.backward()
    assert balancer.log_weights.grad is not None


def test_adaptive_balancer_defers_zero_loss_until_first_activation() -> None:
    parameter = torch.nn.Parameter(torch.tensor([2.0]))
    balancer = AdaptiveLossBalancer(("active", "initially_zero"))
    first = {
        "active": parameter.square().sum(),
        "initially_zero": (parameter - parameter).square().sum(),
    }
    objective = balancer.gradnorm_objective(first, parameter)
    assert torch.isfinite(objective)
    assert balancer.active.tolist() == [True, False]

    second = {
        "active": parameter.square().sum(),
        "initially_zero": (parameter - 1.0).square().sum(),
    }
    objective = balancer.gradnorm_objective(second, parameter)
    assert torch.isfinite(objective)
    assert balancer.active.tolist() == [True, True]
    torch.testing.assert_close(
        balancer.initial_losses,
        torch.tensor([4.0, 1.0]),
    )
    objective.backward()
    assert torch.isfinite(balancer.log_weights.grad).all()


def test_angle_difference_violation_uses_active_branch_limits() -> None:
    va = torch.tensor([[0.0, -0.3, 0.5]], dtype=DTYPE)
    violation = angle_difference_violation(
        va,
        torch.tensor([[0, 1]]),
        torch.tensor([[1, 2]]),
        torch.tensor([[-0.2, -0.2]], dtype=DTYPE),
        torch.tensor([[0.2, 0.2]], dtype=DTYPE),
        torch.tensor([[1.0, 0.0]], dtype=DTYPE),
    )
    torch.testing.assert_close(violation, torch.tensor([[0.1, 0.0]], dtype=DTYPE))


def test_samplewise_economic_regret_prevents_batch_cancellation() -> None:
    predicted = torch.tensor([120.0, 80.0], dtype=DTYPE)
    reference = torch.tensor([100.0, 100.0], dtype=DTYPE)
    loss = samplewise_economic_regret(predicted, reference, cost_scale=1.0)
    torch.testing.assert_close(loss, torch.tensor(0.02, dtype=DTYPE))


def test_samplewise_economic_regret_handles_negative_reference_cost() -> None:
    predicted = torch.tensor([-90.0, -110.0], dtype=DTYPE)
    reference = torch.tensor([-100.0, -100.0], dtype=DTYPE)
    loss = samplewise_economic_regret(predicted, reference, cost_scale=1.0)
    torch.testing.assert_close(loss, torch.tensor(0.005, dtype=DTYPE))


def test_economic_cost_is_returned_per_sample() -> None:
    pg = torch.tensor([[2.0, 1.0], [3.0, 0.5]], dtype=DTYPE)
    coefficients = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [0.5, 1.0, 0.0]],
            [[1.0, 2.0, 3.0], [0.5, 1.0, 0.0]],
        ],
        dtype=DTYPE,
    )
    actual = economic_cost_per_sample(pg, coefficients)
    expected = torch.tensor([12.5, 18.625], dtype=DTYPE)
    torch.testing.assert_close(actual, expected)


def test_inactive_gradnorm_weights_do_not_enter_normalization() -> None:
    parameter = torch.nn.Parameter(torch.tensor([2.0], dtype=DTYPE))
    balancer = AdaptiveLossBalancer(("active", "inactive")).to(dtype=DTYPE)
    losses = {
        "active": parameter.square().sum(),
        "inactive": (parameter - parameter).square().sum(),
    }
    balancer.weighted_total(losses)
    torch.testing.assert_close(
        balancer.positive_weights(),
        torch.tensor([1.0, 0.0], dtype=DTYPE),
    )


def test_angle_excess_is_an_explicit_physics_objective() -> None:
    output = physics_informed_loss(
        vm=torch.ones((1, 2), dtype=DTYPE),
        va=torch.tensor([[0.3, 0.0]], dtype=DTYPE),
        pg_pu=torch.zeros((1, 1), dtype=DTYPE),
        qg_pu=torch.zeros((1, 1), dtype=DTYPE),
        p_load_pu=torch.zeros((1, 2), dtype=DTYPE),
        q_load_pu=torch.zeros((1, 2), dtype=DTYPE),
        p_renewable_pu=torch.zeros((1, 2), dtype=DTYPE),
        ybus_pu=torch.zeros((1, 2, 2), dtype=CDTYPE),
        gen_bus=torch.tensor([[0]]),
        y_from_pu=torch.zeros((1, 1, 2), dtype=CDTYPE),
        y_to_pu=torch.zeros((1, 1, 2), dtype=CDTYPE),
        rate_pu=torch.zeros((1, 1), dtype=DTYPE),
        v_min=torch.full((1, 2), 0.9, dtype=DTYPE),
        v_max=torch.full((1, 2), 1.1, dtype=DTYPE),
        p_min_pu=torch.zeros((1, 1), dtype=DTYPE),
        p_max_pu=torch.ones((1, 1), dtype=DTYPE),
        q_min_pu=-torch.ones((1, 1), dtype=DTYPE),
        q_max_pu=torch.ones((1, 1), dtype=DTYPE),
        cost_coefficients=torch.zeros((1, 1, 3), dtype=DTYPE),
        base_mva=100.0,
        branch_from=torch.tensor([[0]]),
        branch_to=torch.tensor([[1]]),
        branch_angle_min_rad=torch.tensor([[-0.2]], dtype=DTYPE),
        branch_angle_max_rad=torch.tensor([[0.2]], dtype=DTYPE),
        branch_status=torch.ones((1, 1), dtype=DTYPE),
        branch_mask=torch.zeros((1, 1), dtype=DTYPE),
    )
    torch.testing.assert_close(output.angle, torch.tensor(0.01, dtype=DTYPE))

"""Differentiable AC network equations and physics-informed OPF losses.

"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor, nn


def _complex_voltage(vm: Tensor, va: Tensor) -> Tensor:
    """Return complex bus voltage phasors from magnitude and radians."""
    if vm.shape != va.shape:
        raise ValueError(f"vm and va must have identical shapes, got {vm.shape} and {va.shape}")
    return torch.polar(vm, va)


def ac_power_injections(vm: Tensor, va: Tensor, ybus: Tensor) -> tuple[Tensor, Tensor]:
    """Compute bus injections ``S = V * conj(Ybus @ V)``.

    Args:
        vm: Voltage magnitudes, shape ``(..., n_bus)``.
        va: Voltage angles in radians, shape ``(..., n_bus)``.
        ybus: Complex bus-admittance matrix, shape ``(..., n_bus, n_bus)``.

    Returns:
        Active and reactive injections in per unit, each ``(..., n_bus)``.

    Positive values mean net power injected into the network. This complex
    implementation is algebraically identical to the explicit G/B equations
    in the paper and avoids materializing an ``N x N`` angle-difference tensor.
    """
    if not ybus.is_complex():
        raise TypeError("ybus must be a complex PyTorch tensor")
    voltage = _complex_voltage(vm, va)
    current = torch.matmul(ybus, voltage.unsqueeze(-1)).squeeze(-1)
    power = voltage * current.conj()
    return power.real, power.imag


def aggregate_generators(
    pg: Tensor,
    qg: Tensor,
    gen_bus: Tensor,
    n_bus: int,
    gen_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Scatter per-generator outputs to their buses.

    ``gen_bus`` may be unbatched ``(n_gen,)`` or batched ``(..., n_gen)``.
    Padding entries must be masked and may use any in-range bus index.
    """
    if pg.shape != qg.shape:
        raise ValueError("pg and qg must have identical shapes")
    target_shape = pg.shape[:-1] + (n_bus,)
    p_bus = torch.zeros(target_shape, device=pg.device, dtype=pg.dtype)
    q_bus = torch.zeros_like(p_bus)
    index = gen_bus.to(device=pg.device, dtype=torch.long)
    index = torch.broadcast_to(index, pg.shape)
    if torch.any((index < 0) | (index >= n_bus)):
        raise ValueError("gen_bus contains an out-of-range bus index")
    if gen_mask is None:
        weights = torch.ones_like(pg)
    else:
        weights = torch.broadcast_to(gen_mask.to(device=pg.device, dtype=pg.dtype), pg.shape)
    p_bus.scatter_add_(-1, index, pg * weights)
    q_bus.scatter_add_(-1, index, qg * weights)
    return p_bus, q_bus


def power_balance_residual(
    vm: Tensor,
    va: Tensor,
    pg: Tensor,
    qg: Tensor,
    p_load: Tensor,
    q_load: Tensor,
    p_renewable: Tensor,
    ybus: Tensor,
    gen_bus: Tensor,
    gen_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Return active/reactive complex-power-balance residuals at every bus.

    The injection convention is
    ``generation + renewable - load - network_injection = 0``.
    """
    p_net, q_net = ac_power_injections(vm, va, ybus)
    p_gen_bus, q_gen_bus = aggregate_generators(pg, qg, gen_bus, vm.shape[-1], gen_mask)
    p_residual = p_gen_bus + p_renewable - p_load - p_net
    q_residual = q_gen_bus - q_load - q_net
    return p_residual, q_residual


def branch_power_flows(
    vm: Tensor,
    va: Tensor,
    y_from: Tensor,
    y_to: Tensor,
    branch_from: Tensor,
    branch_to: Tensor,
) -> tuple[Tensor, Tensor]:
    """Compute complex branch powers at both ends using Yf/Yt matrices.

    Args:
        y_from: Complex matrix mapping bus voltage to from-end branch current,
            shape ``(..., n_branch, n_bus)``.
        y_to: Equivalent to-end matrix with the same shape.

    Returns:
        ``(S_from, S_to)`` in per unit, shape ``(..., n_branch)``.
    """
    if not y_from.is_complex() or not y_to.is_complex():
        raise TypeError("y_from and y_to must be complex tensors")
    voltage = _complex_voltage(vm, va)
    i_from = torch.matmul(y_from, voltage.unsqueeze(-1)).squeeze(-1)
    i_to = torch.matmul(y_to, voltage.unsqueeze(-1)).squeeze(-1)
    # Explicit indices are required for exact transformer handling and because
    # ordinary line terminal coefficients can have equal magnitude.
    from_bus = branch_from
    to_bus = branch_to
    from_bus = torch.broadcast_to(from_bus.to(device=vm.device, dtype=torch.long), i_from.shape)
    to_bus = torch.broadcast_to(to_bus.to(device=vm.device, dtype=torch.long), i_to.shape)
    v_from = voltage.gather(-1, from_bus)
    v_to = voltage.gather(-1, to_bus)
    return v_from * i_from.conj(), v_to * i_to.conj()


def thermal_violation(
    s_from: Tensor,
    s_to: Tensor,
    rate: Tensor,
    branch_mask: Tensor | None = None,
) -> Tensor:
    """Return non-negative MVA-limit excess for the worse branch end."""
    limit = torch.broadcast_to(rate.to(s_from.real.dtype), s_from.shape)
    excess = torch.relu(torch.maximum(s_from.abs(), s_to.abs()) - limit)
    if branch_mask is not None:
        excess = excess * torch.broadcast_to(branch_mask.to(excess.dtype), excess.shape)
    return excess


def voltage_violation(vm: Tensor, v_min: Tensor, v_max: Tensor) -> Tensor:
    """Return elementwise distance outside voltage box limits."""
    return torch.relu(v_min - vm) + torch.relu(vm - v_max)


def generator_box_violation(
    pg: Tensor,
    qg: Tensor,
    p_min: Tensor,
    p_max: Tensor,
    q_min: Tensor,
    q_max: Tensor,
    gen_mask: Tensor | None = None,
) -> Tensor:
    """Return combined active/reactive generator box-limit excess."""
    violation = (
        torch.relu(p_min - pg)
        + torch.relu(pg - p_max)
        + torch.relu(q_min - qg)
        + torch.relu(qg - q_max)
    )
    if gen_mask is not None:
        violation = violation * gen_mask.to(violation.dtype)
    return violation


def angle_difference_violation(
    va: Tensor,
    branch_from: Tensor,
    branch_to: Tensor,
    angle_min_rad: Tensor,
    angle_max_rad: Tensor,
    branch_status: Tensor | None = None,
) -> Tensor:
    """Return branch angle-difference excess in radians.

    PowerModels constrains ``va[from] - va[to]`` using the MATPOWER
    ``ANGMIN``/``ANGMAX`` fields. Out-of-service records are masked because
    they are absent from the corresponding AC-OPF constraint set.
    """
    if branch_from.ndim == 1:
        branch_from = branch_from.unsqueeze(0).expand(va.shape[0], -1)
        branch_to = branch_to.unsqueeze(0).expand(va.shape[0], -1)
    from_angle = va.gather(-1, branch_from.to(torch.long))
    to_angle = va.gather(-1, branch_to.to(torch.long))
    difference = from_angle - to_angle
    violation = torch.relu(angle_min_rad - difference) + torch.relu(
        difference - angle_max_rad
    )
    if branch_status is not None:
        violation = violation * branch_status.to(violation.dtype)
    return violation


def economic_cost_per_sample(
    pg: Tensor,
    cost_coefficients: Tensor,
    gen_mask: Tensor | None = None,
) -> Tensor:
    """Evaluate quadratic generator cost separately for every sample.

    ``cost_coefficients[..., g, :]`` is ordered ``[c2, c1, c0]`` and powers
    must use the same units as the fitted costs (normally MW, not p.u.).
    """
    c2, c1, c0 = cost_coefficients.unbind(dim=-1)
    cost = c2 * pg.square() + c1 * pg + c0
    if gen_mask is not None:
        cost = cost * gen_mask.to(cost.dtype)
    return cost.sum(dim=-1)


def economic_cost(pg: Tensor, cost_coefficients: Tensor, gen_mask: Tensor | None = None) -> Tensor:
    """Evaluate mean quadratic generator cost across a batch."""
    return economic_cost_per_sample(pg, cost_coefficients, gen_mask).mean()


def samplewise_economic_regret(
    predicted_cost: Tensor,
    reference_cost: Tensor,
    cost_scale: Tensor | float = 1.0,
) -> Tensor:
    """Return one-sided normalized regret, averaged only after samplewise evaluation.

    Costs and ``cost_scale`` have currency-per-hour units. The default scale is
    therefore one currency unit per hour, not a dimensionless constant. A
    prediction is penalized only when its cost is larger than the corresponding
    locally converged reference; lower, potentially infeasible costs are not
    rewarded. Computing the ReLU before the batch mean prevents cancellation
    between over-cost and under-cost samples and remains correct for negative
    reference costs.
    """
    scale = torch.as_tensor(
        cost_scale,
        device=predicted_cost.device,
        dtype=predicted_cost.dtype,
    )
    if torch.any(scale <= 0):
        raise ValueError("cost_scale must be positive")
    denominator = torch.maximum(reference_cost.abs(), scale)
    return torch.relu((predicted_cost - reference_cost) / denominator).square().mean()


def masked_mean_square(value: Tensor, mask: Tensor | None = None) -> Tensor:
    """Mean square with a safe, broadcastable validity mask."""
    if mask is None:
        return value.square().mean()
    float_mask = torch.broadcast_to(mask.to(value.dtype), value.shape)
    return (value.square() * float_mask).sum() / float_mask.sum().clamp_min(1.0)


@dataclass(frozen=True)
class PhysicsLossOutput:
    """Named scalar components used by the adaptive training objective."""

    economic: Tensor
    powerflow: Tensor
    thermal: Tensor
    angle: Tensor
    voltage: Tensor
    generator: Tensor
    supervised: Tensor

    def as_dict(self) -> dict[str, Tensor]:
        return {
            "economic": self.economic,
            "powerflow": self.powerflow,
            "thermal": self.thermal,
            "angle": self.angle,
            "voltage": self.voltage,
            "generator": self.generator,
            "supervised": self.supervised,
        }


def physics_informed_loss(
    *,
    vm: Tensor,
    va: Tensor,
    pg_pu: Tensor,
    qg_pu: Tensor,
    p_load_pu: Tensor,
    q_load_pu: Tensor,
    p_renewable_pu: Tensor,
    ybus_pu: Tensor,
    gen_bus: Tensor,
    y_from_pu: Tensor,
    y_to_pu: Tensor,
    rate_pu: Tensor,
    v_min: Tensor,
    v_max: Tensor,
    p_min_pu: Tensor,
    p_max_pu: Tensor,
    q_min_pu: Tensor,
    q_max_pu: Tensor,
    cost_coefficients: Tensor,
    base_mva: Tensor | float,
    reference_objective: Tensor | None = None,
    economic_cost_scale: Tensor | float = 1.0,
    branch_from: Tensor,
    branch_to: Tensor,
    branch_angle_min_rad: Tensor,
    branch_angle_max_rad: Tensor,
    branch_status: Tensor,
    bus_mask: Tensor | None = None,
    gen_mask: Tensor | None = None,
    branch_mask: Tensor | None = None,
    targets: Mapping[str, Tensor] | None = None,
) -> PhysicsLossOutput:
    """Compute all separate PINO objective terms.

    Box violations are retained as diagnostics even when the model uses its
    hard sigmoid parameterization; they should then be zero up to roundoff.
    """
    p_res, q_res = power_balance_residual(
        vm,
        va,
        pg_pu,
        qg_pu,
        p_load_pu,
        q_load_pu,
        p_renewable_pu,
        ybus_pu,
        gen_bus,
        gen_mask,
    )
    s_from, s_to = branch_power_flows(vm, va, y_from_pu, y_to_pu, branch_from, branch_to)
    thermal_excess = thermal_violation(s_from, s_to, rate_pu, branch_mask)
    angle_excess = angle_difference_violation(
        va,
        branch_from,
        branch_to,
        branch_angle_min_rad,
        branch_angle_max_rad,
        branch_status,
    )
    volt_excess = voltage_violation(vm, v_min, v_max)
    gen_excess = generator_box_violation(
        pg_pu, qg_pu, p_min_pu, p_max_pu, q_min_pu, q_max_pu, gen_mask
    )
    base = torch.as_tensor(base_mva, device=pg_pu.device, dtype=pg_pu.dtype)
    pg_mw = pg_pu * base.unsqueeze(-1) if base.ndim else pg_pu * base
    supervised = torch.zeros((), device=vm.device, dtype=vm.dtype)
    if targets:
        fields = {"vm": vm, "va": va, "pg_pu": pg_pu, "qg_pu": qg_pu}
        terms = [masked_mean_square(fields[name] - target) for name, target in targets.items()]
        supervised = torch.stack(terms).mean() if terms else supervised
    predicted_cost = economic_cost_per_sample(pg_mw, cost_coefficients, gen_mask)
    economic = predicted_cost.mean()
    if reference_objective is not None:
        economic = samplewise_economic_regret(
            predicted_cost,
            reference_objective,
            economic_cost_scale,
        )
    return PhysicsLossOutput(
        economic=economic,
        powerflow=masked_mean_square(p_res, bus_mask) + masked_mean_square(q_res, bus_mask),
        thermal=masked_mean_square(thermal_excess, branch_mask),
        angle=masked_mean_square(angle_excess, branch_status),
        voltage=masked_mean_square(volt_excess, bus_mask),
        generator=masked_mean_square(gen_excess, gen_mask),
        supervised=supervised,
    )


class AdaptiveLossBalancer(nn.Module):
    """GradNorm-style positive loss weights with a safe active-objective set.

    The model parameters are updated with ``weighted_total``. Call
    ``gradnorm_objective`` against a shared model parameter tensor to update
    only ``log_weights``. An objective enters GradNorm when its detached loss
    first exceeds ``epsilon``; that first positive value becomes its baseline.
    Identically zero hard-constraint diagnostics therefore never produce the
    undefined ``0/0`` relative rate criticized in the manuscript review.
    """

    def __init__(
        self,
        names: tuple[str, ...],
        alpha: float = 0.5,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        if not names:
            raise ValueError("at least one loss name is required")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.names = names
        self.alpha = alpha
        self.epsilon = epsilon
        self.log_weights = nn.Parameter(torch.zeros(len(names)))
        self.register_buffer("initial_losses", torch.zeros(len(names)))
        self.register_buffer("active", torch.zeros(len(names), dtype=torch.bool))

    def positive_weights(self) -> Tensor:
        """Return active-set weights normalized to the number of active objectives.

        Inactive objectives receive exactly zero weight and do not participate
        in the normalization. When a previously zero objective first exceeds
        ``epsilon``, ``_activate`` adds it to the active set before either the
        model or GradNorm update.
        """
        raw = self.log_weights.exp()
        active = self.active.to(raw.dtype)
        active_count = active.sum()
        if not bool(self.active.any()):
            return torch.zeros_like(raw)
        return active_count * raw * active / (raw * active).sum().clamp_min(1e-12)

    @torch.no_grad()
    def _activate(self, values: Tensor) -> None:
        detached = values.detach()
        newly_active = (~self.active) & torch.isfinite(detached) & (
            detached > self.epsilon
        )
        self.initial_losses[newly_active] = detached[newly_active]
        self.active[newly_active] = True

    def weighted_total(self, losses: Mapping[str, Tensor]) -> Tensor:
        values = torch.stack([losses[name] for name in self.names])
        self._activate(values)
        return torch.sum(self.positive_weights().detach() * values)

    def gradnorm_objective(self, losses: Mapping[str, Tensor], shared: Tensor) -> Tensor:
        """Return the detached-target GradNorm objective for active losses."""
        values = torch.stack([losses[name] for name in self.names])
        self._activate(values)
        active_indices = self.active.nonzero(as_tuple=False).flatten()
        if active_indices.numel() < 2:
            return self.log_weights.sum() * 0.0
        weights = self.positive_weights()[active_indices]
        active_values = values[active_indices]
        norms = []
        for weight, value in zip(weights, active_values, strict=True):
            grad = torch.autograd.grad(
                weight * value, shared, retain_graph=True, create_graph=True, allow_unused=False
            )[0]
            norms.append(torch.linalg.vector_norm(grad))
        gradient_norms = torch.stack(norms)
        initial = self.initial_losses[active_indices]
        loss_ratio = (active_values.detach() + self.epsilon) / (
            initial + self.epsilon
        )
        inverse_rate = loss_ratio / loss_ratio.mean().clamp_min(self.epsilon)
        target = gradient_norms.detach().mean() * inverse_rate.pow(self.alpha)
        return torch.abs(gradient_norms - target).sum()

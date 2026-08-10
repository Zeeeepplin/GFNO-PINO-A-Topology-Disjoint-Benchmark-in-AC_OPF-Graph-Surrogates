"""Generate topology-disjoint, solver-labelled IEEE AC-OPF datasets.

Example:
    python -m data_gen.generate --config configs/case30.yaml
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import yaml

from data_gen.classical_baselines import ClassicalTiming, solve_dc_opf
from data_gen.solver import PowerModelsIPOPTSession, solve_ac_opf

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Topology:
    """One base or N-1 line topology."""

    topology_id: str
    outaged_line: int | None


def load_ieee_case(name: str) -> Any:
    """Load a supported pandapower IEEE case."""
    import pandapower.networks as pn

    factories = {
        "case30": pn.case30,
        "case57": pn.case57,
        "case118": pn.case118,
        "case300": pn.case300,
    }
    try:
        return factories[name]()
    except KeyError as exc:
        raise ValueError(f"unsupported case {name!r}; choose from {sorted(factories)}") from exc


def enumerate_non_islanding_n_minus_one(net: Any, include_base: bool = True) -> list[Topology]:
    """Enumerate all single *line* outages that retain one connected bus graph.

    Transformers remain edges in the connectivity test. Parallel line outages
    are handled correctly because pandapower's multigraph retains the other
    circuit.
    """
    import pandapower.topology as top

    topologies = [Topology("base", None)] if include_base else []
    for line_index in net.line.index:
        if not bool(net.line.at[line_index, "in_service"]):
            continue
        candidate = copy.deepcopy(net)
        candidate.line.at[line_index, "in_service"] = False
        graph = top.create_nxgraph(
            candidate,
            respect_switches=True,
            include_lines=True,
            include_trafos=True,
            multi=True,
        )
        active_buses = [int(i) for i in candidate.bus.index[candidate.bus.in_service]]
        subgraph = nx.Graph(graph.subgraph(active_buses))
        if active_buses and nx.is_connected(subgraph):
            topologies.append(Topology(f"line_{int(line_index)}_out", int(line_index)))
        else:
            LOGGER.info("excluding islanding outage of line %s", line_index)
    return topologies


def split_topologies(
    topologies: list[Topology],
    fractions: tuple[float, float, float],
    seed: int,
) -> dict[str, str]:
    """Assign complete topologies to train/validation/test deterministically."""
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("topology split fractions must sum to one")
    base = [item for item in topologies if item.outaged_line is None]
    contingencies = [item for item in topologies if item.outaged_line is not None]
    rng = np.random.default_rng(seed)
    rng.shuffle(contingencies)
    n_total = len(contingencies)
    n_train = int(np.floor(fractions[0] * n_total))
    n_val = int(np.floor(fractions[1] * n_total))
    result = {item.topology_id: "train" for item in base}
    for position, topology in enumerate(contingencies):
        if position < n_train:
            split = "train"
        elif position < n_train + n_val:
            split = "val"
        else:
            split = "test"
        result[topology.topology_id] = split
    return result


class CorrelatedLoadSampler:
    """Reusable bounded multivariate sampler over graph-distance covariance."""

    def __init__(
        self,
        net: Any,
        rng: np.random.Generator,
        lower: float,
        upper: float,
        correlation_length: float,
    ) -> None:
        import pandapower.topology as top

        self.rng = rng
        self.lower = lower
        self.upper = upper
        graph = nx.Graph(
            top.create_nxgraph(
                net,
                respect_switches=True,
                include_lines=True,
                include_trafos=True,
            )
        )
        buses = [int(index) for index in net.bus.index]
        n_bus = len(buses)
        positions = {bus: position for position, bus in enumerate(buses)}
        distance = np.full((n_bus, n_bus), n_bus, dtype=np.float64)
        for source, lengths in nx.all_pairs_shortest_path_length(graph):
            if source not in positions:
                continue
            source_position = positions[source]
            for target, value in lengths.items():
                if target in positions:
                    distance[source_position, positions[target]] = value
        covariance = np.exp(-(distance**2) / (2.0 * correlation_length**2))
        covariance = 0.5 * (covariance + covariance.T)
        # A radial kernel of graph geodesic distance is not guaranteed PSD for
        # every graph. Project to the nearest PSD eigenspectrum once, rather
        # than silently accepting invalid multivariate samples.
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.clip(eigenvalues, 1e-8, None)
        self.factor = eigenvectors * np.sqrt(eigenvalues)[None, :]

    def sample(self) -> np.ndarray:
        latent = self.factor @ self.rng.standard_normal(self.factor.shape[1])
        bounded = 1.0 / (1.0 + np.exp(-latent))
        return self.lower + (self.upper - self.lower) * bounded


def graph_correlated_multipliers(
    net: Any,
    rng: np.random.Generator,
    lower: float,
    upper: float,
    correlation_length: float,
) -> np.ndarray:
    """One-shot compatibility wrapper around :class:`CorrelatedLoadSampler`."""
    return CorrelatedLoadSampler(net, rng, lower, upper, correlation_length).sample()


class RenewableProfiles:
    """Optional real time-series reader with a deterministic synthetic fallback."""

    def __init__(self, csv_path: str | None, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.values: np.ndarray | None = None
        if csv_path:
            frame = np.genfromtxt(csv_path, delimiter=",", names=True)
            if frame.dtype.names is None:
                raise ValueError("renewable CSV must have a header and numeric columns")
            columns = [np.asarray(frame[name], dtype=np.float64) for name in frame.dtype.names]
            values = np.column_stack(columns)
            finite_rows = np.isfinite(values).all(axis=1)
            self.values = values[finite_rows]
            if not len(self.values):
                raise ValueError("renewable CSV contains no finite rows")

    @property
    def source(self) -> str:
        return "user-provided-timeseries" if self.values is not None else "synthetic-diurnal"

    def sample(self, n_sites: int, step: int) -> np.ndarray:
        if n_sites == 0:
            return np.empty(0, dtype=np.float64)
        if self.values is not None:
            row = self.values[step % len(self.values)]
            row = np.resize(row, n_sites)
            maximum = np.maximum(np.nanmax(self.values, axis=0), 1e-9)
            return np.clip(row / np.resize(maximum, n_sites), 0.0, 1.0)
        # Correlated solar-like daily capacity factors. This is deliberately
        # tagged synthetic in metadata and must not be reported as NREL data.
        hour = step % 24
        clear_sky = max(0.0, np.sin(np.pi * (hour - 6.0) / 12.0))
        common_cloud = self.rng.beta(5.0, 2.0)
        local = self.rng.normal(1.0, 0.08, size=n_sites)
        return np.clip(clear_sky * common_cloud * local, 0.0, 1.0)


def apply_scenario(
    net: Any,
    *,
    bus_multiplier: np.ndarray,
    renewable_capacity_fraction: float,
    renewable_profile: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mutate a network with load scaling and fixed renewable injections."""
    bus_position = {int(bus): position for position, bus in enumerate(net.bus.index)}
    for load_index, load in net.load.iterrows():
        multiplier = bus_multiplier[bus_position[int(load.bus)]]
        net.load.at[load_index, "p_mw"] = float(load.p_mw * multiplier)
        net.load.at[load_index, "q_mvar"] = float(load.q_mvar * multiplier)

    n_bus = len(net.bus)
    p_load = np.zeros(n_bus, dtype=np.float64)
    q_load = np.zeros(n_bus, dtype=np.float64)
    for _, load in net.load[net.load.in_service].iterrows():
        position = bus_position[int(load.bus)]
        p_load[position] += float(load.p_mw)
        q_load[position] += float(load.q_mvar)

    p_renewable = np.zeros(n_bus, dtype=np.float64)
    if renewable_capacity_fraction > 0:
        candidate_buses = list(net.gen.bus.astype(int).unique())
        if not candidate_buses:
            candidate_buses = list(net.load.bus.astype(int).unique())
        total_load = max(p_load.sum(), 0.0)
        site_capacity = renewable_capacity_fraction * total_load / max(len(candidate_buses), 1)
        for site, bus in enumerate(candidate_buses):
            injection = float(site_capacity * renewable_profile[site])
            p_renewable[bus_position[bus]] += injection
            # Fixed, unity-power-factor sgen. It is not merged into dispatchable
            # Pg labels and remains an explicit operator input.
            import pandapower as pp

            pp.create_sgen(net, bus=bus, p_mw=injection, q_mvar=0.0, controllable=False)
    return p_load, q_load, p_renewable


def apply_case_adjustments(net: Any, adjustments: dict[str, Any] | None) -> None:
    """Apply explicitly configured feasibility adjustments to an imported case.

    Some legacy IEEE power-flow cases are not feasible AC-OPF benchmarks under
    their imported voltage boxes. Any adjustment is config-driven, retained in
    metadata, and must be disclosed rather than silently changing the case.
    """
    if not adjustments:
        return
    voltage_bounds = adjustments.get("voltage_bounds")
    if voltage_bounds is not None:
        lower, upper = (float(value) for value in voltage_bounds)
        if not 0.0 < lower < upper:
            raise ValueError("case_adjustments.voltage_bounds must satisfy 0 < lower < upper")
        net.bus.loc[:, "min_vm_pu"] = lower
        net.bus.loc[:, "max_vm_pu"] = upper


def _canonical_network_operators(
    reference_net: Any,
    outaged_line: int | None,
) -> dict[str, np.ndarray | float]:
    """Build exact fixed-shape Ybus/Yf/Yt and branch tensors.

    pandapower removes out-of-service rows during conversion. We instead start
    from the base case's canonical branch list and set BR_STATUS to zero. This
    preserves a common edge axis across topologies and makes line status an
    explicit operator input, rather than encoding an outage only via shape.
    """
    from pandapower.converter.pypower.to_ppc import to_ppc
    from pandapower.pypower.idx_brch import (
        ANGMAX,
        ANGMIN,
        BR_B,
        BR_STATUS,
        F_BUS,
        RATE_A,
        SHIFT,
        T_BUS,
        TAP,
        branch_cols,
    )
    from pandapower.pypower.idx_bus import BASE_KV, BUS_TYPE, VMAX, VMIN
    from pandapower.pypower.idx_gen import PMAX, PMIN, QMAX, QMIN, VG
    from pandapower.pypower.makeYbus import makeYbus

    ppc = to_ppc(
        reference_net,
        calculate_voltage_angles=True,
        init="flat",
        mode="opf",
    )
    base_mva = float(ppc["baseMVA"])
    bus = np.asarray(ppc["bus"])
    branch = np.asarray(ppc["branch"]).copy()
    # pandapower 3.5's OPF converter returns a legacy-width matrix while its
    # makeYbus accepts additional asymmetric/shunt columns. Zero is the defined
    # default for those optional electrical parameters.
    if branch.shape[1] < branch_cols:
        branch = np.pad(branch, ((0, 0), (0, branch_cols - branch.shape[1])))
    gen = np.asarray(ppc["gen"])
    line_start, _ = reference_net["_pd2ppc_lookups"]["branch"]["line"]
    active_lines = [
        int(index)
        for index in reference_net.line.index
        if bool(reference_net.line.at[index, "in_service"])
    ]
    line_to_branch = {
        line_index: line_start + position for position, line_index in enumerate(active_lines)
    }
    if outaged_line is not None:
        try:
            branch[line_to_branch[outaged_line], BR_STATUS] = 0.0
        except KeyError as exc:
            raise ValueError(f"outaged line {outaged_line} is not active in the base case") from exc
    ybus_sparse, yf_sparse, yt_sparse = makeYbus(base_mva, bus, branch)
    ybus = np.asarray(ybus_sparse.toarray(), dtype=np.complex128)
    yf = np.asarray(yf_sparse.toarray(), dtype=np.complex128)
    yt = np.asarray(yt_sparse.toarray(), dtype=np.complex128)

    tap = branch[:, TAP].copy()
    tap[tap == 0] = 1.0
    status = branch[:, BR_STATUS].astype(np.float64)
    branch_from = branch[:, F_BUS].astype(np.int64)
    branch_to = branch[:, T_BUS].astype(np.int64)
    row = np.arange(len(branch))
    # Endpoint-specific coefficients preserve transformer tap and phase-shift
    # directionality in the predictor, not only in the physics loss:
    # If = Yff*Vf + Yft*Vt and It = Ytf*Vf + Ytt*Vt.
    yff = yf[row, branch_from]
    yft = yf[row, branch_to]
    ytf = yt[row, branch_from]
    ytt = yt[row, branch_to]
    edge_features = np.column_stack(
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
            branch[:, RATE_A] / base_mva,
        ]
    )
    component_type = np.full(len(branch), 2, dtype=np.int64)
    lookup = reference_net["_pd2ppc_lookups"]["branch"]
    if "line" in lookup:
        start, end = lookup["line"]
        component_type[int(start) : int(end)] = 0
    if "trafo" in lookup:
        start, end = lookup["trafo"]
        component_type[int(start) : int(end)] = 1
    physical_line_index = np.full(len(branch), -1, dtype=np.int64)
    for line_index, branch_index in line_to_branch.items():
        physical_line_index[branch_index] = line_index
    bus_type = bus[:, BUS_TYPE].astype(np.int64)
    bus_type_onehot = np.eye(4, dtype=np.float64)[np.clip(bus_type, 0, 3)]
    return {
        "base_mva": np.asarray(base_mva),
        "ybus_real": ybus.real,
        "ybus_imag": ybus.imag,
        "y_from_real": yf.real,
        "y_from_imag": yf.imag,
        "y_to_real": yt.real,
        "y_to_imag": yt.imag,
        "branch_from": branch_from,
        "branch_to": branch_to,
        "edge_features": edge_features,
        "branch_rate_pu": branch[:, RATE_A] / base_mva,
        "branch_status": status,
        "branch_angle_min_rad": np.deg2rad(branch[:, ANGMIN]),
        "branch_angle_max_rad": np.deg2rad(branch[:, ANGMAX]),
        # 0=line, 1=two-winding transformer, 2=other converter record.
        "branch_component_type": component_type,
        "physical_line_index": physical_line_index,
        "branch_charging": branch[:, BR_B],
        "branch_tap": tap,
        "branch_shift_deg": branch[:, SHIFT],
        "bus_type_onehot": bus_type_onehot,
        "base_kv": bus[:, BASE_KV],
        "v_min": bus[:, VMIN],
        "v_max": bus[:, VMAX],
        "gen_bus": gen[:, 0].astype(np.int64),
        "p_min_pu": gen[:, PMIN] / base_mva,
        "p_max_pu": gen[:, PMAX] / base_mva,
        "q_min_pu": gen[:, QMIN] / base_mva,
        "q_max_pu": gen[:, QMAX] / base_mva,
        "v_setpoint": gen[:, VG],
        "cost_coefficients": _quadratic_costs(np.asarray(ppc["gencost"]), len(gen)),
    }


def _quadratic_costs(gencost: np.ndarray, n_gen: int) -> np.ndarray:
    """Return [c2,c1,c0] for each generator, padding lower-order costs."""
    from pandapower.pypower.idx_cost import COST, MODEL, NCOST, POLYNOMIAL

    result = np.zeros((n_gen, 3), dtype=np.float64)
    for generator in range(min(n_gen, len(gencost))):
        row = gencost[generator]
        if int(row[MODEL]) != POLYNOMIAL:
            raise ValueError("piecewise-linear costs are not supported by the quadratic objective")
        count = int(row[NCOST])
        coefficients = np.asarray(row[COST : COST + count], dtype=np.float64)
        if count > 3:
            raise ValueError("generator cost polynomial degree exceeds quadratic")
        result[generator, 3 - count :] = coefficients
    return result


def _bus_ordered(values: np.ndarray, net: Any) -> np.ndarray:
    """Map arrays in sorted pandapower bus index order to PYPOWER bus order."""
    lookup = np.asarray(net["_pd2ppc_lookups"]["bus"])
    result = np.zeros(len(net["_ppc"]["bus"]), dtype=np.float64)
    for position, bus in enumerate(net.bus.index):
        result[int(lookup[int(bus)])] = values[position]
    return result


def _save_npz_atomic(path: Path, **arrays: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def write_topology_manifest(
    output_root: Path,
    topologies: list[Topology],
    split_by_topology: dict[str, str],
    sample_counts: dict[str, int],
) -> Path:
    """Publish exact topology IDs, outages, splits, and realized sample counts."""
    path = output_root / "topology_manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "topology_id",
                "outaged_line",
                "split",
                "converged_samples",
            ],
        )
        writer.writeheader()
        for topology in topologies:
            writer.writerow(
                {
                    "topology_id": topology.topology_id,
                    "outaged_line": (
                        "" if topology.outaged_line is None else topology.outaged_line
                    ),
                    "split": split_by_topology[topology.topology_id],
                    "converged_samples": sample_counts.get(topology.topology_id, 0),
                }
            )
    return path


def write_constraint_inventory(output_root: Path) -> Path:
    """Write the exact source-case inequality inventory used by evaluation."""
    with np.load(output_root / "topologies" / "base.npz", allow_pickle=False) as base:
        status = np.asarray(base["branch_status"]) > 0
        rated = status & (np.asarray(base["branch_rate_pu"]) > 0)
        angle_min = np.asarray(base["branch_angle_min_rad"])
        angle_max = np.asarray(base["branch_angle_max_rad"])
        inventory = {
            "power_balance": {
                "count": int(2 * len(base["v_min"])),
                "unit": "per-unit active/reactive power",
                "evaluation_tolerance": 1e-3,
            },
            "voltage_box": {
                "count": int(len(base["v_min"])),
                "unit": "per-unit voltage",
                "evaluation_tolerance": 1e-6,
            },
            "generator_box": {
                "count": int(4 * len(base["gen_bus"])),
                "unit": "per-unit active/reactive power",
                "evaluation_tolerance": 1e-6,
            },
            "thermal_two_ended": {
                "rated_active_branches": int(rated.sum()),
                "unit": "per-unit apparent power",
                "evaluation_tolerance": 1e-6,
            },
            "angle_difference": {
                "active_branches": int(status.sum()),
                "minimum_rad": float(angle_min[status].min()),
                "maximum_rad": float(angle_max[status].max()),
                "evaluation_tolerance": 1e-6,
            },
            "reference_angle": {
                "count": 1,
                "unit": "radian",
                "hard_enforced": True,
            },
        }
    path = output_root / "constraint_inventory.json"
    path.write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    return path


def generate_dataset(config: dict[str, Any]) -> Path:
    """Generate all configured topology/scenario samples and return manifest."""
    seed = int(config.get("seed", 0))
    rng = np.random.default_rng(seed)
    case_name = str(config["case"])
    output_root = Path(config.get("output_dir", "data/processed")) / case_name
    topology_dir = output_root / "topologies"
    sample_dir = output_root / "samples"
    topology_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    base_net = load_ieee_case(case_name)
    apply_case_adjustments(base_net, config.get("case_adjustments"))
    topologies = enumerate_non_islanding_n_minus_one(
        base_net, bool(config.get("include_base_topology", True))
    )
    if config.get("max_topologies") is not None:
        # Intended for CI/smoke tests only. Publication configs omit this and
        # enumerate every non-islanding single-line contingency.
        topologies = topologies[: int(config["max_topologies"])]
    fractions = tuple(float(x) for x in config["split"]["fractions"])
    split_by_topology = split_topologies(topologies, fractions, seed)
    renewable = RenewableProfiles(config.get("renewable", {}).get("csv_path"), seed + 1)
    perturbation = config["load_perturbation"]
    load_sampler = CorrelatedLoadSampler(
        base_net,
        rng,
        float(perturbation["min"]),
        float(perturbation["max"]),
        float(perturbation.get("correlation_length", 3.0)),
    )
    samples_per_topology = int(config["scenarios_per_topology"])
    max_attempts_per_topology = int(
        config.get("max_scenario_attempts_per_topology", 5 * samples_per_topology)
    )
    if max_attempts_per_topology < samples_per_topology:
        raise ValueError("max_scenario_attempts_per_topology cannot be smaller than the target")
    solver_config = dict(config.get("solver", {}))
    backend = solver_config.pop("backend", "pandapower")
    persistent_solver: PowerModelsIPOPTSession | None = None
    if backend == "powermodels":
        persistent_solver = PowerModelsIPOPTSession(**solver_config)
        persistent_solver.__enter__()
        warmup_result = persistent_solver.solve(copy.deepcopy(base_net))
        LOGGER.info(
            "PowerModels warm-up termination=%s runtime_s=%.3f",
            warmup_result.termination,
            warmup_result.runtime_s,
        )
    failed_rows: list[dict[str, Any]] = []
    topology_sample_counts: dict[str, int] = {}
    manifest_path = output_root / "manifest.csv"
    generation_started = time.perf_counter()

    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        fieldnames = [
            "sample_id",
            "relative_path",
            "topology_id",
            "scenario_attempt",
            "split",
            "solver_backend",
            "solver_runtime_s",
            "objective",
            "dc_opf_runtime_s",
            "dc_opf_objective",
            "dc_opf_converged",
            "renewable_source",
        ]
        writer = csv.DictWriter(manifest_file, fieldnames=fieldnames)
        writer.writeheader()
        sample_index = 0
        for topology_number, topology in enumerate(topologies, start=1):
            LOGGER.info(
                "case=%s topology=%s (%d/%d) generated_samples=%d elapsed_s=%.1f",
                case_name,
                topology.topology_id,
                topology_number,
                len(topologies),
                sample_index,
                time.perf_counter() - generation_started,
            )
            topology_net = copy.deepcopy(base_net)
            if topology.outaged_line is not None:
                topology_net.line.at[topology.outaged_line, "in_service"] = False
            operators = _canonical_network_operators(base_net, topology.outaged_line)
            _save_npz_atomic(
                topology_dir / f"{topology.topology_id}.npz",
                topology_id=np.asarray(topology.topology_id),
                outaged_line=np.asarray(
                    -1 if topology.outaged_line is None else topology.outaged_line
                ),
                split=np.asarray(split_by_topology[topology.topology_id]),
                **operators,
            )
            n_sites = max(
                len(topology_net.gen.bus.unique()),
                len(topology_net.load.bus.unique()) if topology_net.gen.empty else 0,
            )
            topology_samples = 0
            scenario_attempt = 0
            while (
                topology_samples < samples_per_topology
                and scenario_attempt < max_attempts_per_topology
            ):
                scenario_net = copy.deepcopy(topology_net)
                multipliers = load_sampler.sample()
                profile = renewable.sample(n_sites, scenario_attempt)
                p_load, q_load, p_renewable = apply_scenario(
                    scenario_net,
                    bus_multiplier=multipliers,
                    renewable_capacity_fraction=float(
                        config.get("renewable", {}).get("capacity_fraction", 0.0)
                    ),
                    renewable_profile=profile,
                )
                if bool(config.get("run_dc_baseline", True)):
                    dc_result = solve_dc_opf(copy.deepcopy(scenario_net))
                else:
                    dc_result = ClassicalTiming(
                        converged=False,
                        objective=float("nan"),
                        runtime_s=float("nan"),
                        solver="disabled",
                    )
                result = (
                    persistent_solver.solve(scenario_net)
                    if persistent_solver is not None
                    else solve_ac_opf(scenario_net, backend=backend, **solver_config)
                )
                if not result.converged:
                    failed_rows.append(
                        {
                            "topology_id": topology.topology_id,
                            "scenario_attempt": scenario_attempt,
                            "termination": result.termination,
                        }
                    )
                    scenario_attempt += 1
                    continue
                # Solver calls refresh the ppc lookup. Operators remain topology
                # only; sample injections are mapped to that same canonical order.
                p_load = _bus_ordered(p_load, scenario_net)
                q_load = _bus_ordered(q_load, scenario_net)
                p_renewable = _bus_ordered(p_renewable, scenario_net)
                base_mva = float(operators["base_mva"])
                v_setpoint_bus = np.zeros_like(p_load)
                gen_bus = np.asarray(operators["gen_bus"])
                np.maximum.at(
                    v_setpoint_bus,
                    gen_bus,
                    np.asarray(operators["v_setpoint"], dtype=np.float64),
                )
                bus_features = np.column_stack(
                    [
                        p_load / base_mva,
                        q_load / base_mva,
                        p_renewable / base_mva,
                        np.asarray(operators["bus_type_onehot"]),
                        v_setpoint_bus,
                    ]
                )
                sample_id = f"{sample_index:09d}"
                relative_path = Path("samples") / f"{sample_id}.npz"
                _save_npz_atomic(
                    output_root / relative_path,
                    bus_features=bus_features,
                    p_load_pu=p_load / base_mva,
                    q_load_pu=q_load / base_mva,
                    p_renewable_pu=p_renewable / base_mva,
                    target_vm=result.vm_pu,
                    target_va=result.va_rad,
                    target_pg_pu=result.pg_mw / base_mva,
                    target_qg_pu=result.qg_mvar / base_mva,
                    target_objective=np.asarray(result.objective),
                    solver_runtime_s=np.asarray(result.runtime_s),
                    dc_opf_runtime_s=np.asarray(dc_result.runtime_s),
                    dc_opf_objective=np.asarray(dc_result.objective),
                    dc_opf_converged=np.asarray(dc_result.converged),
                    topology_id=np.asarray(topology.topology_id),
                    scenario_attempt=np.asarray(scenario_attempt),
                )
                writer.writerow(
                    {
                        "sample_id": sample_id,
                        "relative_path": relative_path.as_posix(),
                        "topology_id": topology.topology_id,
                        "scenario_attempt": scenario_attempt,
                        "split": split_by_topology[topology.topology_id],
                        "solver_backend": result.backend,
                        "solver_runtime_s": result.runtime_s,
                        "objective": result.objective,
                        "dc_opf_runtime_s": dc_result.runtime_s,
                        "dc_opf_objective": dc_result.objective,
                        "dc_opf_converged": dc_result.converged,
                        "renewable_source": renewable.source,
                    }
                )
                sample_index += 1
                topology_samples += 1
                scenario_attempt += 1
            if topology_samples < samples_per_topology:
                LOGGER.warning(
                    "case=%s topology=%s reached attempt limit with %d/%d converged samples",
                    case_name,
                    topology.topology_id,
                    topology_samples,
                    samples_per_topology,
                )
            topology_sample_counts[topology.topology_id] = topology_samples

    if persistent_solver is not None:
        persistent_solver.close()

    write_topology_manifest(
        output_root,
        topologies,
        split_by_topology,
        topology_sample_counts,
    )
    write_constraint_inventory(output_root)
    split_topology_counts = {
        split: sum(value == split for value in split_by_topology.values())
        for split in ("train", "val", "test")
    }
    usable_topology_counts = {
        split: sum(
            split_by_topology[topology_id] == split and count > 0
            for topology_id, count in topology_sample_counts.items()
        )
        for split in ("train", "val", "test")
    }
    metadata = {
        "case": case_name,
        "seed": seed,
        "created_unix_s": time.time(),
        "generation_runtime_s": time.perf_counter() - generation_started,
        "n_topologies": len(topologies),
        "n_samples": sample_index,
        "n_failed": len(failed_rows),
        "n_requested_samples": len(topologies) * samples_per_topology,
        "max_scenario_attempts_per_topology": max_attempts_per_topology,
        "topology_sample_counts": topology_sample_counts,
        "split_by_topology": split_by_topology,
        "split_topology_counts": split_topology_counts,
        "usable_topology_counts": usable_topology_counts,
        "solver_requested": backend,
        "renewable_source": renewable.source,
        "config": config,
        "failed_samples": failed_rows,
    }
    (output_root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    LOGGER.info(
        "wrote %d converged samples across %d topologies to %s",
        sample_index,
        len(topologies),
        output_root,
    )
    return manifest_path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--scenarios-per-topology", type=int)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)
    if args.scenarios_per_topology is not None:
        config["scenarios_per_topology"] = args.scenarios_per_topology
    generate_dataset(config)


if __name__ == "__main__":
    main()

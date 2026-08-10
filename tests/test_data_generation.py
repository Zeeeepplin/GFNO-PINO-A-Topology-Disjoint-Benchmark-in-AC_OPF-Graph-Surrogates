from __future__ import annotations

import numpy as np
import pytest

from data_gen.classical_baselines import solve_dc_opf
from data_gen.generate import (
    Topology,
    _canonical_network_operators,
    split_topologies,
)
from data_gen.solver import _write_matpower_case


def test_topology_split_is_disjoint_and_base_is_train() -> None:
    topologies = [Topology("base", None)] + [Topology(f"line_{i}_out", i) for i in range(20)]
    split = split_topologies(topologies, (0.6, 0.2, 0.2), seed=7)
    assert split["base"] == "train"
    train = {key for key, value in split.items() if value == "train"}
    validation = {key for key, value in split.items() if value == "val"}
    test = {key for key, value in split.items() if value == "test"}
    assert not train & validation
    assert not train & test
    assert not validation & test
    assert train | validation | test == set(split)


def test_topology_split_is_reproducible() -> None:
    topologies = [Topology(f"line_{i}_out", i) for i in range(12)]
    assert split_topologies(topologies, (0.5, 0.25, 0.25), seed=11) == split_topologies(
        topologies, (0.5, 0.25, 0.25), seed=11
    )


def test_outage_preserves_edge_axis_and_zeroes_status_and_admittance() -> None:
    networks = pytest.importorskip("pandapower.networks")
    net = networks.case30()
    base = _canonical_network_operators(net, None)
    outage = _canonical_network_operators(net, 0)
    assert np.asarray(base["edge_features"]).shape == np.asarray(outage["edge_features"]).shape
    assert np.asarray(base["edge_features"]).shape[1] == 10
    assert np.asarray(base["edge_features"])[0, -2] == 1
    assert np.asarray(outage["edge_features"])[0, -2] == 0
    assert np.allclose(np.asarray(outage["edge_features"])[0, :8], 0)
    assert np.allclose(np.asarray(outage["y_from_real"])[0], 0)
    assert np.allclose(np.asarray(outage["y_from_imag"])[0], 0)


def test_dc_opf_records_public_pandapower_result_fields() -> None:
    networks = pytest.importorskip("pandapower.networks")
    result = solve_dc_opf(networks.case30())
    assert result.converged
    assert np.isfinite(result.objective)
    assert result.runtime_s > 0


def test_matpower_writer_replaces_missing_generator_mbase(tmp_path) -> None:
    generator = np.zeros((1, 21))
    generator[0, 6] = np.nan
    case = {
        "baseMVA": 100.0,
        "bus": np.zeros((1, 13)),
        "gen": generator,
        "branch": np.zeros((1, 13)),
        "gencost": np.array([[2.0, 0.0, 0.0, 3.0, 0.1, 1.0, 0.0]]),
    }
    path = tmp_path / "case.m"
    _write_matpower_case(path, case)
    text = path.read_text(encoding="utf-8")
    assert "nan" not in text.lower()
    assert "\t100\t" in text

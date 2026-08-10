"""Refresh derived topology tensors without regenerating solver labels."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import yaml

from data_gen.generate import (
    Topology,
    _canonical_network_operators,
    _save_npz_atomic,
    apply_case_adjustments,
    load_ieee_case,
    write_constraint_inventory,
    write_topology_manifest,
)


def refresh(config_path: Path) -> None:
    """Rebuild exact directional edge fields and source constraint metadata."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_root = Path(config["output_dir"]) / config["case"]
    metadata_path = output_root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    split_by_topology = dict(metadata["split_by_topology"])
    sample_counts = {key: int(value) for key, value in metadata["topology_sample_counts"].items()}
    net = load_ieee_case(config["case"])
    apply_case_adjustments(net, config.get("case_adjustments"))
    topologies: list[Topology] = []
    for path in sorted((output_root / "topologies").glob("*.npz")):
        with np.load(path, allow_pickle=False) as payload:
            topology_id = str(payload["topology_id"])
            raw_outage = int(payload["outaged_line"])
        outaged_line = None if raw_outage < 0 else raw_outage
        topology = Topology(topology_id, outaged_line)
        topologies.append(topology)
        operators = _canonical_network_operators(net, outaged_line)
        _save_npz_atomic(
            path,
            topology_id=np.asarray(topology_id),
            outaged_line=np.asarray(raw_outage),
            split=np.asarray(split_by_topology[topology_id]),
            **operators,
        )
    write_topology_manifest(
        output_root,
        topologies,
        split_by_topology,
        sample_counts,
    )
    write_constraint_inventory(output_root)
    metadata["config"] = config
    metadata["split_topology_counts"] = {
        split: sum(value == split for value in split_by_topology.values())
        for split in ("train", "val", "test")
    }
    metadata["usable_topology_counts"] = {
        split: sum(
            split_by_topology[topology_id] == split and count > 0
            for topology_id, count in sample_counts.items()
        )
        for split in ("train", "val", "test")
    }
    metadata["topology_artifact_schema"] = {
        "edge_features": [
            "Re(Yff)",
            "Im(Yff)",
            "Re(Yft)",
            "Im(Yft)",
            "Re(Ytt)",
            "Im(Ytt)",
            "Re(Ytf)",
            "Im(Ytf)",
            "status",
            "rate_pu",
        ],
        "branch_component_type": {"0": "line", "1": "transformer", "2": "other"},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path)
    source.add_argument(
        "--config-root",
        type=Path,
        help="refresh every case*.yaml file in this directory",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    paths = (
        [args.config] if args.config is not None else sorted(args.config_root.glob("case*.yaml"))
    )
    if not paths:
        raise FileNotFoundError(f"no case configuration found under {args.config_root}")
    for path in paths:
        refresh(path)
        print(json.dumps({"refreshed": str(path)}))


if __name__ == "__main__":
    main()

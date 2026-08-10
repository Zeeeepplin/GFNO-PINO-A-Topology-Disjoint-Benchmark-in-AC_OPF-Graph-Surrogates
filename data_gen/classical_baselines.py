"""Classical AC/DC OPF timing helpers for benchmark tables."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ClassicalTiming:
    converged: bool
    objective: float
    runtime_s: float
    solver: str


def solve_dc_opf(net: Any) -> ClassicalTiming:
    """Run pandapower's DC-OPF speed/linearization baseline."""
    import pandapower as pp

    started = time.perf_counter()
    try:
        pp.rundcopp(net, suppress_warnings=True)
        runtime = time.perf_counter() - started
        ppc = net["_ppc"]
        # pandapower 3.x stores DC-OPF convergence and objective on the net,
        # while older PYPOWER-compatible versions also exposed ppc["success"]
        # and ppc["f"]. Prefer the public result fields and retain the fallback
        # so benchmark metadata remains correct across supported versions.
        converged = bool(net.get("OPF_converged", ppc.get("success", False)))
        objective = float(net.get("res_cost", ppc.get("f", float("nan"))))
        return ClassicalTiming(
            converged=converged,
            objective=objective,
            runtime_s=runtime,
            solver="pandapower-dc-opf",
        )
    except Exception:
        return ClassicalTiming(
            False, float("nan"), time.perf_counter() - started, "pandapower-dc-opf"
        )

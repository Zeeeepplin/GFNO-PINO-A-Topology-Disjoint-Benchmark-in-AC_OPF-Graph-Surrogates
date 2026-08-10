"""Classical AC-OPF solver adapters used to create labels and timing baselines."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np


@dataclass(frozen=True)
class OPFResult:
    """Canonical solver output in MATPOWER/PYPOWER ordering."""

    converged: bool
    objective: float
    runtime_s: float
    vm_pu: np.ndarray
    va_rad: np.ndarray
    pg_mw: np.ndarray
    qg_mvar: np.ndarray
    backend: str
    termination: str


def solve_pandapower(net: Any, init: str = "flat") -> OPFResult:
    """Solve AC-OPF using pandapower's PYPOWER primal-dual interior point.

    This is a runnable fallback and useful classical baseline, but it is not
    IPOPT. Publication runs that claim IPOPT must select the PowerModels adapter
    below and preserve the per-sample backend metadata.
    """
    import pandapower as pp
    from pandapower.pypower.idx_bus import VA, VM
    from pandapower.pypower.idx_gen import PG, QG

    started = time.perf_counter()
    try:
        pp.runopp(
            net,
            init=init,
            calculate_voltage_angles=True,
            suppress_warnings=True,
            verbose=False,
            numba=False,
        )
        runtime = time.perf_counter() - started
        ppc = net["_ppc"]
        success = bool(ppc.get("success", net.get("OPF_converged", False)))
        return OPFResult(
            converged=success,
            objective=float(ppc.get("f", np.nan)),
            runtime_s=runtime,
            vm_pu=np.asarray(ppc["bus"][:, VM], dtype=np.float64),
            va_rad=np.deg2rad(np.asarray(ppc["bus"][:, VA], dtype=np.float64)),
            pg_mw=np.asarray(ppc["gen"][:, PG], dtype=np.float64),
            qg_mvar=np.asarray(ppc["gen"][:, QG], dtype=np.float64),
            backend="pandapower-pypower",
            termination="converged" if success else "failed",
        )
    except Exception as exc:  # convergence failures are data, not process failures
        return OPFResult(
            converged=False,
            objective=float("nan"),
            runtime_s=time.perf_counter() - started,
            vm_pu=np.empty(0),
            va_rad=np.empty(0),
            pg_mw=np.empty(0),
            qg_mvar=np.empty(0),
            backend="pandapower-pypower",
            termination=f"{type(exc).__name__}: {exc}",
        )


def solve_powermodels_ipopt(
    net: Any,
    *,
    julia_executable: str = "julia",
    julia_project: str | None = None,
    timeout_s: float = 600.0,
    max_iter: int = 3000,
) -> OPFResult:
    """Solve AC-OPF with PowerModels.jl + IPOPT through a Julia subprocess.

    Julia environments are isolated from Python packaging, making a subprocess
    more reproducible than PyJulia's process-global runtime. Install Julia
    packages with ``julia --project=julia -e 'using Pkg; Pkg.instantiate()'``.
    """
    from pandapower.converter.matpower.to_mpc import to_mpc

    if shutil.which(julia_executable) is None:
        raise FileNotFoundError(
            f"Julia executable {julia_executable!r} was not found; "
            "use solver.backend=pandapower for the clearly labelled fallback"
        )
    project_dir = julia_project or str(Path(__file__).resolve().parents[1] / "julia")
    driver = Path(__file__).with_name("solve_powermodels.jl")
    with tempfile.TemporaryDirectory(prefix="topology_pino_opf_") as temporary:
        temp_dir = Path(temporary)
        input_path = temp_dir / "case_generated.m"
        output_path = temp_dir / "result.json"
        # pandapower's exporter writes MATLAB binary .mat files, while
        # PowerModels.parse_file expects the textual MATPOWER case format.
        # Serialize the standard v2 matrices explicitly and reproducibly.
        mpc = to_mpc(net, filename=None, init="flat")["mpc"]
        _write_matpower_case(input_path, mpc)
        command = [
            julia_executable,
            f"--project={project_dir}",
            str(driver),
            str(input_path),
            str(output_path),
            str(max_iter),
        ]
        started = time.perf_counter()
        process = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        runtime = time.perf_counter() - started
        if process.returncode != 0:
            diagnostic = process.stderr.strip()
            if len(diagnostic) > 1600:
                diagnostic = diagnostic[:800] + "\n...\n" + diagnostic[-800:]
            return OPFResult(
                converged=False,
                objective=float("nan"),
                runtime_s=runtime,
                vm_pu=np.empty(0),
                va_rad=np.empty(0),
                pg_mw=np.empty(0),
                qg_mvar=np.empty(0),
                backend="powermodels-ipopt",
                termination=diagnostic,
            )
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        base_mva = float(mpc["baseMVA"])
        return _powermodels_result(payload, base_mva, runtime)


class PowerModelsIPOPTSession:
    """Persistent PowerModels/Ipopt subprocess for reproducible batch solves.

    Julia package loading and JIT compilation happen once when entering the
    session. Per-sample runtimes therefore measure the warmed solver path,
    which is the relevant classical baseline and avoids adding interpreter
    startup to every scenario.
    """

    def __init__(
        self,
        *,
        julia_executable: str = "julia",
        julia_project: str | None = None,
        timeout_s: float = 600.0,
        max_iter: int = 3000,
    ) -> None:
        if shutil.which(julia_executable) is None:
            raise FileNotFoundError(f"Julia executable {julia_executable!r} was not found")
        self.julia_executable = julia_executable
        self.julia_project = julia_project or str(Path(__file__).resolve().parents[1] / "julia")
        self.timeout_s = timeout_s
        self.max_iter = max_iter
        self._temporary: tempfile.TemporaryDirectory[str] | None = None
        self._process: subprocess.Popen[str] | None = None
        self._stderr_stream: Any | None = None
        self._stderr_path: Path | None = None
        self._counter = 0

    def __enter__(self) -> PowerModelsIPOPTSession:
        self._temporary = tempfile.TemporaryDirectory(prefix="topology_pino_ipopt_session_")
        temporary_path = Path(self._temporary.name)
        self._stderr_path = temporary_path / "server.stderr"
        self._stderr_stream = self._stderr_path.open("w+", encoding="utf-8")
        server = Path(__file__).with_name("solve_powermodels_server.jl")
        command = [
            self.julia_executable,
            f"--project={self.julia_project}",
            str(server),
            str(self.max_iter),
        ]
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_stream,
            text=True,
            bufsize=1,
        )
        assert self._process.stdout is not None
        ready, messages = self._read_protocol({"__TOPOLOGY_PINO_READY__"})
        if ready != "__TOPOLOGY_PINO_READY__":
            self.close()
            raise RuntimeError(
                f"PowerModels server failed to start; response={ready!r}; "
                f"output={messages[-1200:]!r}"
            )
        return self

    def solve(self, net: Any) -> OPFResult:
        """Solve one pandapower network through the warmed Julia process."""
        from pandapower.converter.matpower.to_mpc import to_mpc

        if self._process is None or self._process.poll() is not None:
            raise RuntimeError("PowerModels session is not running")
        assert self._temporary is not None
        assert self._process.stdin is not None
        assert self._process.stdout is not None
        temporary_path = Path(self._temporary.name)
        input_path = temporary_path / f"case_{self._counter:09d}.m"
        output_path = temporary_path / f"result_{self._counter:09d}.json"
        self._counter += 1
        mpc = to_mpc(net, filename=None, init="flat")["mpc"]
        _write_matpower_case(input_path, mpc)
        started = time.perf_counter()
        self._process.stdin.write(f"{input_path}\t{output_path}\n")
        self._process.stdin.flush()
        response, protocol_output = self._read_protocol(
            {"__TOPOLOGY_PINO_OK__", "__TOPOLOGY_PINO_ERROR__"}
        )
        runtime = time.perf_counter() - started
        if not output_path.exists():
            diagnostic = self._read_stderr()
            return OPFResult(
                converged=False,
                objective=float("nan"),
                runtime_s=runtime,
                vm_pu=np.empty(0),
                va_rad=np.empty(0),
                pg_mw=np.empty(0),
                qg_mvar=np.empty(0),
                backend="powermodels-ipopt",
                termination=(
                    f"server response={response!r}; output={protocol_output[-600:]!r}; "
                    f"{diagnostic[-600:]}"
                ),
            )
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        if response != "__TOPOLOGY_PINO_OK__":
            payload["termination_status"] = (
                f"{payload.get('termination_status', 'ERROR')}: "
                f"{payload.get('error', protocol_output or response)}"
            )
        return _powermodels_result(payload, float(mpc["baseMVA"]), runtime)

    def _read_protocol(self, markers: set[str]) -> tuple[str, str]:
        assert self._process is not None
        assert self._process.stdout is not None
        messages: list[str] = []
        while True:
            raw = self._process.stdout.readline()
            if raw == "":
                return "", "\n".join(messages)
            line = raw.strip()
            if line in markers:
                return line, "\n".join(messages)
            if line:
                messages.append(line)

    def _read_stderr(self) -> str:
        if self._stderr_stream is None or self._stderr_path is None:
            return ""
        self._stderr_stream.flush()
        return self._stderr_path.read_text(encoding="utf-8", errors="replace")

    def close(self) -> None:
        if self._process is not None and self._process.poll() is None:
            try:
                if self._process.stdin is not None:
                    self._process.stdin.write("__QUIT__\n")
                    self._process.stdin.flush()
                self._process.wait(timeout=30)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                self._process.kill()
                self._process.wait(timeout=10)
        if self._stderr_stream is not None:
            self._stderr_stream.close()
        if self._temporary is not None:
            self._temporary.cleanup()
        self._process = None
        self._stderr_stream = None
        self._temporary = None

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def _powermodels_result(
    payload: dict[str, Any],
    base_mva: float,
    runtime_s: float,
) -> OPFResult:
    solution = payload.get("solution", {})
    buses = sorted(solution.get("bus", {}).items(), key=lambda item: int(item[0]))
    generators = sorted(solution.get("gen", {}).items(), key=lambda item: int(item[0]))
    status = str(payload.get("termination_status", "unknown"))
    converged = status.upper() in {"LOCALLY_SOLVED", "ALMOST_LOCALLY_SOLVED", "OPTIMAL"}
    objective_value = payload.get("objective")
    objective = float(objective_value) if objective_value is not None else float("nan")
    return OPFResult(
        converged=converged,
        objective=objective,
        runtime_s=runtime_s,
        vm_pu=np.asarray([row["vm"] for _, row in buses], dtype=np.float64),
        # PowerModels solutions use radians and per-unit powers internally.
        va_rad=np.asarray([row.get("va", 0.0) for _, row in buses], dtype=np.float64),
        pg_mw=base_mva * np.asarray([row["pg"] for _, row in generators], dtype=np.float64),
        qg_mvar=base_mva * np.asarray([row["qg"] for _, row in generators], dtype=np.float64),
        backend="powermodels-ipopt",
        termination=status,
    )


def solve_ac_opf(
    net: Any,
    backend: Literal["pandapower", "powermodels"] = "pandapower",
    **kwargs: Any,
) -> OPFResult:
    """Dispatch to a configured AC-OPF backend."""
    if backend == "pandapower":
        return solve_pandapower(net, **kwargs)
    if backend == "powermodels":
        return solve_powermodels_ipopt(net, **kwargs)
    raise ValueError(f"unknown OPF backend: {backend}")


def _write_matpower_case(path: Path, mpc: dict[str, Any]) -> None:
    """Write the standard MATPOWER v2 fields accepted by PowerModels."""

    base_mva = float(mpc["baseMVA"])
    bus = np.asarray(mpc["bus"], dtype=np.float64)
    gen = np.asarray(mpc["gen"], dtype=np.float64).copy()
    branch = np.asarray(mpc["branch"], dtype=np.float64)
    gencost = np.asarray(mpc["gencost"], dtype=np.float64)
    # pandapower leaves the optional generator MBASE column as NaN for some
    # imported MATPOWER cases. MATPOWER defines a zero/unspecified MBASE as
    # system baseMVA, while PowerModels' text parser rejects NaN in this field.
    if gen.shape[1] > 6:
        gen[~np.isfinite(gen[:, 6]), 6] = base_mva
    for name, values in {
        "bus": bus,
        "gen": gen,
        "branch": branch,
        "gencost": gencost,
    }.items():
        if not np.isfinite(values).all():
            locations = np.argwhere(~np.isfinite(values))
            raise ValueError(
                f"MATPOWER {name} contains non-finite values at {locations[:10].tolist()}"
            )

    def matrix(name: str, values: np.ndarray, columns: int | None = None) -> str:
        array = np.asarray(values)
        if columns is not None:
            array = array[:, :columns]
        rows = ["\t" + "\t".join(f"{float(value):.16g}" for value in row) + ";" for row in array]
        return f"mpc.{name} = [\n" + "\n".join(rows) + "\n];\n"

    lines = [
        "function mpc = case_generated\n",
        "mpc.version = '2';\n",
        f"mpc.baseMVA = {base_mva:.16g};\n",
        matrix("bus", bus, 13),
        matrix("gen", gen, 21),
        matrix("branch", branch, 13),
        matrix("gencost", gencost),
    ]
    path.write_text("".join(lines), encoding="utf-8", newline="\n")

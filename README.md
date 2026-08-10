# Physics-Informed Fine-Tuning for AC-OPF Surrogates Under Line Outages

Publication-grade PyTorch implementation of the GFNO-PINO benchmark described
in the accompanying two-column Elsevier manuscript. The model consumes bus
fields, directional two-port branch admittances, line status, limits, and
generator economics, then predicts voltage phasors and dispatch under unseen
non-islanding line outages.

GFNO-PINO is the archived experiment name. Its Chebyshev layer is a localized
graph-spectral convolution, and the benchmark trains one checkpoint per grid
size; the repository therefore does not claim discretization transfer or a
shared neural operator across IEEE 30--300.

The implementation makes its scope explicit:

- complete topology IDs, rather than scenario rows, are assigned to
  train/validation/test;
- the predictor receives
  `Re/Im(Yff,Yft,Ytt,Ytf), status, rate_pu`, preserving transformer tap and
  phase-shift directionality;
- voltage and generator P/Q boxes and the slack reference are enforced by
  parameterization;
- complex-power balance, thermal limits, and branch angle-difference limits are
  optimized as soft objectives, are not hard-guaranteed, and are measured after
  every inference;
- zero-valued hard-box diagnostics are excluded from the GradNorm active set,
  eliminating undefined zero-over-zero training rates;
- all learned results use five seeds (2026--2030) and include a
  parameter-matched topology-blind GFNO trained on the same rows.

See [DESIGN.md](DESIGN.md) for the Chebyshev-versus-eigenbasis decision.

## Released benchmark

`data/pilot`, `runs/revision`, and `results/revision` contain the executed
review-revision benchmark:

- 564 converged PowerModels--IPOPT AC-OPF labels for IEEE 30, 57, 118, and 300;
- 5 neural models × 5 seeds × 4 systems = 100 checkpoints;
- 36 held-out scenarios on three unseen outage topologies per system;
- 3600 neural test evaluations;
- exact topology and source-constraint inventories;
- per-sample, per-seed, and aggregate metrics;
- 25 numerical/unit tests focused on AC physics, topology conditioning, and
  aggregation;
- SHA-256 hashes for released inputs and generated artifacts.

The paper reports device-specific surrogate latency, not an operational
AC-OPF speedup. Preprocessing, transfer, feasibility verification, and
restoration are excluded from forward-pass timing. A prediction that fails the
declared feasibility proxy cannot replace a solver result.

## Repository layout

```text
data_gen/                    scenario generation and topology artifact refresh
models/                      GFNO-PINO and neural baselines
losses/                      AC equations, constraints, adaptive weighting
configs/pilot/               exact executed case configurations
data/pilot/                  solver labels and topology/constraint manifests
runs/revision/               five-seed checkpoints and histories
results/revision/            per-sample and aggregate evaluations
tests/                       analytic and IEEE-case numerical verification
manuscript/                  Elsevier source, generated tables, and figures
manuscript/cover_letter.tex  submission cover letter
train.py                     warm start and physics fine-tuning
evaluate.py                  topology-disjoint metrics and latency
run_revision_benchmark.py    resumable 100-run benchmark driver
aggregate_revision_results.py regenerated tables, plots, and aggregate CSVs
analyze_active_bounds.py      active/inactive bound error stratification
reproduce_submission.py      one-command reproduction driver
build_artifact_manifest.py   deterministic SHA-256 inventory
build_submission_archive.py  clean deterministic submission ZIP
```

## Reference environment

Python 3.12.13 is the reported benchmark version; Python 3.10--3.12 is
supported.

```powershell
conda env create -f environment.yml
conda activate topology-pino-opf
julia --project=julia -e "using Pkg; Pkg.instantiate()"
```

PowerModels--IPOPT requires Julia 1.10. The included `Project.toml` and
`Manifest.toml` lock PowerModels, JuMP, IPOPT, and their dependencies.
`requirements-lock.txt` records the exact Python 3.12.13 package versions used
for the reported Windows/CUDA runs; the unpinned files remain the portable
installation specification.
`solver.backend: pandapower` is available for pipeline debugging but invokes
PYPOWER, not IPOPT, and must not be merged into IPOPT-labelled results.

## One-command reproduction

From the repository root:

```powershell
python reproduce_submission.py --device cuda
```

The command:

1. runs the complete test suite and writes
   `results/revision/aggregate/test_report.txt`;
2. reconstructs directional topology artifacts and source-constraint
   inventories from the supplied case/configuration files;
3. resumes or executes all 100 training/evaluation runs;
4. regenerates aggregate CSVs, LaTeX tables, and PDF/PNG figures;
5. compiles `manuscript/main.pdf` with `latexmk` or `tectonic`;
6. writes `artifact_manifest.csv`, its summary hash, and the clean submission
   ZIP.

Existing complete runs are reused. Add `--force` only when deliberately
retraining every checkpoint. Use `--skip-pdf` on machines without TeX Live.

## Data generation

Generate a case from its complete configuration with:

```powershell
python -m data_gen.generate --config configs/case30.yaml
```

Each case directory contains:

```text
topologies/*.npz          exact Ybus, Yf/Yt, directional edge fields and limits
samples/*.npz             inputs and IPOPT decision labels
manifest.csv              split, topology, backend, runtime, objective
topology_manifest.csv     exact outage eligibility, split and sample count
constraint_inventory.json source voltage/thermal/generator/angle constraints
metadata.json             configuration and failed convergence attempts
```

Every in-service physical line outage is connectivity-tested while retaining
transformers and parallel circuits. Transformers are represented in the graph
and exact physics with status one. Each topology artifact stores the full
line-plus-transformer branch axis, component type, and physical-line index map.
Edge encoding and Laplacian construction mask by full-branch status and divide
incident messages by the active post-outage degree. Transformers remain outside
the outage-candidate set.
The realized topology split is 8/1/3 (66.7%/8.3%/25.0%) in every case. IEEE 30
has 7/1/3 usable topology IDs because one assigned training outage has no
converged samples. The exact assignments, not nominal percentages, are
authoritative.

Loads use bounded graph-correlated perturbations. The released benchmark uses
profiles tagged `synthetic-diurnal`; it does not claim use of operational NREL
data. To use an externally licensed renewable series, set
`renewable.csv_path` and retain its provenance alongside the generated data.

## Models and training

The benchmark driver runs:

- `pino`: topology-conditioned GFNO with physics fine-tuning;
- `data_only_gfno`: identical topology-conditioned GFNO trained on labels only;
- `gnn`: topology-conditioned spatial message-passing ablation;
- `topology_blind_gfno`: matched GFNO receiving one fixed base topology while
  training on the same multi-topology rows;
- `fixed_pinn`: base-only dense reference, explicitly treated as a confounded
  out-of-distribution comparison rather than the causal topology ablation.

The 12 warm-start epochs optimize supervised error. During 18 physics epochs,
power-balance and thermal objectives are multiplied by `1 + 9t` and `1 + 4t`;
economic, balance, thermal, angle, and supervised losses then use zero-safe
adaptive weights. The economic objective is evaluated per sample against its
locally converged reference using a one-currency-unit-per-hour denominator
floor, so batch samples cannot cancel one another. Inactive adaptive objectives
receive zero weight and do not enter the active-set normalization. Voltage and
generator box diagnostics are logged but excluded from the optimizer because
the hard head makes them identically zero.

## Metrics and units

Network quantities and admittance operators are per unit; angles are radians;
generator cost is evaluated in the source polynomial's MW/currency-per-hour
units; latency is milliseconds. For bus `i` and sample `b`, the reported
complex-power-balance mismatch is

```text
sqrt(rP[i,b]^2 + rQ[i,b]^2)
```

The evaluator reports its per-bus mean, sample maximum, P95, worst sample, and
separate active/reactive maxima. It also reports signed and absolute
reference-cost difference, voltage/thermal/generator/angle violation rate and
magnitude, label MAE, AC feasibility proxy, neural latency, and stored DC/AC
solver latency. The power-balance tolerance is `1e-3` p.u.; every inequality
tolerance is `1e-6` (radians for angle and p.u. otherwise). Statistics are
computed within each seed before reporting five-seed means and sample standard
deviations. No
telescoping cycle-angle identity is presented as a learned KVL metric.

## Limits

- The hard head does not project onto the coupled non-convex AC-feasible set.
- The benchmark covers selected non-islanding N−1 line outages, not all N−1
  elements, islanded states, bus additions, or N-k combinations.
- One checkpoint is trained per system size; this is not evidence of a single
  model transferring from 30 to 300 buses.
- Dense padded Laplacians are practical for these cases but do not establish
  scalability to much larger interconnections.
- IPOPT references are locally converged points, not global-optimality
  certificates.

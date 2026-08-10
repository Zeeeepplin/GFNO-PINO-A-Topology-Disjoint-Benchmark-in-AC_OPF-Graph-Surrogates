"""Aggregate five-seed reviewer-revision results into auditable paper artifacts."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CASES = ("case30", "case57", "case118", "case300")
MODELS = (
    "fixed_pinn",
    "topology_blind_gfno",
    "gnn",
    "data_only_gfno",
    "pino",
)
MODEL_LABELS = {
    "fixed_pinn": "Base-only PINN (OOD)",
    "topology_blind_gfno": "Topology-blind GFNO",
    "gnn": "Plain GNN",
    "data_only_gfno": "Data-only GFNO",
    "pino": "GFNO-PINO",
}
BUS_COUNTS = {"case30": 30, "case57": 57, "case118": 118, "case300": 300}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "legend.fontsize": 7.2,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
        }
    )


def _save(fig: plt.Figure, directory: Path, stem: str) -> None:
    fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(directory / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def _seed_record(
    frame: pd.DataFrame,
    run_metadata: dict[str, Any],
    case: str,
    model: str,
    seed: int,
) -> dict[str, Any]:
    return {
        "case": case,
        "n_bus": BUS_COUNTS[case],
        "model_kind": model,
        "model": MODEL_LABELS[model],
        "seed": seed,
        "n_samples": len(frame),
        "n_topologies": frame["topology_id"].nunique(),
        "signed_reference_difference_mean_percent": frame[
            "signed_reference_cost_difference_percent"
        ].mean(),
        "signed_reference_difference_p95_percent": frame[
            "signed_reference_cost_difference_percent"
        ].quantile(0.95),
        "absolute_reference_difference_mean_percent": frame[
            "absolute_reference_cost_difference_percent"
        ].mean(),
        "absolute_reference_difference_p95_percent": frame[
            "absolute_reference_cost_difference_percent"
        ].quantile(0.95),
        "feasible_reference_difference_mean_percent": frame[
            "feasible_reference_cost_difference_percent"
        ].mean(),
        "balance_mean_pu": frame["power_balance_mean_pu"].mean(),
        "balance_max_mean_pu": frame["power_balance_max_pu"].mean(),
        "balance_max_p95_pu": frame["power_balance_max_pu"].quantile(0.95),
        "balance_max_worst_pu": frame["power_balance_max_pu"].max(),
        "balance_violation_rate_percent": 100.0 * frame["power_balance_violation_rate"].mean(),
        "p_balance_max_mean_pu": frame["p_balance_max_pu"].mean(),
        "q_balance_max_mean_pu": frame["q_balance_max_pu"].mean(),
        "voltage_violation_rate_percent": 100.0 * frame["voltage_violation_rate"].mean(),
        "voltage_violation_max_pu": frame["voltage_violation_max_pu"].max(),
        "thermal_violation_rate_percent": 100.0 * frame["thermal_violation_rate"].mean(),
        "thermal_violation_max_pu": frame["thermal_violation_max_pu"].max(),
        "generator_violation_rate_percent": 100.0 * frame["generator_violation_rate"].mean(),
        "generator_violation_max_pu": frame["generator_violation_max_pu"].max(),
        "angle_violation_rate_percent": 100.0 * frame["angle_violation_rate"].mean(),
        "angle_violation_max_rad": frame["angle_violation_max_rad"].max(),
        "ac_feasible_rate_percent": 100.0 * frame["ac_feasible_proxy"].mean(),
        "latency_median_ms": frame["inference_latency_ms"].median(),
        "latency_q25_ms": frame["inference_latency_ms"].quantile(0.25),
        "latency_q75_ms": frame["inference_latency_ms"].quantile(0.75),
        "vm_mae": frame["vm_mae"].mean(),
        "va_mae_rad": frame["va_mae_rad"].mean(),
        "pg_mae_pu": frame["pg_mae_pu"].mean(),
        "qg_mae_pu": frame["qg_mae_pu"].mean(),
        "parameter_count": run_metadata["parameter_count"],
        "training_runtime_s": run_metadata["training_runtime_s"],
    }


def _aggregate_seed_records(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    identity = {
        "case",
        "n_bus",
        "model_kind",
        "model",
        "seed",
        "n_samples",
        "n_topologies",
        "parameter_count",
    }
    numeric = [
        column
        for column in seed_metrics.select_dtypes(include=[np.number]).columns
        if column not in identity
    ]
    for (case, model), frame in seed_metrics.groupby(["case", "model_kind"]):
        record: dict[str, Any] = {
            "case": case,
            "n_bus": BUS_COUNTS[case],
            "model_kind": model,
            "model": MODEL_LABELS[model],
            "n_seeds": frame["seed"].nunique(),
            "seeds": ",".join(str(value) for value in sorted(frame["seed"].unique())),
            "n_samples_per_seed": int(frame["n_samples"].iloc[0]),
            "n_topologies": int(frame["n_topologies"].iloc[0]),
            "parameter_count": int(frame["parameter_count"].iloc[0]),
        }
        for column in numeric:
            record[column] = frame[column].mean()
            record[f"{column}_seed_std"] = frame[column].std(ddof=1)
        records.append(record)
    return pd.DataFrame(records)


def _paired_physics_effect(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize paired data-only/PINO balance changes across identical seeds."""
    records: list[dict[str, Any]] = []
    for case in CASES:
        data_only = (
            seed_metrics.loc[
                (seed_metrics["case"] == case)
                & (seed_metrics["model_kind"] == "data_only_gfno")
            ]
            .set_index("seed")["balance_max_mean_pu"]
            .sort_index()
        )
        pino = (
            seed_metrics.loc[
                (seed_metrics["case"] == case) & (seed_metrics["model_kind"] == "pino")
            ]
            .set_index("seed")["balance_max_mean_pu"]
            .sort_index()
        )
        if not data_only.index.equals(pino.index):
            raise RuntimeError(f"unpaired data-only/PINO seeds for {case}")
        paired_reduction = 100.0 * (data_only - pino) / data_only
        rng = np.random.default_rng(20260728 + BUS_COUNTS[case])
        indices = rng.integers(0, len(paired_reduction), size=(10_000, len(paired_reduction)))
        bootstrap_means = paired_reduction.to_numpy()[indices].mean(axis=1)
        records.append(
            {
                "case": case,
                "n_bus": BUS_COUNTS[case],
                "n_paired_seeds": len(paired_reduction),
                "data_only_balance_max_mean_pu": data_only.mean(),
                "data_only_balance_max_seed_std_pu": data_only.std(ddof=1),
                "pino_balance_max_mean_pu": pino.mean(),
                "pino_balance_max_seed_std_pu": pino.std(ddof=1),
                "paired_reduction_mean_percent": paired_reduction.mean(),
                "paired_reduction_seed_std_percent": paired_reduction.std(ddof=1),
                "paired_seed_reductions_percent": ",".join(
                    f"{value:.1f}" for value in paired_reduction.to_numpy()
                ),
                "paired_reduction_resampling_low_percent": np.quantile(
                    bootstrap_means, 0.025
                ),
                "paired_reduction_resampling_high_percent": np.quantile(
                    bootstrap_means, 0.975
                ),
            }
        )
    return pd.DataFrame(records)


def _dataset_summary(data_root: Path) -> pd.DataFrame:
    records = []
    for case in CASES:
        metadata = json.loads((data_root / case / "metadata.json").read_text(encoding="utf-8"))
        manifest = pd.read_csv(data_root / case / "manifest.csv")
        topology = pd.read_csv(data_root / case / "topology_manifest.csv")
        records.append(
            {
                "case": case,
                "n_bus": BUS_COUNTS[case],
                "requested_samples": metadata["n_requested_samples"],
                "converged_samples": metadata["n_samples"],
                "failed_attempts": metadata["n_failed"],
                "train_topologies_all": int((topology["split"] == "train").sum()),
                "validation_topologies_all": int((topology["split"] == "val").sum()),
                "test_topologies_all": int((topology["split"] == "test").sum()),
                "train_topologies_usable": int(
                    ((topology["split"] == "train") & (topology["converged_samples"] > 0)).sum()
                ),
                "validation_topologies_usable": int(
                    ((topology["split"] == "val") & (topology["converged_samples"] > 0)).sum()
                ),
                "test_topologies_usable": int(
                    ((topology["split"] == "test") & (topology["converged_samples"] > 0)).sum()
                ),
                "train_samples": int((manifest["split"] == "train").sum()),
                "validation_samples": int((manifest["split"] == "val").sum()),
                "test_samples": int((manifest["split"] == "test").sum()),
                "solver_backend": manifest["solver_backend"].iloc[0],
                "renewable_source": manifest["renewable_source"].iloc[0],
                "load_min": metadata["config"]["load_perturbation"]["min"],
                "load_max": metadata["config"]["load_perturbation"]["max"],
                "renewable_capacity_fraction": metadata["config"]["renewable"]["capacity_fraction"],
            }
        )
    return pd.DataFrame(records)


def _classical_metrics(frames: dict[tuple[str, str, int], pd.DataFrame]) -> pd.DataFrame:
    records = []
    for case in CASES:
        key = next(key for key in frames if key[0] == case and key[1] == "pino")
        frame = frames[key]
        dc_difference = frame["dc_reference_cost_difference_percent"]
        for method, latency, signed, absolute in [
            ("IPOPT AC-OPF", "classical_ac_latency_ms", 0.0, 0.0),
            (
                "DC-OPF",
                "classical_dc_latency_ms",
                dc_difference.mean(),
                dc_difference.abs().mean(),
            ),
        ]:
            records.append(
                {
                    "case": case,
                    "n_bus": BUS_COUNTS[case],
                    "method": method,
                    "signed_reference_difference_mean_percent": signed,
                    "absolute_reference_difference_mean_percent": absolute,
                    "latency_median_ms": frame[latency].median(),
                    "latency_q25_ms": frame[latency].quantile(0.25),
                    "latency_q75_ms": frame[latency].quantile(0.75),
                }
            )
    return pd.DataFrame(records)


def _plot_training(runs_root: Path, figure_dir: Path) -> None:
    _style()
    fig, axis = plt.subplots(figsize=(7.1, 3.2), constrained_layout=True)
    for case in CASES:
        histories = [
            pd.read_csv(path)
            for path in sorted((runs_root / case / "pino").glob("seed_*/history.csv"))
        ]
        values = np.stack([frame["validation_powerflow"].to_numpy() for frame in histories])
        epoch = histories[0]["epoch"].to_numpy() + 1
        mean = values.mean(axis=0)
        std = values.std(axis=0, ddof=1)
        axis.semilogy(epoch, mean, marker="o", markersize=2.2, label=f"IEEE {BUS_COUNTS[case]}")
        axis.fill_between(epoch, np.maximum(mean - std, 1e-8), mean + std, alpha=0.14)
    axis.axvline(12, color="0.35", linestyle="--", linewidth=0.9)
    axis.text(12.3, axis.get_ylim()[1] / 1.8, "physics fine-tuning", color="0.3")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Validation power-balance loss")
    axis.grid(True, which="both", alpha=0.22)
    axis.legend(ncol=2, frameon=False)
    _save(fig, figure_dir, "fig_residual_convergence")

    histories = [
        pd.read_csv(path)
        for path in sorted((runs_root / "case30" / "pino").glob("seed_*/history.csv"))
    ]
    fig, axis = plt.subplots(figsize=(7.1, 3.2), constrained_layout=True)
    curves = {
        "Economic (adaptive)": "weight_economic",
        "Balance (adaptive x curriculum)": "effective_weight_powerflow",
        "Thermal (adaptive x curriculum)": "effective_weight_thermal",
        "Supervised (adaptive)": "weight_supervised",
    }
    epoch = histories[0]["epoch"].to_numpy() + 1
    line_styles = ("-", "--", "-.", ":")
    markers = ("o", "s", "^", "D")
    for index, (label, column) in enumerate(curves.items()):
        values = np.stack([frame[column].to_numpy() for frame in histories])
        mean = values.mean(axis=0)
        std = values.std(axis=0, ddof=1)
        axis.semilogy(
            epoch,
            mean,
            label=label,
            linestyle=line_styles[index],
            marker=markers[index],
            markevery=3,
            markersize=3.0,
        )
        axis.fill_between(epoch, np.maximum(mean - std, 1e-8), mean + std, alpha=0.12)
    axis.axvline(12, color="0.35", linestyle="--", linewidth=0.9)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Effective objective weight")
    axis.grid(True, which="both", alpha=0.22)
    axis.legend(ncol=2, frameon=False)
    _save(fig, figure_dir, "fig_adaptive_weights")


def _plot_scalability(metrics: pd.DataFrame, classical: pd.DataFrame, figure_dir: Path) -> None:
    _style()
    fig, (latency_axis, balance_axis) = plt.subplots(
        1, 2, figsize=(7.1, 3.0), constrained_layout=True
    )
    for model in MODELS:
        selected = metrics.loc[metrics["model_kind"] == model].sort_values("n_bus")
        latency_axis.errorbar(
            selected["n_bus"],
            selected["latency_median_ms"],
            yerr=selected["latency_median_ms_seed_std"],
            marker="o",
            label=MODEL_LABELS[model],
        )
        balance_axis.errorbar(
            selected["n_bus"],
            selected["balance_max_mean_pu"],
            yerr=selected["balance_max_mean_pu_seed_std"],
            marker="o",
            label=MODEL_LABELS[model],
        )
    for method, linestyle in [("DC-OPF", "--"), ("IPOPT AC-OPF", ":")]:
        selected = classical.loc[classical["method"] == method].sort_values("n_bus")
        latency_axis.plot(
            selected["n_bus"],
            selected["latency_median_ms"],
            marker="s",
            linestyle=linestyle,
            label=method,
        )
    latency_axis.set_yscale("log")
    balance_axis.set_yscale("log")
    latency_axis.set_xlabel("Bus count")
    balance_axis.set_xlabel("Bus count")
    latency_axis.set_ylabel("Device-specific latency (ms)")
    balance_axis.set_ylabel("Mean max. power-balance mismatch (p.u.)")
    latency_axis.grid(True, which="both", alpha=0.22)
    balance_axis.grid(True, which="both", alpha=0.22)
    latency_axis.legend(frameon=False, fontsize=6.2, ncol=2)
    balance_axis.legend(frameon=False, fontsize=6.2)
    _save(fig, figure_dir, "fig_scalability")


def _plot_architecture(figure_dir: Path) -> None:
    """Draw the exact directional, topology-conditioned data flow used in code."""
    _style()
    fig, axis = plt.subplots(figsize=(11.0, 4.0), constrained_layout=True)
    axis.set_xlim(0, 11)
    axis.set_ylim(0, 4)
    axis.axis("off")
    boxes = [
        (0.2, 2.45, 2.0, 1.0, "Bus field\nloads, renewable,\nbus type, setpoint", "#dceaf7"),
        (
            0.2,
            0.55,
            2.0,
            1.25,
            "Directional edge field\n$Y^{ff},Y^{ft},Y^{tt},Y^{tf}$\nstatus, rating",
            "#fbe3c5",
        ),
        (2.8, 1.5, 1.7, 1.0, "Endpoint-specific\nedge encoding\n+ bus lifting", "#e5e0f4"),
        (
            5.0,
            1.5,
            1.8,
            1.0,
            "Chebyshev GFNO\n$K$ terms, degree $K-1$\n+ pointwise path",
            "#dcefdc",
        ),
        (
            7.3,
            2.45,
            1.7,
            1.0,
            "Bus heads\n$\\widehat V,\\widehat\\theta$\nbox + slack maps",
            "#f2dce8",
        ),
        (7.3, 0.55, 1.7, 1.25, "Generator head\nlimits, setpoint,\ncosts, identity", "#f2dce8"),
        (9.5, 1.5, 1.3, 1.0, "Exact AC checks\n$Y,Y_f,Y_t$\nbalance/limits", "#eadfd5"),
    ]
    for x, y, width, height, label, color in boxes:
        patch = plt.Rectangle(
            (x, y), width, height, facecolor=color, edgecolor="0.25", linewidth=0.8
        )
        axis.add_patch(patch)
        axis.text(x + width / 2, y + height / 2, label, ha="center", va="center")
    arrows = [
        ((2.2, 2.95), (2.8, 2.25)),
        ((2.2, 1.15), (2.8, 1.75)),
        ((4.5, 2.0), (5.0, 2.0)),
        ((6.8, 2.0), (7.3, 2.95)),
        ((6.8, 2.0), (7.3, 1.15)),
        ((9.0, 2.95), (9.5, 2.25)),
        ((9.0, 1.15), (9.5, 1.75)),
    ]
    for start, end in arrows:
        axis.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={"arrowstyle": "->", "linewidth": 1.0, "color": "0.25"},
        )
    axis.text(
        5.9,
        3.45,
        "Sample-specific Laplacian from active transfer admittances",
        ha="center",
        va="center",
        fontsize=8,
    )
    axis.annotate(
        "",
        xy=(5.9, 2.5),
        xytext=(5.9, 3.25),
        arrowprops={"arrowstyle": "->", "linewidth": 0.9, "color": "0.25"},
    )
    axis.text(
        5.5,
        0.18,
        "An outage changes inputs and exact physics operators; learned parameters are unchanged.",
        ha="center",
        va="center",
        fontsize=8,
        style="italic",
    )
    _save(fig, figure_dir, "fig_architecture")


def _fmt(mean: float, std: float, digits: int = 2) -> str:
    if not np.isfinite(mean):
        return r"\textemdash"
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def _write_tables(
    metrics: pd.DataFrame,
    classical: pd.DataFrame,
    paired_effect: pd.DataFrame,
    output_dir: Path,
) -> None:
    headers = " & ".join(f"IEEE {BUS_COUNTS[case]}" for case in CASES)
    dc = classical.loc[classical["method"] == "DC-OPF"].set_index("case")
    rows = [
        "DC-OPF & "
        + " & ".join(
            f"{dc.at[case, 'signed_reference_difference_mean_percent']:.2f}"
            for case in CASES
        )
        + r" \\"
    ]
    for model in MODELS:
        selected = metrics.loc[metrics["model_kind"] == model].set_index("case")
        rows.append(
            f"{MODEL_LABELS[model]} & "
            + " & ".join(
                _fmt(
                    selected.at[case, "signed_reference_difference_mean_percent"],
                    selected.at[
                        case,
                        "signed_reference_difference_mean_percent_seed_std",
                    ],
                )
                for case in CASES
            )
            + r" \\"
        )
    (output_dir / "table_optimality.tex").write_text(
        "\n".join(
            [
                r"\begin{table*}[t]",
                r"\centering",
                (
                    r"\caption{Signed reference-cost difference (\%) on "
                    r"topology-disjoint test sets. "
                    r"Neural entries are mean $\pm$ standard deviation across five "
                    r"training seeds; negative values from infeasible predictions "
                    r"are not optimality gains.}"
                ),
                r"\label{tab:optimality}",
                r"\scriptsize",
                r"\begin{tabular}{lrrrr}",
                r"\toprule",
                f"Method & {headers}" + r" \\",
                r"\midrule",
                *rows,
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table*}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    rows = []
    for case in CASES:
        row = metrics.loc[(metrics["case"] == case) & (metrics["model_kind"] == "pino")].iloc[0]
        voltage = (
            f"{row['voltage_violation_rate_percent']:.2f}/{row['voltage_violation_max_pu']:.2e}"
        )
        thermal = (
            f"{row['thermal_violation_rate_percent']:.2f}/{row['thermal_violation_max_pu']:.2e}"
        )
        generator = (
            f"{row['generator_violation_rate_percent']:.2f}/{row['generator_violation_max_pu']:.2e}"
        )
        angle = f"{row['angle_violation_rate_percent']:.2f}/{row['angle_violation_max_rad']:.2e}"
        rows.append(
            f"IEEE {BUS_COUNTS[case]} & "
            f"{voltage} & {thermal} & {generator} & {angle} & "
            f"{row['ac_feasible_rate_percent']:.1f}" + r" \\"
        )
    (output_dir / "table_constraints.tex").write_text(
        "\n".join(
            [
                r"\begin{table*}[t]",
                r"\centering",
                (
                    r"\caption{GFNO-PINO inequality diagnostics on unseen topologies. "
                    r"Each entry is violation rate (\%)/mean seed-maximum excess "
                    r"(p.u., except angle in rad). The AC-feasible "
                    r"column additionally requires a $10^{-3}$ p.u. maximum "
                    r"complex-power-balance mismatch.}"
                ),
                r"\label{tab:violations}",
                r"\scriptsize",
                r"\begin{tabular}{lrrrrr}",
                r"\toprule",
                (r"System & Voltage & Thermal & Generator & Angle & AC feasible (\%) \\"),
                r"\midrule",
                *rows,
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table*}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    rows = []
    for case in CASES:
        row = metrics.loc[(metrics["case"] == case) & (metrics["model_kind"] == "pino")].iloc[0]
        rows.append(
            f"IEEE {BUS_COUNTS[case]} & "
            f"{row['balance_mean_pu']:.3f} & "
            f"{_fmt(row['balance_max_mean_pu'], row['balance_max_mean_pu_seed_std'], 3)} & "
            f"{row['balance_max_p95_pu']:.3f} & "
            f"{row['balance_max_worst_pu']:.3f} & "
            f"{row['p_balance_max_mean_pu']:.3f} & "
            f"{row['q_balance_max_mean_pu']:.3f}" + r" \\"
        )
    (output_dir / "table_balance.tex").write_text(
        "\n".join(
            [
                r"\begin{table*}[t]",
                r"\centering",
                (
                    r"\caption{GFNO-PINO complex-power-balance diagnostics in p.u. "
                    r"on unseen topologies. Mean max. is mean $\pm$ seed standard "
                    r"deviation; P95 and worst are five-seed means of within-seed "
                    r"held-out statistics.}"
                ),
                r"\label{tab:balance}",
                r"\scriptsize",
                r"\begin{tabular}{lrrrrrr}",
                r"\toprule",
                (
                    r"System & Mean bus & Mean max. & P95 max. & Worst max. & "
                    r"Mean max. $|r^P|$ & Mean max. $|r^Q|$ \\"
                ),
                r"\midrule",
                *rows,
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table*}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    rows = []
    for _, row in paired_effect.iterrows():
        data_only_value = _fmt(
            row["data_only_balance_max_mean_pu"],
            row["data_only_balance_max_seed_std_pu"],
            3,
        )
        pino_value = _fmt(
            row["pino_balance_max_mean_pu"],
            row["pino_balance_max_seed_std_pu"],
            3,
        )
        paired_value = _fmt(
            row["paired_reduction_mean_percent"],
            row["paired_reduction_seed_std_percent"],
            1,
        )
        rows.append(
            f"IEEE {int(row['n_bus'])} & "
            f"{data_only_value} & "
            f"{pino_value} & "
            f"{paired_value} & "
            f"{row['paired_seed_reductions_percent']} & "
            f"[{row['paired_reduction_resampling_low_percent']:.1f}, "
            f"{row['paired_reduction_resampling_high_percent']:.1f}]" + r" \\"
        )
    (output_dir / "table_physics_effect.tex").write_text(
        "\n".join(
            [
                r"\begin{table*}[t]",
                r"\centering",
                (
                    r"\caption{Observed effect of physics fine-tuning on mean "
                    r"sample-maximum complex-power-balance mismatch (p.u.). "
                    r"Model columns are mean $\pm$ seed standard deviation. "
                    r"Paired reduction compares identical seeds. All five seed "
                    r"effects are shown; the final column is a deterministic "
                    r"10,000-resample percentile interval used descriptively, "
                    r"not as population-level confidence.}"
                ),
                r"\label{tab:physics_effect}",
                r"\scriptsize",
                r"\resizebox{\textwidth}{!}{%",
                r"\begin{tabular}{lrrrrr}",
                r"\toprule",
                (
                    r"System & Data-only GFNO & GFNO-PINO & "
                    r"Paired mean (\%) & Seed effects 2026--2030 (\%) & "
                    r"Seed-resampling interval (\%) \\"
                ),
                r"\midrule",
                *rows,
                r"\bottomrule",
                r"\end{tabular}",
                r"}",
                r"\end{table*}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    rows = []
    for case in CASES:
        c = classical.loc[classical["case"] == case].set_index("method")
        n = metrics.loc[metrics["case"] == case].set_index("model_kind")
        rows.append(
            f"IEEE {BUS_COUNTS[case]} & "
            + f"{c.at['IPOPT AC-OPF', 'latency_median_ms']:.2f} & "
            + f"{c.at['DC-OPF', 'latency_median_ms']:.2f} & "
            + _fmt(
                n.at["topology_blind_gfno", "latency_median_ms"],
                n.at["topology_blind_gfno", "latency_median_ms_seed_std"],
            )
            + " & "
            + _fmt(
                n.at["gnn", "latency_median_ms"],
                n.at["gnn", "latency_median_ms_seed_std"],
            )
            + " & "
            + _fmt(
                n.at["pino", "latency_median_ms"],
                n.at["pino", "latency_median_ms_seed_std"],
            )
            + r" \\"
        )
    (output_dir / "table_latency.tex").write_text(
        "\n".join(
            [
                r"\begin{table*}[t]",
                r"\centering",
                (
                    r"\caption{Device-specific batch-one latency in milliseconds. "
                    r"Neural values are five-seed mean $\pm$ standard deviation "
                    r"of medians after warm-up; solver "
                    r"values are stored CPU medians. Preprocessing, transfer, "
                    r"verification, and restoration are excluded.}"
                ),
                r"\label{tab:latency}",
                r"\begin{tabular}{lrrrrr}",
                r"\toprule",
                r"System & IPOPT AC-OPF & DC-OPF & Topology-blind GFNO & Plain GNN & GFNO-PINO \\",
                r"\midrule",
                *rows,
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table*}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    averaged = metrics.groupby("model_kind").mean(numeric_only=True)
    attributes = {
        "fixed_pinn": ("Base only", "No", "Yes"),
        "topology_blind_gfno": ("No", "Yes", "Yes"),
        "gnn": ("Yes", "No", "Yes"),
        "data_only_gfno": ("Yes", "Yes", "No"),
        "pino": ("Yes", "Yes", "Yes"),
    }
    rows = []
    for model in MODELS:
        topology, spectral, physics = attributes[model]
        row = averaged.loc[model]
        rows.append(
            f"{MODEL_LABELS[model]} & {topology} & {spectral} & {physics} & "
            f"{row['absolute_reference_difference_mean_percent']:.1f} & "
            f"{row['balance_mean_pu']:.3f} & "
            f"{row['balance_max_mean_pu']:.3f} & "
            f"{row['thermal_violation_rate_percent']:.2f} & "
            f"{row['thermal_violation_max_pu']:.2e} & "
            f"{row['ac_feasible_rate_percent']:.1f}" + r" \\"
        )
    (output_dir / "table_ablation.tex").write_text(
        "\n".join(
            [
                r"\begin{table*}[t]",
                r"\centering",
                (
                    r"\caption{Four-system, five-seed ablation averages. "
                    r"Mismatch and thermal-excess magnitudes are p.u. "
                    r"Voltage, generator, and angle violation rates and excesses "
                    r"are zero for every method.}"
                ),
                r"\label{tab:generalization}",
                r"\resizebox{\textwidth}{!}{%",
                r"\begin{tabular}{lcccrrrrrr}",
                r"\toprule",
                (
                    r"Model & Topol. input & Spectral & Physics & "
                    r"$|\Delta C_{\rm ref}|$ (\%) & Mean bus & Mean max. & "
                    r"Therm. rate (\%) & Therm. max. & AC feas. (\%) \\"
                ),
                r"\midrule",
                *rows,
                r"\bottomrule",
                r"\end{tabular}",
                r"}",
                r"\end{table*}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    rows = []
    for model in MODELS:
        row = averaged.loc[model]
        thermal = (
            f"{row['thermal_violation_rate_percent']:.2f}/{row['thermal_violation_max_pu']:.2e}"
        )
        rows.append(
            f"{MODEL_LABELS[model]} & {row['balance_mean_pu']:.3f} & "
            f"{row['balance_max_mean_pu']:.3f} & {thermal} & "
            f"{row['ac_feasible_rate_percent']:.1f}" + r" \\"
        )
    (output_dir / "table_all_model_constraints.tex").write_text(
        "\n".join(
            [
                r"\begin{table}[H]",
                r"\centering",
                (
                    r"\caption{Four-system, five-seed physical diagnostics for "
                    r"all learned methods. Thermal cells show violation rate "
                    r"(\%)/mean seed-maximum excess (p.u.). Voltage, generator, "
                    r"and angle violation rates and excesses are zero for every "
                    r"method; none passes the declared AC-feasibility proxy.}"
                ),
                r"\label{tab:all_model_constraints}",
                r"\resizebox{\columnwidth}{!}{%",
                r"\begin{tabular}{lrrrr}",
                r"\toprule",
                (
                    r"Model & Mean bus bal. & Mean max. bal. & Thermal & "
                    r"AC feasible (\%) \\"
                ),
                r"\midrule",
                *rows,
                r"\bottomrule",
                r"\end{tabular}",
                r"}",
                r"\end{table}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def aggregate(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    args.generated_tex_dir.mkdir(parents=True, exist_ok=True)
    frames: dict[tuple[str, str, int], pd.DataFrame] = {}
    records = []
    for case in CASES:
        for model in MODELS:
            for path in sorted((args.results_root / case / model).glob("seed_*")):
                seed = int(path.name.removeprefix("seed_"))
                frame = pd.read_csv(path / "per_sample_metrics.csv")
                frames[(case, model, seed)] = frame
                run_metadata = json.loads(
                    (args.runs_root / case / model / path.name / "run_metadata.json").read_text(
                        encoding="utf-8"
                    )
                )
                records.append(_seed_record(frame, run_metadata, case, model, seed))
    seed_metrics = pd.DataFrame(records)
    expected = len(CASES) * len(MODELS) * 5
    if len(seed_metrics) != expected:
        raise RuntimeError(
            f"expected {expected} completed case/model/seed runs, found {len(seed_metrics)}"
        )
    metrics = _aggregate_seed_records(seed_metrics)
    paired_effect = _paired_physics_effect(seed_metrics)
    datasets = _dataset_summary(args.data_root)
    classical = _classical_metrics(frames)
    seed_metrics.to_csv(args.output_dir / "metrics_by_case_model_seed.csv", index=False)
    metrics.to_csv(args.output_dir / "metrics_by_case_model.csv", index=False)
    paired_effect.to_csv(
        args.output_dir / "physics_finetuning_paired_effect.csv",
        index=False,
    )
    datasets.to_csv(args.output_dir / "dataset_summary.csv", index=False)
    classical.to_csv(args.output_dir / "classical_baselines.csv", index=False)
    _plot_training(args.runs_root, args.figure_dir)
    _plot_scalability(metrics, classical, args.figure_dir)
    _plot_architecture(args.figure_dir)
    _write_tables(metrics, classical, paired_effect, args.generated_tex_dir)
    summary = {
        "benchmark_tier": "review-revision-multiseed",
        "cases": list(CASES),
        "models": list(MODELS),
        "training_seeds": sorted(seed_metrics["seed"].unique().tolist()),
        "n_solver_labels": int(datasets["converged_samples"].sum()),
        "n_neural_evaluations": int(seed_metrics["n_samples"].sum()),
        "all_models_ac_feasible_rate_percent": float(metrics["ac_feasible_rate_percent"].mean()),
        "power_balance_definition": "sqrt(rP^2 + rQ^2) per bus",
    }
    (args.output_dir / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results/revision"))
    parser.add_argument("--runs-root", type=Path, default=Path("runs/revision"))
    parser.add_argument("--data-root", type=Path, default=Path("data/pilot"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/revision/aggregate"))
    parser.add_argument("--figure-dir", type=Path, default=Path("manuscript/figures"))
    parser.add_argument("--generated-tex-dir", type=Path, default=Path("manuscript/generated"))
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    aggregate(args)
    print(json.dumps({"output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()

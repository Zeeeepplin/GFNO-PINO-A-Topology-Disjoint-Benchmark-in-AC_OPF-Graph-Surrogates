# Manuscript

This directory contains the two-column Elsevier manuscript:

> Physics-Informed Fine-Tuning for AC Optimal Power Flow Surrogates Under Line
> Outages: A Topology-Disjoint Benchmark and Ablation Study

`main.tex` reports the complete five-seed revision benchmark. All numerical
tables and figures are regenerated from archived case/model/seed artifacts;
there are no hand-entered model results.

## Reproduce and compile

From the repository root:

```powershell
python reproduce_submission.py --device cuda
```

To regenerate only aggregate CSVs, LaTeX fragments, and figures:

```powershell
python aggregate_revision_results.py
```

The historical `make_review_figures.py` filename is retained only as a
compatibility entry point; it now calls the same complete revision aggregator
and cannot produce the obsolete single-seed tables.

Compile directly with:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The released PDF is rendered to page images and inspected for clipping,
overflow, malformed equations, unreadable figures, and incorrect two-column
placement before delivery.

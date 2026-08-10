"""Tests for seed-paired benchmark aggregation."""

from __future__ import annotations

import pandas as pd

from aggregate_revision_results import CASES, _paired_physics_effect


def test_paired_physics_effect_aligns_identical_seeds() -> None:
    records = []
    for case in CASES:
        for seed in (5, 1, 4, 2, 3):
            records.extend(
                [
                    {
                        "case": case,
                        "model_kind": "data_only_gfno",
                        "seed": seed,
                        "balance_max_mean_pu": 10.0,
                    },
                    {
                        "case": case,
                        "model_kind": "pino",
                        "seed": seed,
                        "balance_max_mean_pu": 5.0,
                    },
                ]
            )
    effect = _paired_physics_effect(pd.DataFrame(records))
    assert effect["n_paired_seeds"].eq(5).all()
    assert effect["paired_reduction_mean_percent"].eq(50.0).all()
    assert effect["paired_reduction_seed_std_percent"].eq(0.0).all()
    assert effect["paired_reduction_resampling_low_percent"].eq(50.0).all()
    assert effect["paired_reduction_resampling_high_percent"].eq(50.0).all()
    assert effect["paired_seed_reductions_percent"].eq(
        "50.0,50.0,50.0,50.0,50.0"
    ).all()

# ruff: noqa: T201  a spike reports by printing; it is not shipped code
"""Where `solve_over` earns its place inside a decomposition, and where it does not.

`benders.py` needs nothing from the driver: it runs on `main`, which has no
`strategy.py`. This asks the other question — **would it profit later?** — and
the answer is a specific yes with a specific shape.

Multi-cut Benders solves one subproblem *per scenario* each iteration and emits
one cut per scenario. That inner fan-out is a fold over a partition, which is
exactly what `solve_over` is: independent slices, no state between them, and a
pool if you want one.

The nesting is the point. `solve_over` refuses `carry` together with `executor`
(#584 wall 4), and it is right to — but a decomposition never asks it to. The
carry lives in the *outer* loop, which is sequential because cut i+1 needs solve
i; the parallelism lives in the *inner* fold, which has no carry at all. They
never meet, so the refusal that looked like a wall is not one when `solve_over`
is called from inside a step rather than asked to be the step.
"""

from __future__ import annotations

import polars as pl

from lpspec.strategy import EachCoordinate, solve_over
from spike.data import GENERATORS, SNAPSHOTS

SCENARIOS = ['dry', 'windy']
WIND = {'dry': [0.10, 0.15, 0.05, 0.20], 'windy': [0.90, 1.00, 0.60, 0.80]}

SOURCES = {
    'cost': pl.DataFrame({'generator': GENERATORS, 'value': [0.0, 25.0]}),
    'load': pl.DataFrame(
        {
            'scenario': [s for s in SCENARIOS for _ in SNAPSHOTS],
            'snapshot': SNAPSHOTS * len(SCENARIOS),
            'value': [40.0, 80.0, 55.0, 95.0] * len(SCENARIOS),
        }
    ),
    'avail': pl.DataFrame(
        {
            'scenario': [s for s in SCENARIOS for _ in SNAPSHOTS for _ in GENERATORS],
            'snapshot': [t for _ in SCENARIOS for t in SNAPSHOTS for _ in GENERATORS],
            'generator': GENERATORS * len(SNAPSHOTS) * len(SCENARIOS),
            'value': [v for s in SCENARIOS for t, w in enumerate(WIND[s]) for v in (w, 1.0)],
        }
    ),
}


def cuts_for(capacity: pl.DataFrame) -> pl.DataFrame:
    """One cut per scenario, from one fold.

    `EachCoordinate('scenario')` cuts the sources; each slice is an independent
    dispatch at the same capacity. `Runs.dual` then hands every scenario's
    capacity prices back **in one frame, keyed by scenario** — which is exactly
    the shape a multi-cut needs, because the group that makes a cut is the key.

    Before #586 this was impossible: a sweep returned primals only, and the one
    value a cut is built from could not leave the fold.

    **The subproblem file is unchanged.** `EachCoordinate` filters the sources
    carrying `scenario` and drops the column, "so the model never mentions it" —
    which means the same `sub.yaml` serves the single-scenario loop and this one.
    Adopting the driver for multi-cut costs no YAML at all.
    """
    runs = solve_over(
        'examples/benders/sub.yaml',
        {**SOURCES, 'cap_hat': capacity},
        EachCoordinate('scenario'),
    )
    prices = runs.dual('capacity')
    avail = SOURCES['avail'].rename({'value': 'avail'})
    return (
        prices.join(avail, on=['scenario', 'snapshot', 'generator'])
        .with_columns((pl.col('value') * pl.col('avail')).alias('term'))
        .group_by('scenario', 'generator')
        .agg(pl.col('term').sum().alias('slope'))
        .sort('scenario', 'generator')
    )


if __name__ == '__main__':
    capacity = pl.DataFrame({'generator': GENERATORS, 'value': [30.0, 100.0]})
    slopes = cuts_for(capacity)
    print('one fold, one cut per scenario:\n')
    print(slopes)
    print(f'\ncuts this iteration would emit: {slopes["scenario"].n_unique()}')
    print('the outer loop stays sequential; this fold is the part that can take an executor.')

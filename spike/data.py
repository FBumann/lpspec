"""The sources the probes share, matching `examples/benders/` on main.

The models themselves are not duplicated here: this branch is only about what
`solve_over` can and cannot carry, and the worked decomposition it probes lives
on main as an example with its own committed output.
"""

from __future__ import annotations

import polars as pl

SNAPSHOTS = [0, 1, 2, 3]
GENERATORS = ['wind', 'gas']

SOURCES = {
    'invest': pl.DataFrame({'generator': GENERATORS, 'value': [90.0, 30.0]}),
    'cost': pl.DataFrame({'generator': GENERATORS, 'value': [0.0, 25.0]}),
    'load': pl.DataFrame({'snapshot': SNAPSHOTS, 'value': [40.0, 80.0, 55.0, 95.0]}),
    'avail': pl.DataFrame(
        {
            'snapshot': [s for s in SNAPSHOTS for _ in GENERATORS],
            'generator': GENERATORS * len(SNAPSHOTS),
            'value': [0.9, 1.0, 0.2, 1.0, 0.6, 1.0, 0.1, 1.0],
        }
    ),
}
DISPATCH = {name: frame for name, frame in SOURCES.items() if name != 'invest'}

#: Enough capacity that the strict subproblem is feasible, so a probe about the
#: seam is not accidentally a probe about infeasibility.
FEASIBLE = pl.DataFrame({'generator': GENERATORS, 'value': [30.0, 100.0]})

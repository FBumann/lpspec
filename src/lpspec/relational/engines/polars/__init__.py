"""The polars engine: plan → frames → `sinks.ModelTables`.

Selected with `LPSPEC_ENGINE=polars`; `duck/` is what an unset environment
gets. Ahead of it on every rung of the committed ladder (see that package's
docstring) — which is why the ladder is published against this one and why the
choice is a switch a machine can throw rather than a routing decision.

It needs neither pyarrow nor pandas, and `tests/test_api.py` pins that: on this
engine a bridge out (`to_pandas`, `to_dataarray`) stays a bridge and never
becomes something the build path walks over on its own.

Everything here is engine-private. The contract is one level up —
`relational/plan.py` going in, `relational/sinks/tables.py` coming out — and
nothing outside this package may reach past those two.

Split out when the question of a second engine was priced
(`bench/duckdb-spike.md`): with one engine the boundary between *what a model
is* and *how it is built* was real but invisible, and a reader had to know
which of the eleven modules under `relational/` were which.
"""

from lpspec.relational.engines.polars.executor import PolarsExecutor

__all__ = ['PolarsExecutor']

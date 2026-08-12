"""The duckdb engine: plan → duckdb relations → `sinks.ModelTables`.

**The default engine**, and what `LPSPEC_ENGINE` unset selects. Not routed to:
there is no routing here, only a choice, and the two engines answer the same
YAML with the same numbers (`tests/test_engine_parity.py`). `LPSPEC_ENGINE=polars`
is the other one.

**On the committed ladder it does not currently win anything**, and the claim
that used to stand here — 2.66 s against the polars engine's 6.84 s at the top
rung, 0.55x on `dispatch/l` — has been overtaken rather than disproved. Measured
against the polars engine on all seven cases, both sinks, `l` rung: the build
phase is **1.6-3.1x slower**, never faster, and peak is **0.90-1.84x** — lighter
on `dispatch` and `nodal` through the solver, heavier on eleven of fourteen.

**Default anyway, and deliberately so.** Being behind on this ladder is what
makes it the engine worth having under the instruments: it is the default so
that CodSpeed, `bench.yml` and the unflagged CI pass all measure it without
being asked to, which is the only way the gap above closes or is shown not to.
The ladder is what the choice is answerable to — see `docs/ROADMAP.md` Track 4
for what it still cannot say.

Two things moved it, and one of them is ours rather than the engine's. A perf
series landed on the polars engine while this one was on a branch. And `cols`
going positional (#433) is *free* there — the order falls out of a cross join's
emission order — where here it costs an `ORDER BY`, which roughly doubled
`dispatch/l`'s build when it landed. A shared contract that is cheap for one
engine and not the other widens the gap without either engine regressing.

One gap is this engine's own and is the obvious thing to take next:
`_needs_aggregate` here answers yes whenever a constraint has more than one
term, where the polars engine asks the dimension table whether two fragments
can actually collide (`may_share_a_column`). Correct, and it sorts every
nonzero in the model to collapse nothing on the ordinary multi-term
constraint.

What the ladder cannot say is what happens above it: every rung fits in RAM, so
the argument this engine was built on — a model that does not — is untested
rather than refuted. Nor does a `memory_limit` reach it: at 40M columns the
build peaks at 4.49 GB capped at 500 MB and at 4.49 GB uncapped, the engine's
own working set having been under the cap all along. Only 1.65 GB of that is
the four frames, on the far side of `.pl()` — the rest is build transient, and
the polars engine's is the same shape.

`bench/duckdb-spike.md` carries the whole measurement. Read §7 as the provenance
of the *decision* rather than as the current cost — it records the out-of-tree
engine this one was ported from, whose 2.1-4.2x memory advantage the port has
never reproduced.

It **needs pyarrow**, which the polars engine does not: duckdb and polars hand
frames to each other through Arrow. It does *not* need pandas — pyarrow imports
pandas only when pandas is already installed, which is easy to mistake for a
requirement in a development environment. Both are runtime dependencies, since
this is the engine a bare install gets; `tests/test_api.py` pins the pandas
half and the narrower polars-engine claim.
"""

from lpspec.relational.engines.duck.compiler import DuckCompiler
from lpspec.relational.engines.duck.executor import DuckExecutor

__all__ = ['DuckCompiler', 'DuckExecutor']

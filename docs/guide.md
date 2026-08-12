# Writing a model

Five ideas carry the whole language. Each one is shown below in a model that
lives in the repo and is run by the test suite, so nothing here is a snippet
that only works on this page.

If you would rather see the machinery than read about it,
`python examples/walkthrough.py` prints every stage — YAML → schema → AST →
plan → frames → LP text → solution — for one small model.

## 1. A dimension is declared; its coordinates usually are not

```yaml
dimensions:
  snapshot: {dtype: int}  # coordinates come from the data
  generator: {values: [wind, solar, gas]}  # coordinates are given here
```

A dimension is an axis. You either list its coordinates in the file, or leave
them to be read off whatever data binds — `coords={"snapshot": range(6)}` at
call time, or the union of what the parameters carry.

**One master coordinate set per dimension, resolved before any data binds.**
Every parameter is reindexed onto it, so two tables that disagree about which
snapshots exist is an error you get at load time rather than a silently
truncated model.

## 2. Absence is how you say "sparse"

```yaml
variables:
  p:
    foreach: [snapshot, generator]
    where: "p_max > 0"
```

`where` does not zero a variable out — it means the variable **has no column
there at all**. A retired generator with `p_max = 0` costs nothing to carry in
the data, and the built model is smaller than the coordinate product.

The same idea runs through data binding, with one distinction worth learning
early: a **variable** the mask removed is *absent*, and a term carrying it takes
its whole row with it — while a **parameter** row that is simply missing is a
zero coefficient, and the row survives without it. Absence is a property of
variables. → [dispatch](models/dispatch.md), [SPEC §6](SPEC.md#6-absence)

## 3. A dimension can carry coordinates, and that is your topology

```yaml
dimensions:
  generator: {dtype: str, coords: [bus]}  # each generator sits on a bus
  line: {dtype: str, coords: {from: bus, to: bus}}  # both endpoints are buses
```

```yaml
- expression: >-
    sum(p, over=generator, group_by=bus)
    + sum(f, over=line, group_by=to)
    - sum(f, over=line, group_by=from)
    == load
```

`sum(group_by=)` sums along a coordinate, landing the result on the dimension that
coordinate points at. The same `f` is summed twice through two different
coordinates — once as inflow, once as outflow.

No adjacency matrix and no join written by hand: the network is data on the
dimension. → [transport](models/transport.md)

## 4. `shift` reaches along an axis

```yaml
- expression: soc == shift(soc, over=snapshot, by=1, edge='wrap') + charge * 0.9 - discharge
```

One operator, and `edge=` says what happens at the boundary. `edge='wrap'` is
cyclic — the first snapshot reads the last, which is what makes a battery
cyclic without writing the boundary condition out. Omit `edge` and positions
translated past the edge are **absent**, so the row they would have fed is not
built. `edge=0` keeps the row and contributes zero there instead.

This is the only construct whose cost is not obviously linear in model size.
→ [storage](models/storage.md)

## 5. The dims of an equation must equal its `foreach`

```yaml
constraints:
  power_balance:
    foreach: [snapshot]
    expression: sum(p, over=generator) == load
```

`p` has dims `(snapshot, generator)`; summing over `generator` leaves
`(snapshot)`; `load` has `(snapshot)`. The union is `(snapshot)`, which is what
`foreach` says — so it compiles.

Get it wrong and you are told at load time, not at solve time. A stray dim
would multiply rows and an unused `foreach` dim would repeat one row across
them; either way you would build a different model than the file reads as.
→ [SPEC §5.2](SPEC.md#52-dim-algebra)

## Then: check, build, solve

```python
import lpspec as lps

lps.check('model.yaml')  # compiles? no data needed
sol = lps.solve('model.yaml', sources)  # to an answer
sol.objective
sol.primal('p')  # a polars.DataFrame
sol.dual('power_balance')
```

Two engines build the same YAML, both installed, neither behind an extra. The
default is `duckdb`; `LPSPEC_ENGINE=polars` is the other, and is ahead on every
rung of the published ladder. Same model either way, integer for integer, which
is why it is an environment variable and not an argument — nothing you can say
in YAML picks one, and nothing either one picks changes the answer.
→ [benchmarks](benchmarks.md)

`lps.check` is the CI verb — it parses, expands, resolves and lowers without
binding anything, so a model repository can be validated on every commit
without shipping the data.

Sources accept polars, pandas, pyarrow, or parquet paths — anything exposing
the Arrow PyCapsule protocol, and the recogniser imports none of them.
Results come back as frames; `to_pandas`, `to_dataarray` and `to_parquet` are
the bridges out. → [SPEC §8](SPEC.md#8-data-binding), [Python API](api.md)

## What it will not do

Worth knowing before you start, rather than after:

- **Bounds take a name or a number, never arithmetic.** `upper: p_max` is
  fine; `upper: -rating` is not. This one has bitten a real port —
  [#31](https://github.com/fluxopt/lpspec/issues/31), and the workaround is to
  ship the negated column as data.
- **Every expression is affine in the variables.** Degree 1, always: no
  variable times variable. That is the ceiling the whole design is built
  around, not an unimplemented feature. →
  [The ceiling](design/ceiling.md#two-tiers-and-the-ceiling)
- **Several plausible features are refused on purpose**, with reasons.
  → [ROADMAP](ROADMAP.md)

## Where next

| | |
|---|---|
| [Models](models/index.md) | every model in the repo, and which constructs each exercises |
| [SPEC](SPEC.md) | the reference — what a file may contain, exactly |
| [Benchmarks](benchmarks.md) | what it costs, measured against linopy |
| [ARCHITECTURE](ARCHITECTURE.md) | why it is shaped this way |

# Roadmap

Why this project exists, and what it is becoming. **No work items live here.**
The work is issues, grouped under three parents:

- [Track 1 — primitives](https://github.com/fluxopt/lpspec/issues/470)
- [Track 2 — the operational surface](https://github.com/fluxopt/lpspec/issues/471)
- [Track 3 — capabilities, and the degree line](https://github.com/fluxopt/lpspec/issues/472)

An index maintained by hand beside an issue tracker is a second copy that
drifts, so there is not one here: the issues are the list, and this page is the
argument for what the list is *for*.

## Why

An optimisation model is math, and math is worth reading. It usually arrives as
Python that *builds* math — the equations entangled with the loops, the frames
and the library that assembled them, so a diff shows scaffolding rather than
constraints and nothing can read the model except the program that wrote it.
Reviewing such a model means reviewing a program, which is a different and much
harder job.

lpspec makes the math the artifact. A YAML file says what the variables,
constraints and objective *are*; the file is validated at load time, built at
runtime, and can be reviewed by someone who understands the math without
understanding the builder. That is the whole thesis, and every rule below is
downstream of it.

## Where it is going

**One language, more than one place to run it.** The same file builds natively
on the relational engine or onto a `linopy.Model` that already exists in memory.
Not a fallback and not a dialect — one language, so a differential test between
the two lanes is an oracle rather than a comparison.

**A build that streams, with a ceiling you can declare.** The model is frames
and the build is relational, so nothing dense is ever materialised and peak
tracks the model rather than a number someone guessed. What is missing is the
*declaration*: there is no way to say "build this within N gigabytes or fail".
The honest version is partition-wise execution, which the locality closure
already guarantees is safe.

**Answers, not just solutions.** A solved model should tell you why it is
infeasible, what a row costs, and what changed since the last solve — without
opening a file no editor can hold. Most of that is a query over frames that
already exist.

**Component libraries, composed rather than generated.** A fixed set of
parametrised templates agreeing on a port/flow convention, merged into one
program before a single build pass. **Topology is data** — wiring a system is
rows in a connectivity table, never generated YAML — so structure stays bounded
by the number of component *types* while cardinality lives entirely in data.

## What it will not become

**Two durable losses, and they are the price of the closed AST.** Structure that
needs the solver's *answer* to decide the next row, inside one plan; and
imperative modeling at all. What that price buys is load-time validation, two
lanes on one language, and a build that streams. Everything else is scheduling.

Everything a model needs after "it builds". Mostly queries over frames we
already materialise.

- **Reading results.** Duals have shipped; reduced costs and slacks ride the
  same join and have not. Derived results (LCOE, curtailment, emissions by
  group) are SQL over the solution tables.
- **Infeasibility.** HiGHS has no IIS — Gurobi does, but an answer only the
  opt-in sink can give is not the answer — so it is **elastic relaxation**
  ([#80](https://github.com/fluxopt/lpspec/issues/80)) — slacks with penalty
  costs, then a query of nonzero slacks grouped by block. Needs no solver
  feature and works on every sink. Taxed like a primitive, since new variables
  mean a schema-level expansion pass.
- **The REPL gap is verbs, not structure.** A built model is four polars frames,
  which answers "what is in row 12" with a filter where a labelled `Dataset`
  needs a scatter — `build()` already returns the live executor, it just has
  nothing readable on it. Render a **bound** row, preview which rows survive a
  `where`, evaluate an expression against bound data, coefficient ranges, model
  statistics. All **read-only**: inspecting a built model, never declaring one,
  which is the line that keeps this cheap and rule 5 intact.
- **Lifecycle.** `var_label` *is* the solver column index with no remapping, so
  value-only re-solve is a label query plus `changeColsBounds`, integrality is
  the same contract (`changeColsIntegrality`, already called at build), and
  **appending rows moves no label at all**. The session that holds the handle is
  [#204](https://github.com/fluxopt/lpspec/issues/204); warm starts are
  [#382](https://github.com/fluxopt/lpspec/issues/382), and they are what make
  rebuild-instead-of-edit cheap rather than merely correct.
- **Decomposition.** Benders and successive substitution, and **the shape
  favours us** — worth saying because the opposite is the natural assumption.
  Decomposition wants sparse triplets plus label tables; that *is* the model
  here, so the master/sub split is a `GROUP BY` over `A`
  ([#39](https://github.com/fluxopt/lpspec/issues/39)), cut coefficients are
  `duals ⋈ subproblem rows`, and a cut lands with zero translation. The open
  question is not feasibility but **who writes the cut**
  ([#381](https://github.com/fluxopt/lpspec/issues/381)).
- **AST consumers.** The first has shipped — `to_latex` / `to_typst` /
  `to_markdown`, one tree walk, no data, no solver. Remaining: CLI
  ([#35](https://github.com/fluxopt/lpspec/issues/35)), observability
  ([#34](https://github.com/fluxopt/lpspec/issues/34)).

## Track 3 — capabilities, and the degree line

**The ceiling and sink capability are two axes.** A declared capability set per
**sink** (not per solver — `lp_file` is not a solver but has capabilities),
modelled on linopy's `Solver.features` with two divergences: entries are
three-valued (`native` / `reformulated` / `absent`) so satisfying one by
reformulation is additive later, and the model expresses **conjunction
exclusions**, because linopy declares HiGHS with `INTEGER_VARIABLES` *and*
`QUADRATIC_OBJECTIVE` in one flat `frozenset` while HiGHS refuses the pair.
`check(model, sink=...)` takes the sink optionally. Design:
[#89](https://github.com/fluxopt/lpspec/issues/89).

That unblocks three things, in order of effort:

| | Blocked on | Note |
|---|---|---|
| **Semi-continuous** | nothing | HiGHS has `kSemiContinuous` natively and linopy has the oracle — [#383](https://github.com/fluxopt/lpspec/issues/383) |
| **SOS / indicator** | the capability model | `lp_file` carries SOS as a text section, Gurobi natively, HiGHS not at all — [#23](https://github.com/fluxopt/lpspec/issues/23) |
| **Quadratic** | the capability model — the second solver ships | below |

**Quadratic is planned, not refused.** The cost side is settled and small, and
the oracle is free — linopy's `QuadraticExpression` builds the comparison, which
is normally the expensive half of a primitive. Performance is not the question.

The blocker is *where it can land*: HiGHS returns `kError` for
`Hessian + integrality`, and `binary:`, `integer:` and nonconvex `piecewise:` all
ship today, so on the default path quadratic conflicts with features already in
the language. That is a capability finding, not a reason to refuse the math — so
it needs the table above **and** a solver without the exclusion. The second
solver is no longer the blocking half: the `gurobi` solver ships
([#106](https://github.com/fluxopt/lpspec/issues/106)), and Gurobi takes a
Hessian alongside integrality. What is left is the capability table, without
which landing the primitive would ship math the *default* solver refuses.

Until then `piecewise: {convex: true}` and the epigraph pattern are the answer
for convex 1-D curves — and they keep the LP duals, warm starts and MILP
compatibility a quadratic objective gives up, so they stay the *preferred*
spelling even after quadratic lands. The scope, the lowering, and whether
*coordinate-aligned* is the right restriction at all, are
[#261](https://github.com/fluxopt/lpspec/issues/261) and
[#84](https://github.com/fluxopt/lpspec/issues/84).

## Track 4 — the memory axis

Both engines hold the model they build, so peak tracks the model rather than a
number the caller sets. That is the right default and it is what makes the
lifetime disappear from the API, but there is **no declared ceiling** — no way
to say "build this within N gigabytes or fail".

**Choosing an engine is not the answer**, and that is worth knowing before this
track is picked up. The figure this section used to quote — 2.1–4.2x less
memory on the write path — was measured on the engine the in-tree duckdb one
was ported from, and the port has never reproduced it: peak lands between 0.90x
and 1.84x of the polars engine's, lighter on two rungs of fourteen and heavier
on eleven
([bench/duckdb-spike.md](https://github.com/fluxopt/lpspec/blob/main/bench/duckdb-spike.md)).
It was briefly ahead on *speed* at the top of the ladder; the polars perf series
has since taken that back too. duckdb is the default regardless — being behind
on the ladder is what makes it the engine worth having under CI's instruments
unasked — but the default is not what closes this gap and was never meant to be.

**But it locates the ceiling, which is worth more than a ratio.** duckdb *does*
have the knob this section asks for — `SET memory_limit`, and it spills past it.
Setting it shows the knob is aimed at the wrong half: `dispatch/xl`, 40M
columns, engine capped at **500 MB**, builds unchanged and peaks at **4.49 GB —
the same figure, to the last measured digit, as the uncapped build**. Its own
working set was already under the cap, so there is nothing for the knob to bind
on, and no engine-side limit reaches peak.

**What a bound would have to be declared over has moved, though.** The frames
are **1.65 GB of that 4.49** — CSR halved the matrix on both engines — so peak
is no longer mostly the output the way this section used to say it was. The
build transient is now the larger term, on polars (4.02 GB against the same
1.65) as much as on duckdb. A ceiling declared over `ModelTables` would miss
most of what a build costs.

What remains is a declared bound, and the honest version is partition-wise
execution, which the locality closure already guarantees is safe. Measured on
polars it takes 23–36% off peak and then floors, because the matrix stops being
the binding term while the label frames, `cols`, `rows`, `obj` and the
parameters stay resident. That floor and the transient above are the same
finding from two directions, and between them they are what this track has to
answer. Worth most for the write path either way: the solver is the larger term
by roughly an order of magnitude at 10⁷ variables ([benchmarks](benchmarks.md)).

## What we will not build

The specific refusals — data prep, arbitrary array ops, domain helpers,
normalisation, in-plan conditionals, a Python modeling API — are in
[the ceiling](design/ceiling.md#deliberate-non-primitives), with the reason and
the rewrite for each. Read them before proposing a feature: parity with another
tool is not by itself a reason to add anything.

## Honest snapshot

**Cheaper here, because the model is tables:** model statistics and
coefficient-range diagnostics; IIS read-back (a join, not a scatter);
serialization to parquet; elastic relaxation; dualization, since transposing a
COO matrix is swapping two column names.

**Ahead of comparable declarative layers:** sparse-by-construction build with no
dense intermediate, and a hand-off straight to the solver rather than through a
file; parameterised `macros:` ([Calliope](design/prior-art.md)'s sub-expressions
take no arguments);
binary and integer variables; piecewise as N links with per-link signs, convex
mode and `active` gating; load-time validation of every expression, `where`
string and *uncalled* macro template.

**Behind linopy**, and none of it a ceiling question: the post-solve object
(labelled DataArrays vs tidy tables — `to_dataarray` bridges), debugging (IIS
via Gurobi, `print()` of a row), lifecycle (mutate, re-solve, warm start,
`relax`/`fix`), solver breadth (ten backends and four handoffs vs HiGHS-direct
plus LP files), and the variable types and capabilities behind the capability
model.

**The ranking this implies:** indexed access blocks whole model classes today;
the operational verbs block using the engine at 3am; solver breadth blocks
arrival from linopy at all; semi-continuous and `cumsum`-over-data are cheap,
unblocked and unscheduled.

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
arrival from linopy at all; semi-continuous ([#383](https://github.com/fluxopt/lpspec/issues/383))
is cheap, unblocked and unscheduled.

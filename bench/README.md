# The performance harness

Not shipped in the wheel, not imported by `lpspec`, not run in CI. It exists so
that [docs/about/benchmarks.md](../docs/about/benchmarks.md) has a *provenance* — the last
set of published numbers came from a `scratch/` script that was deleted, and a
claim nobody can re-run is a claim with a shelf life.

It is **one pytest suite** (`bench/test_ladder.py`), and every question below is
a selection out of it: `--cases / --sizes / --arms / --sinks`, plus `-k`.

```bash
uv sync --group bench

# every rung docs/about/benchmarks.md publishes. The size ladder and each sweep go
# to separate files: a run REPLACES its results file rather than adding to
# it, and the report takes as many files as you give it
uv run pytest bench --benchmark-memory --sizes xs s m l \
    --benchmark-json=bench/results/latest.json
uv run pytest bench --benchmark-memory --sizes d100 d50 d25 d08 --skip-gate \
    --benchmark-json=bench/results/density.json
uv run pytest bench --benchmark-memory --sizes n002 n008 n032 n128 --skip-gate \
    --benchmark-json=bench/results/declarations.json

uv run python -m bench.report bench/results/latest.json \
    bench/results/density.json \
    bench/results/declarations.json                         # -> markdown
uv run python -m bench.plot                                 # -> the chart page

# anything narrower than the published ladder: send it somewhere else
uv run pytest bench --cases dispatch --sizes m l --benchmark-json=/tmp/two.json
uv run pytest bench --sinks highs --benchmark-json=/tmp/highs.json
```

The committed `results/*.jsonl` are the provenance of the tables
`docs/about/benchmarks.md` publishes *today*, written by the pre-pytest harness. The
readers no longer parse them — a full ladder run replaces them with `.json`,
and until someone takes one on an idle machine the published numbers stand on
files nothing in this directory can still read.

A bare `pytest bench` is **not** the committed ladder: `--sizes` defaults to
`xs s m`, so it stops below the rung every interesting claim lives at.
Narrowing the run and then committing the file leaves the published tables with
no provenance, and nothing about the file looks wrong afterwards.

**`bench.plot` rewrites one line of `docs/about/benchmarks-scaling.html`** — the
`const DATA = {...};` literal — and nothing else. The page is a tracked source
file, so its markup and prose are reviewed in the diff like any other code and
only the measurements inside it are mechanical.

**Point `--benchmark-json` somewhere else for every run that is not the full
ladder.** Aim it at the committed `results/latest.json` and the run *replaces*
it, so a one-rung smoke test overwrites the provenance of every published table
with four measurements — silently, and in a file whose diff nobody reads
closely. `git checkout` gets it back; noticing is the hard part.

## What it measures

**Peak RSS and wall time**, per phase, for the same model built two ways:

| | `lp` | `highs` | `gurobi` |
|---|---|---|---|
| `lpspec` | `lps.build(...)` then `bound.write(...)` | `lps.build(...)` then `build_highs(...)` | `lps.build(...)` then `build_gurobi(...)` |
| `linopy` | `lpspec.linopy.build(...)` then `Model.to_file(io_api='lp-polars')` | … then `Model.to_highspy(set_names=False)` | … then `Model.to_gurobipy(set_names=False)` |

`gurobi` is opt-in (`--sinks gurobi`): it needs the `[gurobi]` extra, where the
other two need nothing a contributor does not already have. It measures the
same seam — `build_gurobi` never calls `optimize()`, `to_gurobipy` never
returns a solved model.

**`set_names=False` on the linopy side is load-bearing.** linopy names every
variable and constraint by default and neither of our solver sinks names
anything, so the default call would time a feature only one arm's model
carries. It is not a rounding error: naming is **82% of linopy's HiGHS
hand-off** (0.11s against 0.02s at 200k variables) and 35% of its Gurobi one.
The two arms have to end holding the same artifact or the number means
nothing — and the correction runs against us, which is the direction an
honest harness should err.

**The solver sinks stop at the handoff — `run()` / `optimize()` is never called.** That is the
whole discipline of it. HiGHS's simplex is the same work whoever filled the
model, so including it would swamp the phase this harness exists to measure and
publish a number about HiGHS under our name. Both arms end holding a populated
`highspy.Highs` and neither runs it, which is the only reason the two are
comparable: `Model.to_highspy()` is the same seam on linopy's side.

`highs` is the sink most callers actually reach for, and it is **not the lp sink
minus a file** — HiGHS's own dense model is resident in both arms and narrows
the gap between them. Measuring only the LP path reports the wrong number for
the common case, which is why both run by default.

Both arms read the same parquet files and end at the same seam — an LP file on
disk, or a populated `highspy.Highs` — so the comparison is one language, one
destination, two engines. The linopy arm is the right
comparison and the only one worth making first: it accepts *exactly* the same
YAML (docs/about/architecture.md hard rule 3), which is what makes it the oracle rather
than a rival dialect.

Not measured, deliberately: solve time (that is HiGHS, identical either way, and
it would swamp the build), and anything about expressiveness.

**A number the run cannot stand behind is marked, not dropped.** Every
measurement's distribution — `iqr`, `median`, `rounds` — is carried into the
result file beside the minimum the tables publish, and `bench.report` appends
`~` to any wall cell, and to the ratio beside it, whose IQR exceeds
`SPREAD_BUDGET` of its own median. That is the case `min` cannot survive: not
one wild round, which the minimum ignores by construction, but *every* round
slow, which leaves no clean one to fall back on — #797 is the cell that was
publication-ready at 2.33x wrong. A marked cell is one to re-take on an idle
machine, never one to quote.

**Nine rounds is the floor, and the harness sets it.** pytest-benchmark's own
default is 5, and its calibration gives the fewest rounds to the slowest cells —
exactly where sustained interference is most likely and a clean round hardest
to come by. `--benchmark-min-rounds` still wins where a run wants more.

## Why it is built this way

**One process per measurement.** Peak RSS is a property of a process: a second
arm in the same interpreter inherits the first's high-water mark and its warm
allocator. `@pytest.mark.benchmem(isolate=True)` is what gives each pass its own
interpreter, and it is the same declaration that makes whole-process `rss`
available at all.

**`rss`, not the memray peak, for anything published.** pytest-benchmem records
both. `rss` is the whole-process high-water mark — the number `/usr/bin/time -l`
agrees with — and it is the only one honest across two libraries; the memray
peak is deterministic and attributable to a call stack, which is what makes it
right for comparing lpspec to itself. Both are in every result file, and which
one a table reads is a decision, not an accident. The measured reason is below.

**The harness is pytest, and deliberately nothing more.** Selection, the
ragged parametrization, per-pass isolation, the JSON, the repeats and the
minimum are all things pytest and its plugins already do and have tested. What
is left in this directory is what is specific to lpspec: the cases, the verbs,
and the parity gate.

**The parity gate runs before any timing.** The smallest rung of each case is
solved on both arms and the objectives compared to 1e-9 relative; a mismatch
aborts the whole run. The differential test suite proves the two lanes agree on
the *language* — it says nothing about the data this harness generates, and a
performance number describing two different models is worse than none.

## Where the clock starts and stops

The easiest way to publish a wrong number is to time something in one arm that
the other never does. The boundaries are therefore explicit:

| | lpspec | linopy |
|---|---|---|
| **before the clock** | splitting parquet paths into parameters vs dimensions (harness bookkeeping — it re-parses the YAML only because the *runner* decides which file is which) | — |
| `import` | `import lpspec` | `import lpspec.linopy` → linopy, xarray |
| `build` | `lps.build(...)` — the engine scans the parquet itself | `read_parquet` + reshape + `lpspec.linopy.build(...)` |
| `emit` | `bound.write(path)` / `build_highs(_tables(bound))` | `Model.to_file(path, io_api='lp-polars')` / `Model.to_highspy()` |
| `teardown` | `bound.close()` — releases the built model | — (nothing to release) |
| **after the clock** | row, column and nonzero counts off the built frames | `nvars` / `ncons` |

Three of those are deliberate calls rather than defaults:

- **Import is excluded from `wall_seconds`** but recorded. It is fixed, paid
  once per process, and at the `xs` rung linopy's import alone exceeds lpspec's
  entire build — including it would make the small end meaningless.
- **Teardown is included, and it is now near-free.** It was there to charge the
  arm holding a scratch database for releasing it. There is no scratch database
  any more — `close()` drops frames this process owns — so the phase is kept as
  a tripwire rather than a cost: if it ever stops reading ~0, something
  acquired a lifetime again.
- **`progress=False` is passed to linopy.** Its default is
  `m._xCounter > 10_000`, so every rung above `xs` would render tqdm bars that
  the lpspec arm has no equivalent of — ~7% of the write at 10M variables, and
  stderr noise in a harness that parses stdout.

Both arms start from the same parquet files and stop at the same seam, so
each pays for its own data ingestion. That is the honest unit; note that the
*phases* are not comparable one-for-one, because linopy defers coefficient
materialisation to `to_polars()` inside `to_file` — its `build` allocates dense
arrays and little else. Compare totals, and read the phases as attribution
within an arm.

**Peak RSS is the whole cost, because nothing spills to disk.** An engine that
traded RAM for a workdir could show a peak-RSS win while holding a
multi-gigabyte temp file, and the harness once recorded `workdir_bytes` to stop
that. Neither arm writes anything but the LP file now, so that field is gone
rather than left reading zero — a column that is always 0 reads as "measured
and fine", which is the same failure in the other direction. Restore it in
`bench/workloads.py` if a sink ever spills again.

**Failures are results.** A run that dies is written to the JSONL with the
exception line that killed it, and the report renders it as a cell. An OOM is
the single most informative thing this harness can find — and this is where a
cost claim is settled, because cost is not one of the architecture's rules.

**Repeats collapse by minimum.** Noise only ever adds.

**Comparing two versions of the same arm? Alternate them.** Repeats inside one
invocation collapse noise *within* a few seconds; they do nothing about drift
across a session, and this machine has drifted 2x on wall time between the
start of a session and the end of one. Check out A, measure, check out B,
measure, and go back — not A once and B once an hour later. The tell that you
needed to is the other arm: if linopy moved too, the machine moved, because
nothing in `src/lpspec/relational/` can reach it. Peak RSS is far steadier than
wall time and is usually the honest half of a before/after claim.

## The cases

Chosen so each stresses a *different* SQL shape (docs/about/architecture.md, "read the
verdict off the SQL"), not to cover the language:

| case | shape | why |
|---|---|---|
| `dispatch` | pointwise bounds + one `sum` per row | raw throughput, and the case a dense eager broadcast is best at — so our worst ratio |
| `nodal` | `(snapshot, node, tech)`, `where: installed > 0` | sparsity as it actually occurs — see below |
| `transport` | three `sum(by=)` joins per row | the mapping-table path, where the eager lane must materialise a bus x generator product |
| `sector` | dense snapshots x dense carriers x sparse portfolio | mixed density in one model — the shape a sector-coupled model actually has, and where the sparsity claim is visible |
| `storage` | a cyclic `shift` recurrence | the self-join, and the only locality class with no eager cost analogue: xarray shifts an array, we join a term stream against itself on `snapshot.ord - 1` |
| `commitment` | dispatch gated by a binary `u`, `p <= p_max * u` | the MILP — the only case whose `vtype` stream is not all-continuous, so integrality reaches every sink at scale |

**`nodal` is the case worth explaining.** It is dispatch over nodes and
technologies, and a technology only generates at a node where it is installed:
no offshore wind inland, no hydro without a river. PyPSA spells that by
attaching generators to buses, Calliope by declaring techs at nodes; in YAML it
is a `where` over the capacity table. 50 nodes x 12 technologies is 600
coordinates per snapshot, of which 3 per node — a quarter — exist. That gap is
the comparison: relationally an absent pair is an absent row, eagerly it is a
NaN that still costs eight bytes and a broadcast.

The sparsity is *structural and time-invariant*, which is not incidental —
`installed` carries node and tech but not snapshot. A random Bernoulli mask
would sweep the same densities while misrepresenting the shape, and the shape is
what an engine can exploit.

**Measured, this sweep alone does not show it** — at a 1.2M coordinate product
a dense array over it is ~10 MB and the fixed cost of the process dominates.
`sector` runs the same sparsity at a 12M product and the effect is plain. See
[docs/about/benchmarks.md](../docs/about/benchmarks.md#the-density-sweep-and-a-claim-it-refuses).

`Shape.density` (technologies per node: 12 / 6 / 3 / 1) is swept at one model
size, because sweeping size and density together leaves no way to tell one
effect from the other. Run the full ladder with `--sizes all`.

**The declaration count is the third swept axis.** Every size rung grows
`snapshot` and holds its case's declaration count fixed, so a cost paid *per
declaration* — a labelled frame each, a stack at the end — is sampled at
whatever counts the cases happen to have. The `declarations` case splits a
fixed pool of 512 units per snapshot into 2 / 8 / 32 / 128 variable
declarations (rungs `n002`…`n128`), each with its own capacity constraint and
objective term and one balance over all of them, at one model size for the
density sweep's reason. Its model YAML varies per rung, so it is generated —
`_declarations_model` in `bench/cases.py` — and cached beside the rung's data.

**The report measures what survived rather than trusting the declaration.**
`dispatch` declares `where: p_max > 0` against a p_max that is always positive,
so its mask removes nothing and the engine pays for it anyway; the `live` column
says `100%` and makes that visible instead of leaving it as a trap. Keeping that
vacuous mask is itself a measurement, which is why `nodal` is a separate case
rather than a fix to `dispatch`'s data.

Data is generated deterministically (a blake2b digest of the shape seeds the
RNG — `hash()` is salted per process and would give the two arms different
numbers), cached under `bench/.cache/`, and feasible by construction.

**`commitment` is a MILP, and the gate still costs one cheap solve.** The gate
solves only the *smallest* rung of a case, once per arm, and the measured pass
never solves at all — so the `l`/`xl` rungs of a MILP ladder cost the gate
nothing. What the case has to guarantee is that its bottom rung solves to
proven optimality: `GATE_RTOL` is 1e-9 and HiGHS's default `mip_rel_gap` is
1e-4, so a rung where branch and bound stops at a gap could hand the two arms
different incumbents. The bottom rung is therefore deliberately tiny, with
every cost a distinct float — there is no MIP-aware tolerance, and that is a
decision rather than an omission.

## The speed-of-light floor

The ladder's ratios have linopy as their only denominator, which ranks two
engines without saying how much headroom either has left. `bench/floor.py` is
the missing denominator: it hand-writes **one** model — `transport` — from the
case's cached parquet straight into numpy arrays and a CSR matrix, no lpspec
and no expression engine anywhere in the path, and ends at the same seam as
the `highs` sink: a populated `highspy.Highs` with `run()` never called. What
it costs is the irreducible price of emitting the coefficients, and with it
the sentence becomes *"we are at Nx the floor and linopy is at Mx"* — a claim
about engineering rather than a ranking.

```bash
uv run python -m bench.floor l            # phase minima + peak RSS
uv run python -m bench.floor xs --check   # one solve each way, objectives compared
```

It is **not a fourth arm**: it hardcodes one model, so it has no place in the
`case x size x sink x arm` product, and its numbers are quoted beside the
ladder's rather than inside it. `--check` solves the smallest rung through the
floor and through lpspec and compares objectives at the gate's tolerance;
`bench/test_harness.py` pins the cheaper fingerprint — the floor's column, row
and nonzero counts against lpspec's — on every bare `pytest bench`.

## The warm-start payoff

*Does carrying a basis across a genuine rebuild pay?* is the question #382 has
to answer before the engine work is worth writing, and until this module there
was nothing in the tree to answer it on: `examples/benders/run.py` is the only
driver that rebuilds a model every iteration, and its master is 3 columns and
25 rows, where a cold solve costs one simplex iteration.

`bench/warm_payoff.py` is that missing case — a capacity-expansion Benders
whose master is sized from data (`bench/expansion/*.yaml`), with the master
solved three ways at every rebuild: cold, from the previous iteration's basis
spliced per declaration, and from that basis merely truncated to the new
height. The subproblem is a real dispatch LP and is deliberately *not* what is
measured: `cap_hat * avail` reaches the rows as a right-hand side, so a new
capacity pushes values onto the loaded solver and never rebuilds.

```bash
uv run python -m bench.warm_payoff s m l --steps 400
uv run python -m bench.warm_payoff m --wall   # only on an idle box
```

**Simplex iterations are the measurement.** They are deterministic, so this
ladder needs no idle machine, and they are the quantity a basis actually moves.
`--wall` prints seconds and the load averages beside them, and carries none of
the argument.

It is **not an arm** — like `floor.py` it hardcodes one model, prints its own
table and never touches the ladder's results files — and it is **not a
feature**: no `src/` code carries a basis across a rebuild, and the splice
lives here so that the evidence could be taken before the engine work was
written. Its models sit under `bench/expansion/` rather than `bench/models/`,
which `tests/test_bench_models.py` reserves for files backing a ladder case.

The splice exists because **rows do not append**: a master with two cut
families numbers rows per declaration, so a row gained by `optimality_cut`
shifts every row of `feasibility_cut`. A wrong carry cannot produce a wrong
answer — a basis moves the route, not the optimum — so the third arm is how the
splice is shown to be worth its complication at all.

## The other question: regressions

*Did this change make it worse?* is a different question from *how do we compare
to linopy*, and it wants a different metric — but it does not want a different
harness. It is the same suite with one lane selected:

```bash
uv run pytest bench --arms lpspec --sizes s m --benchmark-memory
uv run pytest bench --arms lpspec --sizes s m --benchmark-memory \
    --benchmark-memory-compare=0001 --benchmark-memory-compare-fail=mean:10%
```

That is what `.github/workflows/bench.yml` runs, twice — once against the pull
request's base and once against its head — and what it gates on.

**Why the metric changes with the question.** Measured on `dispatch/m`:

| arm | `ru_maxrss` | memray peak |
|---|---|---|
| lpspec | 309 MB | 211 MB |
| linopy | 604 MB | **2967 MB** |

memray counts polars' reserved arenas as allocated and does not count the
interpreter or mapped libraries at all, so the bias points in *opposite*
directions in the two lanes: the peak ratio is 0.51x by RSS and 0.07x by memray.
A published cross-library claim built on that would be false the moment a reader
ran `/usr/bin/time`. Within one lane the same bias sits on both sides of a diff
and cancels, leaving a metric that is deterministic and attributable to a call
stack — which RSS, sensitive to machine load, is not.

So: `rss` for the comparison we publish, the memray peak for the regressions we
chase. Both come out of the same run — the choice is which column a table reads,
and `--benchmark-memory-compare-fail` is what turns the second into a gate.

## The same suite, a third instrument: CodSpeed

`bench/` is a plain `pytest-benchmark` suite, so the fixture its tests ask for
is whichever plugin is loaded. That is not a detail — it is why there is no
second set of benchmarks in this repository:

```bash
uv run pytest bench --benchmark-memory   # memray peak + rss + timing
uv run pytest bench --codspeed           # what CI measures
```

`--benchmark-memory` patches the stock fixture and reads the `benchmem` marker;
`--codspeed` replaces the fixture outright and the marker goes inert. Same
tests, same workloads, same rungs — a different instrument. The workloads
cannot drift between them, because there is one of them.

[CodSpeed](https://codspeed.io) runs on every pull request
(`.github/workflows/codspeed.yml`): one ~3-minute job, free runner, no secret.
What it adds over `bench.yml` is not the metric but **the baseline** —
`bench.yml` can only compare against a base it checks out and measures itself,
which costs two passes and is why it waits for a `trigger:bench` label. CodSpeed
stores the number for every commit on `main`.

Only the `memory` instrument runs. `walltime` needs CodSpeed's metered
bare-metal runners to say anything a shared runner's clock cannot, and
`simulation` — their default — runs the workload under an emulator, which suits
neither multi-threaded native code nor these rungs.

**It gates nothing.** The job is `continue-on-error` and no ruleset names it;
`bench.yml` remains the check that fails a pull request. It also needs a
maintainer to connect the repository to the CodSpeed GitHub app — until then the
workflow runs and uploads nothing.

## Adding a case

Add a YAML file under `bench/models/`, a data generator and a ladder to `CASES`
in `bench/cases.py`, and a function turning the same parquet paths into the
linopy lane's `sources` shapes. Nothing else: the parametrization reads
`CASES`, and the gate and the report are case-agnostic.

A case whose YAML has to vary per rung sets `generate_model` instead of
`model` — `declarations` is the template — and `Case.model_path(shape)` hands
every consumer whichever of the two the case has.

## The map

| file | |
|---|---|
| `cases.py` | the models, the data generators, the ladders |
| `workloads.py` | what is measured — one verb per arm, picklable, lpspec imported inside |
| `conftest.py` | selection flags, the ragged parametrization, the data fixture, the parity gate |
| `test_ladder.py` | the two benchmarks: build-and-emit, and rebuild-in-one-process |
| `results.py` | pytest-benchmark JSON -> the flat records the report and the plot read |
| `floor.py` | the speed-of-light floor — `transport` hand-written into a populated `Highs`, no engine involved |
| `warm_payoff.py` / `expansion/` | does a basis carried across a rebuild pay? A scaled Benders, its master solved cold and warm at every rebuild |
| `report.py` / `plot.py` | the published tables, and the chart page's data literal |
| `profile_build.py` | which *query* inside one build spends the time — a profiler, not a benchmark. Wraps every collect, so read its shares and not its seconds |
| `ab.py` | is a change faster, or is the machine noisy — two arms, a bootstrap interval, and a verdict that refuses to call a winner when it crosses zero |
| `profile_phases.py` | which *phase*, in seconds comparable to a real run. Hoists the parse, the lowering and the parquet read out of the loop and reuses one binding, which takes the spread from 12-55% down to a few percent — the difference between a 10% change being visible and not |

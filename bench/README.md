# The performance harness

Not shipped in the wheel, not imported by `lpspec`, not run in CI. It exists so
that [docs/benchmarks.md](../docs/benchmarks.md) has a *provenance* — the last
set of published numbers came from a `scratch/` script that was deleted, and a
claim nobody can re-run is a claim with a shelf life.

It is **one pytest suite** (`bench/test_ladder.py`), and every question below is
a selection out of it: `--cases / --sizes / --arms / --sinks`, plus `-k`.

```bash
uv sync --group bench

# every rung docs/benchmarks.md publishes. The size ladder and the mask sweep
# go to separate files: a run REPLACES its results file rather than adding to
# it, and the report takes as many files as you give it
uv run pytest bench --benchmark-memory --sizes xs s m l \
    --benchmark-json=bench/results/latest.json
uv run pytest bench --benchmark-memory --sizes d100 d50 d25 d08 --skip-gate \
    --benchmark-json=bench/results/density.json

uv run python -m bench.report bench/results/latest.json \
    bench/results/density.json                              # -> markdown
uv run python -m bench.plot                                 # -> the chart page

# anything narrower than the published ladder: send it somewhere else
uv run pytest bench --cases dispatch --sizes m l --benchmark-json=/tmp/two.json
uv run pytest bench --sinks highs --benchmark-json=/tmp/highs.json
```

The committed `results/*.jsonl` are the provenance of the tables
`docs/benchmarks.md` publishes *today*, written by the pre-pytest harness. The
readers no longer parse them — a full ladder run replaces them with `.json`,
and until someone takes one on an idle machine the published numbers stand on
files nothing in this directory can still read.

A bare `pytest bench` is **not** the committed ladder: `--sizes` defaults to
`xs s m`, so it stops below the rung every interesting claim lives at.
Narrowing the run and then committing the file leaves the published tables with
no provenance, and nothing about the file looks wrong afterwards.

**`bench.plot` rewrites one line of `docs/benchmarks-scaling.html`** — the
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
| `lpspec` | `lps.build(...)` then `ex.write(...)` | `lps.build(...)` then `build_highs(...)` | `lps.build(...)` then `build_gurobi(...)` |
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
YAML (docs/ARCHITECTURE.md hard rule 3), which is what makes it the oracle rather
than a rival dialect.

Not measured, deliberately: solve time (that is HiGHS, identical either way, and
it would swamp the build), and anything about expressiveness.

### The `polars` arm

`--arms lpspec polars` times the same lane on the other engine. It is not a
third code path: the arm runs the `lpspec` arm with `LPSPEC_ENGINE` set for
that child, which is the switch a caller has — so the harness measures the
shipped mechanism rather than one only it knows about. One process per
measurement is what makes an environment variable the right tool here; there is
nothing to reset afterwards.

`lpspec` is the *default* engine, so the plain arm measures duckdb and the
named arm measures polars: the one that has to be asked for is the one not
shipped by default. Nothing installs separately — both engines are runtime
dependencies. The findings are in [duckdb-spike.md](duckdb-spike.md), which
labels its columns by engine name rather than by arm so that a ratio there
survives the default moving.

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
| `emit` | `ex.write(path)` / `build_highs(ex._tables())` | `Model.to_file(path, io_api='lp-polars')` / `Model.to_highspy()` |
| `teardown` | `ex.close()` — releases the built model | — (nothing to release) |
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

Chosen so each stresses a *different* SQL shape (docs/ARCHITECTURE.md, "read the
verdict off the SQL"), not to cover the language:

| case | shape | why |
|---|---|---|
| `dispatch` | pointwise bounds + one `sum` per row | raw throughput, and the case a dense eager broadcast is best at — so our worst ratio |
| `nodal` | `(snapshot, node, tech)`, `where: installed > 0` | sparsity as it actually occurs — see below |
| `transport` | three `sum(group_by=)` joins per row | the mapping-table path, where the eager lane must materialise a bus x generator product |
| `sector` | dense snapshots x dense carriers x sparse portfolio | mixed density in one model — the shape a sector-coupled model actually has, and where the sparsity claim is visible |
| `storage` | a cyclic `shift` recurrence | the self-join, and the only locality class with no eager cost analogue: xarray shifts an array, we join a term stream against itself on `snapshot.ord - 1` |

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
[docs/benchmarks.md](../docs/benchmarks.md#the-density-sweep-and-a-claim-it-refuses).

`Shape.density` (technologies per node: 12 / 6 / 3 / 1) is swept at one model
size, because sweeping size and density together leaves no way to tell one
effect from the other. Run the full ladder with `--sizes all`.

**The report measures what survived rather than trusting the declaration.**
`dispatch` declares `where: p_max > 0` against a p_max that is always positive,
so its mask removes nothing and the engine pays for it anyway; the `live` column
says `100%` and makes that visible instead of leaving it as a trap. Keeping that
vacuous mask is itself a measurement, which is why `nodal` is a separate case
rather than a fix to `dispatch`'s data.

Data is generated deterministically (a blake2b digest of the shape seeds the
RNG — `hash()` is salted per process and would give the two arms different
numbers), cached under `bench/.cache/`, and feasible by construction.

A MILP through the `highs` solver is the next rung — see docs/benchmarks.md.

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
linopy lane's `data=`/`coords=` shapes. Nothing else: the parametrization reads
`CASES`, and the gate and the report are case-agnostic.

## The map

| file | |
|---|---|
| `cases.py` | the models, the data generators, the ladders |
| `workloads.py` | what is measured — one verb per arm, picklable, lpspec imported inside |
| `conftest.py` | selection flags, the ragged parametrization, the data fixture, the parity gate |
| `test_ladder.py` | the two benchmarks: build-and-emit, and rebuild-in-one-process |
| `results.py` | pytest-benchmark JSON -> the flat records the report and the plot read |
| `report.py` / `plot.py` | the published tables, and the chart page's data literal |
| `profile_build.py` | which *query* inside one build spends the time — a profiler, not a benchmark. Wraps every collect, so read its shares and not its seconds |
| `profile_phases.py` | which *phase*, in seconds comparable to a real run. Hoists the parse, the lowering and the parquet read out of the loop and reuses one binding, which takes the spread from 12-55% down to a few percent — the difference between a 10% change being visible and not |

# The duckdb engine: what it costs, and what it buys

**Status: settled, and this is the provenance.** `LPSPEC_ENGINE=duckdb`
selects the engine in `relational/engines/duck/`; the numbers the package
claims for it — in `pyproject.toml`, in the engine's own docstring, in SPEC —
are the ones below.

It began as a costing exercise for reversing `c11a0dd` (PR #189, 2026-07-28),
which is why it reads like one: the question was what a duckdb engine would
*touch* and *break*. The answer turned out to be "less than the line count
suggests", and the outcome was neither a reversal nor a rejection — both
engines ship, and a caller picks. §9 is where that lands.

Kept in this shape on purpose. The engine decision has flipped once already and
the reasoning did not survive on its own; a claim nobody can re-run is a claim
with a shelf life.

**Method:** the tree read module by module for the blast radius; the two
engines measured head to head on the full ladder.

The numbers below were taken when duckdb was still a *foreign checkout* — the
engine as it shipped before #189, reached through another interpreter — which
was the only honest way to price it before a port existed. It is now an engine
in the tree (`relational/engines/duck/`), so re-running is:

```bash
uv sync --group bench
uv run pytest bench --arms lpspec polars --benchmark-memory \
    --sizes xs s m l --sinks lp highs \
    --benchmark-json=bench/results/engines.json
uv run python -m bench.report bench/results/engines.json --arms lpspec polars
```

`lpspec` is the *default* engine, so that pair is duckdb against polars. Every
ratio in this file is written `duckdb ÷ polars` — by engine name rather than by
arm, so it stays readable whichever one the default names.

**Two results files, and they are not the same measurement.**
`bench/results/engines.json` is the in-tree engine and is what the table below
reports. `bench/results/duckdb-spike.jsonl` is the *old*, out-of-tree engine,
kept as the measurement §7's argument was built on rather than re-baselined —
provenance, not the current cost.

## Where it stands, measured at `9257b54`

All seven cases, both sinks, `l` rung, five rounds on a clean tree, taken by
the pytest harness in one process per rung so the two arms are adjacent rather
than a run apart. Provenance: `bench/results/engines.json`. `build` is the
steady-state rebuild (`test_rebuild`) and so has no sink column.

| `duckdb ÷ polars`, `l` rung | build | peak lp | wall lp | peak highs | wall highs |
|---|---:|---:|---:|---:|---:|
| `sector` | 3.05× | 1.32× | **2.36×** | 1.14× | **2.84×** |
| `fleet` | 2.75× | 1.04× | 1.56× | 1.11× | 2.18× |
| `transport` | 2.00× | **1.70×** | 1.46× | 1.37× | 1.78× |
| `profiled` | 1.96× | **1.84×** | 1.44× | 1.09× | 1.79× |
| `storage` | 1.93× | 1.47× | 1.37× | 1.21× | 1.63× |
| `dispatch` | 1.84× | 1.00× | 1.19× | **0.94×** | 1.52× |
| `nodal` | 1.59× | 1.08× | 1.28× | **0.90×** | 1.43× |

**Slower on all fourteen rungs, and heavier on eleven of them.** It is lighter
only on `dispatch` and `nodal` through the solver, and level with polars on
`dispatch` through the LP file. The objectives are bit-identical either way —
the parity gate is enforced at measurement time and every one of the 169
measurements above built the same model, which is the claim this engine was
made to make and the one thing that has never moved.

**The spread narrowed and the middle got worse.** Against the previous run
(`e42b9a0`, six cases) build went 1.3–3.4× → 1.6–3.1× and wall 1.07–2.19× →
1.19–2.84×. Peak is the interesting column: it was 0.91–1.59× and is now
0.90–1.84×, because the CSR matrix (#550) halved the largest frame on *both*
engines and so stopped hiding a difference that was never in the matrix. What
is left is `transport` and `profiled` writing an LP file at 1.7–1.8×, which is
where this engine's real memory gap now lives.

**Two things widened this, and one of them is not the engine's fault.** Five
optimisations landed on the polars engine while this one sat on a branch
(#408, #412, #413, #414, #415), and then two more (#421, #433). `cols` going
positional is the instructive one: on polars the order falls out of a cross
join's emission order and costs nothing, while here it needs an `ORDER BY`,
which roughly doubled `dispatch/l`'s build the day it landed. **A shared
contract can be cheap for one engine and not the other, and the gap widens with
neither engine regressing.** That is a structural fact about shipping two, and
it is worth more than any single row above.

**The memory ceiling is not an engine question**, which this engine is the
reason we know. duckdb has the knob — `SET memory_limit`, spilling past it — and
it does not bind. `dispatch/xl`, 40M columns, built in a fresh process with the
data written by another one:

| | peak RSS | of which the four frames |
|---|---:|---:|
| duckdb, `memory_limit='500MB'` | 4.49 GB | 1.65 GB |
| duckdb, default limit | 4.49 GB | 1.65 GB |
| polars | 4.02 GB | 1.65 GB |

**Capped and uncapped are the same number to the last measured digit**, which
is the claim: the engine's own working set was already under the cap, so the
knob has nothing to bind on.

**But the second column has inverted since this was last taken, and that
matters more than the first.** The figure this section used to quote was 2.98 GB
peak with 2.61 GB of it the frames — peak *was* the output, and the conclusion
drawn was that a declared bound is therefore a `ModelTables` question. It is not
that any more. CSR (#550) halved the matrix on both engines, so the frames are
1.65 GB where they were 2.61, and peak did not follow them down: **the frames
are 37% of peak, and the build transient is the larger term** — on polars too,
which is why this is not a duckdb finding. A bound declared over `ModelTables`
would now miss most of what a build actually costs.

(The 2.98 GB is not directly comparable — different commit, different polars,
and it is not re-measured here. What is measured on this tree is the table
above, and the composition is what the argument turns on.)

**Still untested rather than refuted:** every rung here fits in RAM, so the
argument this engine was built on — a model that does not — has not been
measured either way.

---

## 1. The seam that survives

`relational/plan.py` (391 lines) is frozen dataclasses with no engine import —
`Expression`, `Predicate`, the five `*Declaration` types, `Program`. It is the
contract, and it is genuinely engine-agnostic: the duckdb engine consumed the
same shapes before #189. Everything above it survives untouched:

| Survives | Lines | Why |
|---|---:|---|
| `language/` (all) | 2,727 | never sees the engine; hard rule 1 |
| `lowering.py` | 382 | AST → plan; touches no data |
| `linopy/` (all) | 1,317 | separate lane, xarray-side |
| `typeset/` (all) | 1,351 | reads the schema, not the plan |
| `relational/plan.py` | 391 | frozen, engine-free |
| `relational/chunking.py` | 44 | pure arithmetic |
| `relational/status.py` | 104 | no polars reference |

That is the good news and it is real: **~6,300 lines never move.** The plan
being the contract is what makes this a swap rather than a rewrite, exactly as
#189 relied on in the other direction.

## 2. The seam that breaks

Everything between the plan and the sink is written in polars expressions.

| Rewritten | Lines | polars refs | Notes |
|---|---:|---:|---|
| `relational/compiler.py` | 717 | 55 | plan → LazyFrame. The bulk of the work |
| `relational/executor.py` | 516 | 43 | build orchestration, `ModelTables` assembly |
| `relational/binding.py` | 269 | 25 | sources → frames; the parquet path |
| `relational/labels.py` | 242 | 10 | algorithm survives, expressions do not |
| `relational/sinks/lp_file.py` | 189 | 38 | LP text emit, float formatting |
| `relational/data_validation.py` | 189 | 18 | one-row-per-coordinate, containment |
| `relational/frames.py` | 108 | 16 | the Arrow recogniser |
| `relational/result.py` | 190 | 6 | `primal`/`dual` joins, `to_*` bridges |
| `relational/sinks/tables.py` | 60 | 5 | contract type changes, logic does not |
| `relational/sinks/highs.py` | 246 | 11 | mostly numpy at the boundary; lightest |

**~2,300 of the engine's 3,317 lines rewritten**, with `highs.py` and
`tables.py` partially spared. Larger than the ~1,500 my note recorded for the
2026-07-27 estimate, because the engine has since grown `chunking.py`,
`data_validation.py`, a split `sinks/` package and a three-strategy `labels.py`.

**The porting surface itself is shallow.** The whole polars API in use is
`.collect`/`.lazy` (41), `.cast` (16), `.over` (11), `.with_row_index` (6),
`.when` (5), `.concat_str` (5), `.shift` (4), `.fill_null` (3),
`.scan_parquet` (2), `.is_in` (2), and one each of `.struct`, `.sink_parquet`,
`.sink_csv`, `.searchsorted`. Every one has a direct SQL counterpart —
`.over` → window function, `.with_row_index` → `ROW_NUMBER`, `.shift` → `LAG`.
Nothing exotic is stranded. The cost is volume and re-derivation, not a
capability gap.

## 3. What has to be re-derived, not translated

Four things where a line-by-line port is the wrong instinct:

- **Absence propagation** (`compiler._propagate_absence`, SPEC §6). A masked
  variable takes its row with it rather than contributing zero. This is the
  semantics that measured 25.0 against 125.0 when linopy's `legacy` default got
  it wrong — `linopy/__init__.py` sets v1 globally because of it. Any new engine
  re-earns this against the differential oracle.
- **Translate/shift edges** (`_translate_fragment`, `_vacated`, `_filled_edge`,
  SPEC §7). Vacated positions drop; filled edges do not. Subtle enough that the
  two lanes still carry a known disagreement — `test_resolution_parity.py`'s
  xfail on orphaned constraint rows.
- **Labels** (`labels.py`). Three strategies — arithmetic, factored, counted —
  chosen by how much of the coordinate product survives the mask, and *they must
  agree with each other*. The concept is portable and in fact duckdb-shaped
  already: my 2026-07-27 note records that writing the label as arithmetic
  instead of a window was what made duckdb competitive there (0.44 s / 0.14 GB),
  because global windows do not spill. This is the one place the port would
  inherit a solved problem.
- **Stable output** (#109). `sinks/README.md` records the hazard: a parallel
  join returns a group in whatever order it finished, so a sink that gathers a
  row's terms then orders rows has already lost order within one. duckdb
  parallelises the same way. The fix ports (carry a sort key, sort once), but
  the property needs re-proving.

## 4. Outside `src/`

- **Tests.** 109 polars references across 15 files. `test_compiler.py` (355
  lines) is the concentrated cost — it asserts against `PolarsCompiler.explain()`,
  which is ARCHITECTURE's admissibility test in executable form; under duckdb it
  becomes reading a SQL string, which is what it was before #189.
  `test_relational.py` (1,080) is mostly behavioural and survives; the
  differential oracle against the linopy lane is engine-blind by construction
  and is the thing that makes the port checkable at all.
- **Bench.** 4 files reference polars; `bench/_run_case.py`'s lpspec arm is
  engine-blind already (it passes parquet paths). `docs/benchmarks.md` and
  `benchmarks-scaling.html` are wholly re-measured — every published number is
  engine-specific.
- **Docs.** ARCHITECTURE (8 refs, incl. hard rule 2's text and the `.explain()`
  admissibility test), benchmarks (7), guide (2), SPEC (2), index (1),
  ROADMAP (1). Plus `pyproject.toml`'s description and keywords, which #189
  rewrote specifically to stop claiming "at any scale" and "streaming".
- **Dependencies.** `polars` out; `duckdb` and `pyarrow` back in. This reverses
  CLAUDE.md's "no dataframe library beyond polars" and changes what the
  bare-install CI job proves. Net footprint grows: duckdb's wheel is larger than
  polars', and pyarrow returns as a hard runtime dep rather than a `[linopy]`
  extra.

## 5. The one user-visible break

`primal()` returns a `polars.DataFrame` — documented in SPEC §10, in
CLAUDE.md's API block, and in `docs/guide.md`. Under duckdb it becomes an Arrow
table or a duckdb relation. `to_pandas` / `to_dataarray` / `to_parquet` are
unaffected (they are bridges either way), and Arrow-in is unaffected (the
recogniser takes any PyCapsule exporter). But anyone doing
`sol.primal('p').filter(...)` in polars idiom breaks. Pre-1.0 and alpha, so
cheap — but it is the one thing that is not internal.

## 6. A prerequisite worth doing regardless

**Hard rule 2 says "Engine-internal naming encodes neither 'polars' nor
'yaml'". `PolarsCompiler` and `PolarsExecutor` violate it in 44 places, and
`tests/test_architecture.py` does not enforce that clause** — it checks the
import fences and the dependency fences, not the naming one. So the rule is
live in the doc and dead in the code.

Renaming to `Compiler` / `Executor` and adding the missing static check is
worth doing on its own merits, and it happens to shrink any future engine swap.
`lps.build` returns the executor as a public return type, so this is a small
public rename too.

## 7. Measured

Full ladder, both engines, both sinks: `bench/results/duckdb-spike.jsonl`.
Render it with

```bash
uv run python -m bench.report bench/results/duckdb-spike.jsonl --arms lpspec duckdb
```

The gate agrees to 0.0e+00 relative on all six cases, including `fleet` and
`sector`, which the duckdb checkout has never seen — the foreign arm builds
today's models on the old engine and reaches the same optimum bit for bit.
That is what makes the rest of this table a measurement rather than a guess.

**The duckdb arm ran at its own default `memory_limit='1GB'`.** Not a choice
made here — `fk.build` defaults to it, so this is the engine as it shipped
against the engine as it ships. It is also the whole difference in kind: one
arm has a ceiling and spills, the other's peak tracks the model.

### The write path, which is where a ceiling pays

`lps.write` → LP file, at the top two rungs. This is the column #189 never
published: its headline compares to a loaded solver, where the build is ~9% of
peak and any build-side difference is diluted.

| case | vars | wall: polars | wall: duckdb | peak: polars | peak: duckdb | lighter |
|---|---:|---:|---:|---:|---:|---:|
| dispatch `l` | 10M | 2.34 s | 7.67 s | 2.05 GB | 0.76 GB | **2.69×** |
| dispatch `xl` | 40M | 17.99 s | 28.23 s | 4.38 GB | 1.39 GB | **3.15×** |
| profiled `xl` | 48M | 36.73 s | 86.68 s | 7.48 GB | 1.78 GB | **4.20×** |
| nodal `xl` | 12M | 5.44 s | 9.73 s | 3.22 GB | 1.16 GB | 2.77× |
| sector `xl` | 4M | 4.23 s | 8.72 s | 2.02 GB | 0.96 GB | 2.11× |

**The peak gap widens with the model** — dispatch runs 1.11× / 1.25× / 1.63× /
2.69× / 3.15× up the ladder — which is what a ceiling looks like against a peak
that tracks the model. The wall gap does not widen with it: 3.27× at `l` but
1.57× at `xl` on dispatch, because polars is doing more work per byte once the
model stops fitting comfortably.

`fleet/xl` is the other half of the same fact: the duckdb arm **failed there**,
`OutOfMemoryException` at 953 MiB, twice. That is a 1 GB budget being too tight
for a 48M-variable model, not the engine breaking — and it is the behaviour the
ceiling exists for. It fails instead of taking the machine with it.

### The solver path, where it mostly does not

Same models, `solver_direct`. HiGHS's own model is resident in both arms, so it
dominates and the engines converge:

| case | vars | peak: polars | peak: duckdb | lighter |
|---|---:|---:|---:|---:|
| dispatch `xl` | 40M | 8.32 GB | 6.17 GB | 1.35× |
| profiled `xl` | 48M | 7.35 GB | 6.37 GB | 1.15× |
| profiled `l` | 12M | 2.56 GB | 2.70 GB | **0.95×** |

`profiled/l` is the one rung where polars is *lighter*, and it is worth not
smoothing over: the gap is not a law, it is a consequence of how much of the
build survives to the hand-off.

Wall time across the whole ladder: polars **1.6–4.0× faster**, consistent with
#189's 1.7–5.2× — the range narrows here because the top rungs are new.

### The binder's own shape

Separately, both engines on the one operation this spike started from — a
12M-row parquet with four columns the model never names (n=2):

| stage | polars | duckdb |
|---|---|---|
| project + materialise (what `binding.py` does per parameter) | 0.10 s / 750 MB | 0.46 s / 476 MB |
| scan feeds a group-by, nothing materialised | 0.11 s / 368 MB | 0.14 s / 128 MB |

Row two is the interesting one: **5.9× between 750 MB and 128 MB for identical
data**, because nothing becomes a resident table. That gap is not really about
the engine — it is about whether the binder materialises. `binding.py:116`
collects deliberately, and its comment records the measurement: making every
collect lazy cost 29% on a join-heavy model to save the 0.15 GB this one collect
saves alone.

## 8. What would have to be true to justify it

The trade is fixed and known: **slower, lighter, plus a settable ceiling and
models that exceed RAM.** So the case has to come from the ceiling, not the
constant factor. Two conditions — and the third one I originally listed here
turns out to be already answered, in the direction that favours a database.

1. **ROADMAP Track 5's declared ceiling becomes a real requirement** — "build
   this in N GB or fail", for models written once and run on a machine chosen by
   someone else.
2. **On the write path specifically.** Track 5 already records that the solver
   is roughly an order of magnitude above the build at 10⁷ variables, so a build
   ceiling is worth having for `lps.write` → LP/MPS handed elsewhere, and worth
   much less when the same process goes on to solve.

### 8a. Partitioning on polars was already measured, and it floors

`f5519b2` (2026-07-27, "PROTOTYPE — partition-wise assembly, measured") answers
this and corrects Track 5's framing in two ways:

- **Emission is the wrong stage.** Emit is 20% of peak on `dispatch` and 2% on
  `transport` (`bench/emit_peak.py`). The memory is in *assembly*, which
  partitions cleanly because `row` is the leading key of the terminal aggregate.
- **It works, and then it stops.** At the `l` rung: `dispatch` 2,213 → 1,710 MiB
  (6.4 → 4.4 s), `transport` 2,657 → 1,695 MiB (9.1 → 9.2 s). 23–36% off peak,
  wall-neutral or better.
- **But peak floors at ~1.7 GB independent of block size**, against duckdb's
  0.88 / 1.16 GB, because the matrix stops being the binding term and everything
  else is still resident: `cols`, the label frames, `rows`, `obj`, the
  parameters. In the commit's own words — *"bounding the build means
  partitioning all of them, which is a buffer manager in application code."*

So the escape hatch I proposed does not exist. Partitioning on polars buys a
constant factor and then hits a floor set by the number of resident frames, and
the remaining distance is a buffer manager. **That is the argument for a
database, and it is already measured.**

### 8b. duckdb mostly does *not* need hand-managed partitioning — and needs less than it did

My 2026-07-24 note ("global windows and ordered aggregates don't spill in
duckdb") was over-broad even then: `8484bb0` corrected it three days later to
*exactly two* forced sites — the global ordered `ROW_NUMBER` for label
assignment, and the LP-text `string_agg`. The plain numeric `GROUP BY` spilled
single-shot at a 256 MB cap.

Re-measured on duckdb 1.5.5, 35M rows, `memory_limit='256MB'`, each operator in
its own process, `COPY … TO parquet` so the optimizer cannot prune it:

| operator | engine role | result |
|---|---|---|
| `GROUP BY` | A-assembly | 5.1 s, **460 MB** peak, 35M rows out |
| `ROW_NUMBER() OVER (ORDER BY …)` | label assignment | 6.0 s, **512 MB** peak, 31.5M rows out |
| `string_agg(… ORDER BY …)` | LP-text sink | **OutOfMemoryException** at 244 MB |

**The label window now spills — that changed since July.** One of the two forced
sites is gone by upgrade. The other is the ordered `string_agg`, which still
OOMs exactly as recorded, but it is in the debugging sink and is avoidable: the
current polars `lp_file` already emits a frame of lines carrying its own sort
key and sorts once (`sinks/README.md`, "Stable output"), which is a shape that
does not need an ordered aggregate at all.

Caveats on this measurement: synthetic shapes, not the engine's real queries,
and the `sort_only` arm was confounded by input that was already in sort order,
so it is omitted from the table. The two rows that matter each wrote their full
output to disk (192 MB and 131 MB of parquet), which is what rules out the
optimizer having skipped the work.

## 9. Estimate

| | |
|---|---|
| engine rewritten | ~2,300 lines of 3,317 |
| survives untouched | ~6,300 lines above the plan |
| test churn | ~500 lines, concentrated in `test_compiler.py` |
| docs + bench | full re-measure of `benchmarks.md` + the scaling page; ~20 doc refs |
| public break | `primal()` return type |
| prerequisite | `Polars*` rename (44 sites) + the missing hard-rule-2 check |

Comparable to #189 in the other direction and somewhat larger.

**The line count was the wrong worry.** `bench/engine_diff.py` puts the two
implementations of each operator side by side, against the duckdb engine in
`relational/engines/duck/`, which builds the whole suite to frames identical to
polars' — same `row`, same `col`, exactly. Over the
operators both implement, the SQL is **1.05×** the polars line count. The
volume is not the tax.

Where the tax actually falls:

- **Broadcast joins.** `_join_mul` doubles, 12 lines to 25: SQL needs explicit
  column lists on both sides, alias qualification and null-safe
  `IS NOT DISTINCT FROM` predicates where polars gets all three from
  column-name semantics. Every fragment composition pays it.
- **`labels._factored`, 37 lines.** Was the one place polars was
  *algorithmically* ahead rather than incidentally. Now ported, and the port
  moved the *choice* — which of the three routes a mask allows — up into
  `plan.free_prefix`, where it is a question about the plan that both engines
  ask once rather than two implementations that could drift.
- SQL **wins** on `_compare` (15 → 4) and on the semi-joins, where
  `WHERE EXISTS` says plainly what `how='semi'` needs a reader to know is not
  a join.

**Where this leaves it.** Three things have moved since §1 was written, and all
three point the same way:

1. Partitioning polars to a bound was already measured and floors at ~1.7 GB
   (§8a). The escape hatch is not there.
2. The hand-managed partitioning duckdb used to need has largely evaporated
   (§8b) — the label window spills on 1.5.5.
3. The write path, now measured (§7), is **2.1–4.2× lighter under duckdb at the
   top rung, and the gap widens with the model.** On the solver path it is
   1.15–1.35× and once *negative*.

So the trade is not "slower and lighter, plus a knob". It is: **on the write
path, a much smaller build for 1.6–2.4× wall clock; on the solve path, almost
nothing for 2–3×.**

**What was done about it.** Neither reversal nor rejection: both engines ship
and the caller chooses, because the answer depends on which path a given
workload is on and nothing in the library can know that. `LPSPEC_ENGINE` and
not a parameter — an engine cannot change the answer, only what computing it
costs, so it does not belong in the call that produces one.

duckdb is the one an unset environment gets, which is a decision about
*measurement* rather than about which is faster: the default is what CodSpeed,
`bench.yml` and the unflagged CI pass all measure without being asked, and the
engine that is behind is the one that needs to be under those instruments.

That decision cost far less than §9's estimate implied, because most of an
executor turned out not to be engine work: both sinks and the whole solution
read-back are written against `ModelTables` and the label frames, so
`relational/engine.py` holds them once and an engine supplies four things. The
~2,300-line figure priced a replacement; a *second* engine is a compiler and an
assembler.

**Read §7 as provenance, not as the current cost, and read point 3 above as the
argument that was made rather than a property of the shipped engine.** The
committed results are the *foreign checkout*, and the in-tree port has never
reproduced its memory advantage: at `dispatch/xl` the two engines peak within
2% of each other. What the port has instead is the build phase — 2.66 s against
6.84 s at that rung, after five changes since it landed. So the trade is no
longer "lighter for slower"; it is faster at the top of the ladder, slower at
the bottom, and roughly level on memory. The table at the head of this file is
the current one; re-run the command beside it before quoting a ratio.

**Recommended next step, if this is live:** decide the write-path question
first. If the answer is no, this closes for good and
`relational/engines/duck/` should be deleted rather than left to rot. If the answer is yes, the cheaper move is
probably a duckdb-backed `lp_file` sink alone — the write path is where the
whole gap lives, and a sink is a module, not an engine.

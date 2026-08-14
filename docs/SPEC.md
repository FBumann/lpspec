# Language Reference

What a YAML file may contain and what it means. *Why* it is shaped this way:
[docs/ARCHITECTURE.md](ARCHITECTURE.md). What is planned or refused:
[docs/ROADMAP.md](ROADMAP.md). A worked example: [README](https://github.com/fluxopt/lpspec/blob/main/README.md#example).

## 0. The laws

Ten rules the whole language reduces to. Every section below elaborates one,
and each law names the section that does — so a rule stated here is not
restated there.

**Nothing is guessed.** Where a file does not determine the answer, loading
fails and the message names the rewrite. Every law is that one principle,
applied in a different position.

| # | Law | § |
|---|---|---|
| 1 | Nine top-level keys, and the schema is **closed at every level** — an unknown key is an error naming the near miss. Booleans are YAML 1.2, so `no` / `on` / `off` stay labels. | [§1](#1-file-shape) |
| 2 | Everything decidable without data is **decided without data**. | [§9](#9-errors) |
| 3 | **One flat namespace, no shadowing** — a collision is a load error naming both declarations. | [§5.1](#51-name-resolution) |
| 4 | **Position decides which kinds of name are legal**, and a name's kind is fixed at load time. A dimension is never legal in a value position: it is a coordinate space, not data. | [§5.1](#51-name-resolution) |
| 5 | **Dim sets compose by union.** A constraint must *equal* its `foreach`; a `where` or a bound must not *exceed* its frame. | [§5.2](#52-dim-algebra) |
| 6 | **Absence is a property of variables.** Four constructs create it; nothing else does. | [§6](#6-absence) |
| 7 | Through arithmetic absence **spreads, taking the row with it**. Out of a reduction it does not — so a reduction does not distribute over `+`, and `sum(x + y)` and `sum(x) + sum(y)` are different questions. | [§6](#6-absence) |
| 8 | **Identity of the position.** A missing value reads as whatever makes it contribute nothing — zero as a coefficient, the identity of a sum; false in a `where`, where the coordinate then does not exist. Where no such reading exists it is refused: a divisor, a bound. `shift(…, edge=)` is the one place a value may be *asked* for, and it takes the identity of its position too. | [§6](#6-absence), [§7](#7-operators) |
| 9 | **Degree 1, always**: `*` needs a variable-free factor, `/` a variable-free divisor, `**` is refused. Bounds are narrower still — a name or a number, never arithmetic. | [§5](#5-expressions), [§2](#2-declarations) |
| 10 | **The operator set is closed.** Compositions go in `macros:`. | [§7](#7-operators) |

## 1. File shape

Nine top-level keys: `dimensions`, `parameters`, `variables`, `constraints`,
`objectives` (§2), `expressions`, `macros` (§3), `piecewise` (§4), `sos` (§4.1),
plus `version` (below). The schema accepts any subset, but `check`, `solve` and
`write` require an objective — there is nothing to optimise without one.

**`version` declares which surface the file is written against.** It is
optional, and absent means `0`:

<!-- doctest: skip -->
```yaml
version: 0
dimensions: ...
```

**`0` means unstable, and that is the promise being made.** The surface may
change in any release; there is no compatibility guarantee, and saying so in
the file is more honest than silence. What `1` is stays deliberately undecided
— the only thing decided is that `0` does not become `1` without a changelog
entry naming what moved.

**A version this reader does not know is a load error, and nothing else.** The
field gates no behaviour: it never selects an alternative surface, because
keeping two alive in one codebase is a large permanent cost against an error
that costs one line. A file from the future is refused rather than
misinterpreted, which is the whole reason to carry the field:

```
model declares version 1, and lpspec 0.0.1a75 understands [0].
Upgrade lpspec, or write the version this file actually targets.
```

It is a **language** version, not a package one — it moves when the accepted
YAML surface moves, which most releases do not.

**The schema is closed at every level.** An unrecognised key — top-level or
inside any declaration — is a load error naming the near miss (`unknown key
'boundz' … Did you mean 'bounds'?`). Ignoring it would let a typo change the
model: a dropped `bounds:` leaves a variable unbounded, a dropped `where:`
leaves it unmasked.

**Reading rules.** Booleans are YAML 1.2 (`true`/`false` only), everything else
1.1 — under 1.1 `on`/`off`/`yes`/`no`/`y`/`n` become booleans and silently
destroy dimension labels that are country codes, so `values: [no, se, on]` is
three labels here. The implicit timestamp (`2024-01-01`) and sexagesimal ints
(`12:30` → `750`) deliberately survive; the `dtype` guard below catches them
wherever they were not meant. A duplicate key is a load error naming both lines. `<<:` merge keys are
honoured, and a key the mapping declares itself overrides the merged value. The
document must be a mapping.

## 2. Declarations

**An empty dim list is the empty coordinate, everywhere it appears** — one value
for a parameter's `dims: []`, one column for a variable's `foreach: []`, one row
for a constraint's. That is the ordinary reading of a product over nothing, not
a special case, so a dummy dimension of size 1 is never how a scalar is written.
One gap: a scalar **variable** may not carry a `where`
([#340](https://github.com/fluxopt/lpspec/issues/340)) — put the condition on
the constraints that use it.

**`dimensions`** — the master coordinate index. Every dimension named anywhere
must be declared. `dtype` ∈ {`float`, `int`, `str`, `datetime`}, default `str`.
`values` is a list or null; if null, coordinates must arrive from data (§8),
else loading fails. Every declared value must
be of the declared `dtype` — `values: [2024-01-01]` under the default
`dtype: str` is a load error, because YAML resolved it to a date and a date
does not join `'2024-01-01'` in the data.

**A dimension and a coordinate are different things, and the file keeps them
apart.** A **dimension** is an axis of the model: something is indexed by it
(`dims`, `foreach`) or an aggregation lands terms on it (`group_by=`, `at()`).
A **coordinate** is a label a dimension's members carry — a generator's bus, a
snapshot's period — structure, never data: it is not legal in a value position,
and a *value* riding a dimension is what a parameter is. The block invariant
follows: **everything under `dimensions:` is an axis.** If `b` is single-valued
per `a`, `b` is a coordinate of `a`, not a dimension — a `foreach` product over
functionally dependent dims cut back by a mask is the shape `coords` exists to
replace. `check` warns about a declared dimension that is never an axis.

`coords` declares the labels, and the shape of each entry's value says which of
two kinds it is:

- **A string names a target dimension — the groupable kind.** The coordinate's
  values are labels of that dimension, which is what `sum(group_by=)` and
  `at()` land terms on. Written as a list when the two names coincide, or as a
  mapping when they do not:

  ```yaml
  dimensions:
    bus: {dtype: str}
    generator:
      coords: [bus]  # same as {bus: bus}
    line:
      coords: {from: bus, to: bus}  # two coordinates onto one dimension
  ```

  The target must be a declared dimension, must not be the dimension carrying
  the coordinate, and a coordinate must not be named after a *different*
  dimension. Non-null values are checked against the target once data is bound
  (§8) — the check that makes `sum(group_by=)` safe. A **partial** coordinate
  is legal: null says the label belongs to no group (a generator on no bus, a
  line with one open end) and `sum(group_by=)` places its terms nowhere, while
  an unknown *non-null* value is a typo and an error.

- **A mapping declares an inline label space — the selection-only kind.** It
  owns its values, targets nothing, and puts no entry under `dimensions:`,
  because a label space nothing aggregates into is not part of the model's
  dimensionality:

  ```yaml
  dimensions:
    snapshot:
      dtype: int
      coords:
        period: {dtype: int}  # a label on snapshot — nothing else
  ```

  An inline coordinate's name joins the flat namespace (law 3). Grouping into
  one is refused with the rewrite: declare the axis and target it
  (`period: {...}` under `dimensions:`, `coords: {period: period}` on
  `snapshot`) — a one-word promotion, made the day the model genuinely gains
  the axis.

Either kind is single-valued per label, and a dimension declaring `coords`
needs an index source carrying those columns; they are never inferred from the
parameters that use the dimension, since inferring would let a mistyped label
extend the label space instead of being rejected.

**`parameters`** — declared shape only; data binds by name at run time (§8).
`dims` required (`[]` is a scalar); `dtype` ∈ {`float`, `int`, `bool`, `str`},
default `float`.

**`variables`**

| Field | Type | Default |
|---|---|---|
| `foreach` | list[str] | required — dim signature, one variable per coordinate |
| `where` | str or null | `null` — §6; variables exist only where true |
| `bounds.lower` / `.upper` | number or parameter name | `-inf` / `inf` |
| `domain` | str | `continuous`; or `integer`, `binary` — which carries fixed 0/1 bounds — or `semi_continuous` |

Omitting a bound means unbounded on that side — non-negativity is written, not
assumed. Bounds are
a *narrower* language than expressions (a name or a number, never arithmetic) and
the error says so rather than reporting a parse failure; expressions there are
[#31](https://github.com/fluxopt/lpspec/issues/31). A bound parameter's dims must
not exceed `foreach`.

**`domain: semi_continuous`** makes the variable *zero, or between its declared
bounds* — nothing in between: the unit-commitment shape, said without the
auxiliary binary and big-M pair it otherwise costs. Its lower bound must be a
positive, finite number — zero or absent is ordinary continuous, and a
parameter there is not supported yet, both refused at load. A semi-integer
domain is deliberately not exposed
([#383](https://github.com/fluxopt/lpspec/issues/383)).

**Equal bounds pin a variable**, which is how one declaration covers a quantity
that is a decision in one model and data in another: bind `lower` and `upper` to
the same value where it is fixed, and `rate - relmax * size <= 0` is one equation
whether `size` is chosen or given. Presolve substitutes the pinned column, so the
solver receives the LP the pre-multiplied form would have produced. Two limits: a
pinned variable is still a variable, so `size * on` is refused as variable ×
variable (§5), and it cannot appear in another variable's `bounds`.

**`constraints`** — **one rule per block**: `foreach` (required), an optional
`where`, and one `expression` carrying exactly one of `<=`, `>=`, `==`. The
block's name *is* the constraint's name, which is what a row is read back by.
The LHS must involve at least one decision variable.

`foreach: []` is **one scalar row** — a single system-wide budget, where the
expression reduces every dim away. Nothing special: law 5 requires `dims(lhs) ∪
dims(rhs)` to *equal* `foreach`, and `sum(x, over=f) <= 120` has no free dims,
so `[]` is the signature that satisfies it.

Two regimes of one rule are two blocks, and each gets a name a reader chose
rather than a position in a list:

<!-- doctest: wrap=constraints -->
```yaml
storage_balance:
  foreach: [snapshot, storage]
  expression: soc == shift(soc, over=snapshot, by=1) * (1 - loss) + charge - discharge

storage_balance_initial:
  foreach: [snapshot, storage]
  where: "snapshot == 0"
  expression: soc == soc_initial
```

`shift` vacates the first snapshot and a vacated position is absent (§7), so that
row drops without a `where` saying so. Spelling it `edge='wrap'` gated on
`where: "snapshot > 0"` builds the same rows here and a *different* model on a
horizon not starting at 0 — the gate hardcodes the origin, the operator does
not.

**`objectives`** — one `expression`, like a constraint; `sense` ∈ {`minimize`,
`maximize`}, default `minimize`; no `foreach`, since an objective is scalar by
definition. **Every dim the expression carries is summed**, each *term* over the
dims **that term** carries and not repeated because another term carries a dim it
does not: in `x * a + y * b` with `x, a` on `i` and `y, b` on `j` there are
`|i| + |j|` summands, never `|i| · |j|`. Declaring a second objective is a load
error.

## 3. `expressions` and `macros`

Pure AST substitution: they are expanded away before anything consumes the
model, so they cost nothing at build time. A named expression is a macro with no
formals.

```yaml
dimensions:
  generator:
    dtype: str
variables:
  p:
    foreach: [generator]

expressions:
  total_generation: sum(p, over=generator)
macros:
  weighted_sum:
    args: [array, weights]  # positional formals, default []
    kwargs: [over]  # keyword formals, default []
    template: sum(array * weights, over=over)
```

Both hold arithmetic (no comparison). Arguments expand before substitution
(call-by-value), so they may themselves use macros and named expressions.
Formals shadow model names inside a template but may not collide with a declared
**dimension**. Arity is checked per call site; cycles are reported with the
reference chain. Templates are schema-local, so every one is parsed and
name-checked at load time even if never called.

## 4. `piecewise`

N expressions jointly pinned to a breakpoint-indexed piecewise-linear curve.

<!-- doctest: wrap=piecewise -->
```yaml
chp:
  over: bp  # breakpoint dimension
  links:
    - [power, power_bp]  # [expression, values-parameter]
    - [fuel, fuel_bp]
    - [heat, heat_bp]
  method: adjacency  # how the weights are restricted — below
  active: null  # optional gating expression: formulation pinned to 0

# a two-link block may bound one side instead of pinning it
fuel_cap:
  over: bp
  links:
    - [power, power_bp]
    - [fuel, fuel_bp, "<="]
```

*expression* is any affine expression (a bare variable name being the simplest);
*values* names a parameter carrying the `over` dim, so curves may vary along
other dims (per-generator, say); *sign* (`<=`/`>=`, at most one, only with
exactly two links) bounds the link instead of pinning it. Blocks expand **before
building** into plain variables and constraints via λ convex-combination —
weights in `[0,1]` with a convexity row, and one link row per tuple.

**`method` is the one thing that varies**, and it varies in exactly one place:
how the weights are restricted, once they exist.

| `method` | what it adds | |
|---|---|---|
| `adjacency` *(default)* | a binary per segment, and `lam <= seg + shift(seg, over=bp, by=1, edge=0)` | the curve, built |
| `sos2` | a `sos:` block (§4.1) over the same weights | the curve, *said* — a sink that branches on a set does |
| `convex` | nothing | the hull, which is a pure LP |

`adjacency` and `sos2` state the same restriction and reach the same optimum;
they differ in what the sink is handed, so which is faster is a property of the
solver and not of the model. `convex` is a different model — exact only for a
curve of matching curvature under optimisation pressure, which is checked
against the breakpoint *values* at bind time — and it takes exactly two links
and no `active`. `method: lp`, linopy's tangent-line formulation, is
[#695](https://github.com/fluxopt/lpspec/issues/695) and not here.

### 4.1 `sos`

A **special-ordered set**: one dimension of one variable, and how many of that
family may be nonzero at once.

<!-- doctest: wrap=sos -->
```yaml
pick_one_size:
  variable: build  # the variable the set is over
  over: size  # the dim it runs along — one set per coordinate of the rest
  type: 1  # 1: at most one nonzero; 2: at most two, and consecutive
  big_m: 500  # optional, and only a reformulating sink reads it
```

`type: 1` is a choice — at most one member of the family is nonzero. `type: 2`
is an interpolation — at most two, and those two **consecutive**, which is what
makes it the native spelling of a piecewise-linear curve (§4 emits the same
restriction as binaries and an adjacency row).

**A set is over one variable, and a variable holds one set.** A second block
naming the same variable is a load error: what an SOS *is* is a property of the
variable, which is the shape every sink and the eager lane take it in.

**Membership is the variable's own.** Its `where` decides which coordinates
exist, so a masked-out member is not in the set — and for `type: 2`,
consecutive means consecutive *among the members present*, leaving no hole
where a coordinate was masked away. **Order is the `over` dimension's declared
order** — what `shift` walks (§8) — so reordering the set means reordering that
index, and there is no per-set weight to supply.

**A set is a *sink capability*, not a language question**, and it is the one
construct whose sink shows: where a sink has no SOS concept it is handed
binaries and big-M rows instead (which sink does what, and why the sink family
decides it, is [the sink README](https://github.com/fluxopt/lpspec/blob/main/src/lpspec/relational/sinks/README.md)).
Two consequences reach the model, so neither is silent:

- that rewrite is **mixed-integer**, so a set on an otherwise continuous model
  gives up its duals there;
- `M` has to be finite, so every member needs `bounds.upper` or a `big_m:`, and
  a negative `bounds.lower` is refused. `big_m` caps a loose bound — the
  *tighter* of the two is used, and tighter is a better relaxation.

Both are conditions of the *rewrite*, so a model that fails them still solves
on a sink that takes the set, and the message says so.

## 5. Expressions

```text
expression  ::= arithmetic | arithmetic COMPARATOR arithmetic
arithmetic  ::= atom | unary_op arithmetic | arithmetic binary_op arithmetic
             |  function_call | "(" arithmetic ")"
atom        ::= NUMBER | NAME
unary_op    ::= "+" | "-"       binary_op ::= "+" | "-" | "*" | "/" | "**"
COMPARATOR  ::= "<=" | ">=" | "=="
function_call ::= NAME "(" [pos_arg ("," pos_arg)*] ["," kwarg ("," kwarg)*] ")"
kwarg       ::= NAME "=" (arithmetic | NAME)
NAME        ::= [a-zA-Z][a-zA-Z0-9_]*
NUMBER      ::= integer | float | "inf" | ".inf"
```

Precedence, highest first: `**`, then `*` `/`, then binary `+` `-`, then unary
`+` `-`; parentheses override. Affinity is enforced — `*` needs at least one
variable-free factor, `/` a variable-free divisor that is a single factor rather
than a sum. **`**` parses but is not in the language**: it is rejected at load time, so the
refusal can name the operator and its rewrite. A variable base
breaks degree 1; over parameters alone it is data prep.

### 5.1 Name resolution

A **load-time pass** (`resolution.py`), not an evaluation-time lookup: parsers
emit `NameNode` tokens, the pass rewrites each into `VariableNode`, `ParameterNode`
or `DimensionNode`, so no
unresolved name crosses into a backend and no backend can hold its own opinion
about what a name means.

**One flat namespace** covers dimensions, parameters, variables, named
expressions, macros and built-in operators; a collision is a load error naming
both declarations. Ordered resolution with shadowing is wrong for a fail-loud
language: under it, declaring a parameter named `snapshot` would silently change
what an existing `where: "snapshot > 0"` means.

**Constraints and objectives are outside it**, no position resolving to one, so
a model may name a constraint after a variable — `pypsa_unit_commitment` names
both `start_up`. What reads a solve back keys on the label space as well as the
name for that reason.

| Position | Legal kinds |
|---|---|
| expression (`p * cost`) | variable, parameter |
| dimension argument (`over=`, `into=`) | dimension |
| where string | parameter, dimension |
| `bounds.lower` / `.upper` | parameter name, or a number |
| `shift(x, over=d, by=n, edge=0)` — the `edge` key | `'wrap'` **quoted**, or a bare number; never a dimension. A bare word in a kwarg value is a *name to resolve*, and `wrap` is a literal — the same rule §6.1 uses for a `where`, so `over=wrap, edge='wrap'` reads unambiguously even where a dimension is called `wrap` |

`edge` is the one keyword whose *key* is fixed rather than naming a dimension,
so a dimension called `edge` does not change what it means; the position takes
`wrap` or a number and nothing else.

A dimension in a value position is an error — it is a coordinate space, not
data. To use its coordinates as data, declare a parameter over it.

### 5.2 Dim algebra

Parameter `dims` and variable `foreach` are declared and dimension arguments are
name-checked, so **every node's dim set is computable before any data is bound**.
`dimensions.py` computes it at load time on the resolved AST.

| Node | Dim set | Error |
|---|---|---|
| number | `{}` | |
| parameter / variable | its `dims` / its `foreach` | |
| `-x`, `+x` | `dims(x)` | |
| `a + b`, `a * b`, `a / b` | `dims(a) ∪ dims(b)` | |
| `sum(x, over=d)` | `dims(x) − {d}` | if `d ∉ dims(x)` |
| `sum(x, over=d, group_by=c)` | `(dims(x) − {d}) ∪ {target(c)}` | unless `d ∈ dims(x)`, or `d` declares no coordinate `c` |
| `at(x, onto=d, by=c)` | `(dims(x) − {target(c)}) ∪ {d}` | unless `target(c) ∈ dims(x)`, or `d` declares no coordinate `c`, or `d ∈ dims(x)` already |
| `shift(x, over=d, by=n)` | `dims(x)` | if `d ∉ dims(x)` |

Binary operators **union**: an outer product is legitimate when the frame
declares the result. What must not be silent is the declaration disagreeing —
so a **constraint** requires `dims(lhs) ∪ dims(rhs)` to *equal* `foreach` (a
stray dim multiplies rows and an unused `foreach` dim repeats one row across
them, either way building a different model than the file reads as), while a
**where** predicate's dims and a **bound** parameter's dims must not exceed the
frame.

## 6. Absence

A coordinate where a **variable does not exist** — not a value and not a zero,
but a state the language tracks (law 6).

| construct | what is absent |
|---|---|
| `where:` on a variable | the variable, at the masked coordinates |
| `where:` on a constraint | the row |
| `shift(x, over=d, by=n)` with no `edge=` | the vacated edge coordinate (§7) |
| a null value in a dimension's `coords:` | that label's group membership (§2) |

**A sparse parameter table is not one of them.** Missing rows are compressed
encoding, and law 8 says what one reads as: the reading under which the missing
thing contributes nothing — or a refusal, where no such reading exists.

| position | a missing parameter row | why that reading |
|---|---|---|
| coefficient — `w * x` | zero: the term does not participate, the row survives | `0` is the identity of a sum, so the term contributes nothing |
| `where` operand | false | a coordinate whose data is missing is not one the model can claim exists |
| divisor — `x / d` | **refused** at bind *where the model divides by it* | nothing contributes nothing: `0` divides by zero, `1` rescales, dropping rewrites the constraint |
| a comparison's whole constant side — `x <= cap` | **refused** at bind *where the row is built* | nothing contributes nothing: the fill would *be* the bound, so `x <= 0` binds where the model said nothing |
| `bounds:` | an error | nothing contributes nothing: unbounded is not bounded-at-zero |

The refusals are keyed to **the rows a declaration builds**, never to the
coordinate product: a coordinate a `where` already removed asks no question, so
supplying data only where the model uses it stays the ordinary idiom. That is
also what makes masking a real remedy rather than a workaround:

<!-- doctest: wrap=constraints -->
```yaml
c:
  foreach: [g]
  where: cap  # no row where `cap` has none, instead of a row reading `x <= 0`
  expression: x <= cap
```

Three answers, and the language does not pick between them: **supply the rows**
if the value is what was meant, **mask them out** if the row should not exist,
or **drop the declaration** if the model has no such quantity — which is what a
framework emitting a dict does (#217).

### A row with no variable terms is not built

Whatever emptied it. A masked variable takes the row with it (law 7), a
reduction over an absent set contributes `0`, and a missing parameter row is a
zero coefficient (law 8) — three ways to reach one shape, *a row asserting
something about constants only*, and the shape decides, not the provenance.
Such a row constrains nothing a solver can act on.

It is **reported**, never silent: `diagnostics().omissions` on a built model gives
`(constraint, rows_not_built)`, empty for a model whose every declared row was
built. A declared constraint that goes unenforced is a thing the caller has to
be able to see — which is a reason to `build` a model you mean to inspect
rather than to `solve` it, an answer being the one thing that cannot report it.

### How absence travels

**Through arithmetic it spreads** (law 7), taking the row with it: `x + y >= 10`
is *no constraint* where `y` is masked, not `x >= 10`. Its asymmetry with the
table above is the whole hazard, in one example: `x - rel_max * size <= 0`
**loses the row** where the *variable* `size` is masked, and **keeps** it as
`x <= 0` where the *parameter* `rel_max` has no row — feasible, plausible, no
error. A missing correction term tightens in the safe direction and is a
legitimate idiom; a missing coefficient that *is* the bound rewrites what the
constraint says.

**Out of a reduction it does not** — `sum(x, over=d)` is defined when only some
of `d` exists, or one masked component would delete a system-wide accounting
row. So the two spellings below are different questions:

| spelling | sums over | with `y` absent at `f=b` |
|---|---|---|
| `sum(x + y, over=f)` | where the **summand** exists | `x[a] + y[a]` — `x[b]` goes with the absent `y[b]` |
| `sum(x, over=f) + sum(y, over=f)` | each operand over **its own** domain | `x[a] + x[b] + y[a]` |

*The total of the net where the net is defined*, against *the total in minus the
total out*. Rewriting one into the other reads the absent `y[b]` as a zero.

### Asking for the other reading

Each rule has a spelling for the opposite intent:

| you want | you write |
|---|---|
| the row kept, the missing term read as zero | two constraints under complementary `where` clauses |
| a vacated shift position to contribute | `shift(x, over=d, by=n, edge=0)` — the identity of *its* position (§7) |
| to test whether a variable exists here | its bare name in a `where` |
| a sparse coefficient to remove the row rather than zero the term | mask on it — `where: "rel_max"` |
| to divide by a parameter you only have some of | mask the row or the variable — `where: "d"`. The divisor is required where the division survives, not everywhere it is indexed |
| a bound only where the data has one | supply the missing value (`inf` is a value), or mask the variable — the two build **different models**, so neither is inferred |

**Only one of those is a fill** (law 8): the coordinate `shift` vacates is
*created by the operator*, so there is no row a caller could have supplied.
Everywhere else the value is expressible in the data, and §11 keeps it there.



### 6.1 Where strings

A boolean mask; true means "this coordinate exists". Semantics are **row
absence**, not zero-fill: a masked-out variable is not created, a masked-out
constraint row is not built.

```text
where_expr ::= atom | "NOT" where_expr | where_expr ("AND"|"OR") where_expr
            |  "(" where_expr ")"
atom       ::= NAME | NAME COMPARATOR value | "True" | "False"
COMPARATOR ::= "<=" | ">=" | "==" | "!=" | "<" | ">"
value      ::= NUMBER | QUOTED | NAME_OR_STRING
QUOTED     ::= "'" chars "'" | '"' chars '"'
```

| Surface | Names a… | Meaning |
|---|---|---|
| `name` (bare) | parameter | defined: non-null **and** finite |
| `name` (bare) | variable | defined: the variable exists at this coordinate. The counterpart of the parameter row, and the way to say which coordinates the row-dropping rule above applies to |
| `name` (bare) | dimension | load error — true everywhere, so it reads as a condition and is not one; compare it instead |
| `name OP value` | parameter | element-wise, NaN → False. RHS is a literal number, or a bare name read as a string coordinate — a name that is *declared* is a load error instead (below) |
| `name OP value` | dimension | filter on the frame's own coordinate column |
| `AND` `OR` `NOT` | — | case-insensitive; `NOT` > `AND` > `OR` |
| `True` / `False` | — | literals; `True` ≡ no `where` |

Comparing two parameters is not in the language — precompute a boolean parameter
in data prep — and neither is comparing two dimensions. The string reading of an
RHS name is for names the model does *not* declare, which is how a string
coordinate is compared; a **declared** name on the RHS (parameter, variable or
dimension) is a load error naming the near miss, because reading it as text
would compare a coordinate column against another declaration's name and mask
everything out. An undeclared *bare* name is a load error, and a mask dim outside `foreach` is
one too (§5.2).

**Quote a label that is not an identifier**, and quote a date. A bare RHS word
has to look like a name, so `combined-cycle`, `IT-north` and `CCGT 400MW` are
only sayable in quotes — and quoting is also what says *label, not name*, so a
quoted word is never read as a declaration and never a near-miss error.

**A comparison is checked against the declared `dtype`**, and this is the one
place it matters most: a `datetime` dimension compared to a number is compared
against the **epoch**, so `snapshot > 0` would silently mean "after 1970-01-01".
That is a load error naming the fix. A datetime boundary is a quoted ISO date —
`snapshot > '2030-01-01'`, or `'2030-01-01T06:00'` with a time — which is the
only spelling, since the language orders and compares coordinates and never
interprets them. Calendar arithmetic, resampling and timezone conversion stay
data prep.

**String labels order bytewise**, whatever order the dimension declared them
in. Declaration order is a different axis — it is what `shift` walks and what
a label follows — and a `where` never reads it: `node >= 'b'` means the same
thing however the nodes were listed. A label the dimension does not carry
compares equal to nothing, so the mask is false there rather than an error —
quoting already said *label, not name*, and a label is data.

## 7. Operators

The built-in set is **closed** — no Python registry, so the operators are
exactly these and a model cannot depend on what a caller registered. Dimension
arguments are name-checked at load time:
`sum(p, over=snapshto)` is an error, not a no-op.

| Operator | Result | Notes |
|---|---|---|
| `sum(array, over=dim)` | `dim` collapses | `array` must carry `dim` |
| `sum(array, over=dim, group_by=coord)` | `over` → the dimension `coord` targets | `coord` is declared on `over` (§2); its values are the group labels, checked against the target dimension at bind time. The membership sum that makes topology data rather than structure; groups with no members contribute nothing |
| `at(array, onto=dim, by=coord)` | the dimension `coord` targets → `over` | **The adjoint of `sum(group_by=)`, and deliberately the same two arguments**: `(over, by)` names one mapping table and the helper says which way it is walked. `sum(group_by=)` consumes `over` and produces the target; `at` consumes the target and produces `over`, reading one coarse value once per fine label pointing at it. Reads a *variable* as readily as a parameter, which is what a per-component decision gating its flows needs. A fine label whose coordinate is null reads nothing and its row is absent, matching `sum(group_by=)`'s null group |
| `shift(array, over=dim, by=n)` | value at *t−n* | vacated positions are **absent**: they propagate and drop the row (§6) |
| `shift(array, over=dim, by=n, edge='wrap')` | value at *t−n*, cyclic | coordinates fixed, values wrap; nothing is vacated |
| `shift(array, over=dim, by=n, edge=v)` | value at *t−n* | vacated positions contribute the number **`v`** instead, and the row survives (`0` for a sum, `1` for a product) |

`array` is any node of the right dim set, so `shift` re-indexes a **parameter**
as readily as a variable: `shift(dt, over=t, by=1, edge=0)` is the previous
snapshot's duration, without shipping a pre-shifted copy of a table the model
already has.

Four rules govern `edge=`, and all four are law 8 in this position:

- **Bare** — the vacated coordinate is absent in exactly §6's sense, so an
  acyclic recurrence has no row at its first coordinate rather than a row
  asserting the quantity starts at zero. An initial condition is then something
  the model states, under a complementary `where`.
- **Numeric** — asks for a value back, and it is a number rather than a flag
  because the identity is positional: `0` for a sum, `1` for a product. The
  library cannot see which position it is in and the model can.
- **Over a variable, the only representable numeric edge is `0`** — a vacated
  slot there contributes no term at all, and a nonzero one would be a constant
  standing where a term was.
- **A bare `shift` over a variable-free expression is a load error.** A
  parameter's missing row is a zero coefficient (§6), so there is no absence for
  the vacated slot to carry, and inventing one silently turns
  `x <= shift(dt, over=t, by=1)` into `x <= 0`. The error names what it could
  have meant: `edge='wrap'`, `edge=0`, or `edge=0` **together with** a `where`
  excluding the vacated coordinate — which is how an acyclic recurrence omits
  its first row. The two are a pair, not a choice: a `where` alone does not
  lift the refusal, since it is decided on the expression before any mask is
  read, and `edge=0` alone leaves a row at that coordinate whose bound is the
  zero.

Anything composable out of these belongs in `macros:`. Math that is not sayable
at all goes to a declared `escape:` island
([#38](https://github.com/fluxopt/lpspec/issues/38)): named in the file,
bounded by the preceding `where` mask, terminal (it yields a constraint, never a
sub-expression), and billed against a label budget before any Python runs.

## 8. Data binding

**Master coordinates** are resolved per dimension before any parameter loads,
highest precedence first:

1. a key in `sources` — a table carrying a column of that name, or a parquet
   path; first occurrence of each value is its position
2. `coords=` — anything `pd.Index()` accepts, or a table carrying the label
   column plus one column per declared coordinate (§2)
3. `values:` in the YAML
4. derived from the parameter tables that carry the dim, as **sorted** distinct
   values

Step 4 is unavailable to a dimension declaring `coords`: it reads index columns
only, so it cannot supply a coordinate. Otherwise it exists because a dim some
parameter already spans needs no second declaration — but it costs the *declared
order*, which `shift` reads positionally, so pass an explicit index whenever
order matters. A dim that no source names and no parameter carries raises.

**Accepted per parameter** (declared `dims: [d1, d2]`): a parquet path; any
table exposing the Arrow PyCapsule protocol with columns `d1, d2, value`;
`int`/`float` for a 0-D parameter. `pd.Series` and `xr.DataArray` keep their
dims in an *index* rather than in columns, so they are unwrapped first — but
only if that library is already imported, never by importing it. An unnamed
index binds positionally to the declared `dims`; a named one binds by name in
any order, and a name outside the declared dims raises rather than being
overwritten.

The opt-in linopy shim accepts the same *language* but a different set of data
inputs, and has no step 4 —
[docs/design/linopy.md](design/linopy.md#the-same-language-different-data-inputs).

Coordinate values in the data must be a subset of the master coordinate; values
outside it raise rather than being dropped silently. Every declared parameter
must be provided, and every provided key must be declared — the YAML is the
source of truth. Validation order: dimension coords → parameter presence → dim
names → coordinate values → unknown keys.

The loader deliberately does **not** check that values are sensible, that a
parameter is used, or that coordinates *cover* the master index. Missing
coordinates produce no rows — sparse data gives sparse variables.

## 9. Errors

**Fail at load time, not at evaluation time.** Anything detectable before
building is detected before building; the worst error is an opaque xarray or
solver exception with no pointer back to a YAML declaration. Every message names
what went wrong, what to do about it, and where it helps, the valid options:

```text
Constraint 'balance', equation 0: 'p_charge' not found.
  Variables: ['p', 'soc']
  Parameters: ['p_max', 'load', 'efficiency']
Check for typos, or ensure 'p_charge' is declared.
```

A construct outside the language names the construct and its rewrite, never a
silent fallback.

## 10. Python API

How to *run* a model is [docs/api.md](api.md) — five verbs, the result readers
and the linopy shim. It is a separate page because it is not part of the
language: nothing there changes what a file means.

## 11. Out of scope

| Not here | Instead |
|---|---|
| time-series processing (resample, cluster, interpolate, align), file IO, units | data prep; pass a parameter |
| solver breadth | two solver sinks — HiGHS, which ships, and Gurobi via the `[gurobi]` extra — chosen with `solver_name` at the call, never in the file; LP files for everything else ([#106](https://github.com/fluxopt/lpspec/issues/106)) |
| indicator constraints | planned, as a *sink capability* rather than a language question — the same axis `sos:` (§4.1) landed on, and the same split: `lp_file` and Gurobi have the concept, the default solver does not ([#220](https://github.com/fluxopt/lpspec/issues/220), [Track 3](https://github.com/fluxopt/lpspec/issues/472)) |
| multi-objective | one objective — declaring a second is a load error (§2); weight them into one expression |
| schema migrations | — |
| arbitrary array ops (`merge`, `reindex`, `apply_ufunc`) | data prep, or a declared `escape:` island — the closed AST is what makes streaming possible |
| filling a missing value (`.fillna`) | data prep, or a `where` if you meant the coordinate not to exist. In the language only where the data cannot reach — `shift(..., edge=)`, §6 |

[Calliope](design/prior-art.md)'s math language — which this surface is derived
from — is a corpus we score coverage against, not a
specification we match; file portability is not a goal, and neither is
operation parity with xarray/pandas. A model built partly in Python has no
readable `.yaml` representation and will not get one: the *math* side is
feasible, but expression and where strings come back as anonymous arrays, so the
round-trip is functional and not reviewable — which is the whole point of the
file. Whether Python may *emit* declarations at all is a separate and open
question ([#381](https://github.com/fluxopt/lpspec/issues/381)).

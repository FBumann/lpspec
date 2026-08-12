"""Build a `Program` into `ModelTables` through duckdb.

The duckdb twin of `relational/executor.py`, and a **drop-in at the sink seam**:
it hands back the same `sinks.ModelTables` the polars executor does, so
`lp_file`, `solver_direct`, the status codes and the result readers are
untouched and unaware. That is what makes the two comparable — the only thing
that differs between a `PolarsExecutor` build and a `DuckExecutor` build is
which engine filled the four frames.

Scope: the affine core — variables with bounds and masks, constraints over
sum/group_sum/translate, one objective. Enough to build every model in
`bench/models/` and diff the result against polars, which is what pricing the
port is for. Not the whole language: piecewise expansion happens above this
layer anyway, and duals/solution read-back are the polars executor's business
since they are joins against label frames rather than engine work.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Mapping
from functools import reduce
from typing import TYPE_CHECKING, Any

import duckdb
import polars as pl
from duckdb import CoalesceOperator, ConstantExpression, Expression, FunctionExpression, SQLExpression

from lpspec.errors import DataError, LanguageError, null_bounds_message
from lpspec.relational import plan, sinks
from lpspec.relational.binding import BoundSources, bind
from lpspec.relational.engine import Engine
from lpspec.relational.engines.duck.compiler import (
    UNIT,
    DuckCompiler,
    Relation,
    TermFragment,
    _ordinal,
    col,
    matching,
    q,
    restrict_to,
    union_all,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

#: The four frames a sink reads, and their dtypes — both stated by
#: `sinks/tables.py`, which is what reads them. Shared with the polars engine
#: for the reason the frames are: a sink cannot see which engine filled them.
_COLS, _OBJ, _ROWS, _MATRIX = sinks.COLS, sinks.OBJ, sinks.ROWS, sinks.MATRIX


def _ranked(order: Sequence[str], offset: int) -> Expression:
    """A row's position in *order*, counted from *offset*.

    ``ROW_NUMBER`` is the one construct here with no expression-API form —
    `Expression` has no `over`, and `DuckDBPyRelation.row_number` takes its
    window spec and its projection as SQL anyway. So the window is written out
    and the arithmetic around it is not: the names still go through :func:`q`,
    and the offset is a number rather than an identifier.
    """
    by = ', '.join(q(c) for c in order) or '1'
    return (SQLExpression(f'ROW_NUMBER() OVER (ORDER BY {by})') + ConstantExpression(offset - 1)).cast('BIGINT')


class _Labels(Mapping[str, 'pl.LazyFrame']):
    """Label relations, fetched out of duckdb only if a read-back asks.

    `Engine` reads a solution back by joining the solver's answer onto
    `(dims…, label)` frames, and states those as polars. Materialising every
    one at build time would put a second copy of the labels in this process —
    which is most of what choosing this engine was for. So the name is held
    and the frame is fetched on the first access, which for a caller that only
    writes an LP file never happens.
    """

    def __init__(self, con: duckdb.DuckDBPyConnection, tables: dict[str, str], label: str) -> None:
        self._con = con
        self._tables = tables
        self._label = label
        self._frames: dict[str, pl.LazyFrame] = {}

    def __getitem__(self, name: str) -> pl.LazyFrame:
        """The frame, **in label order and in binding's dtypes** — read, not imposed.

        `Engine._read_back` stopped sorting once every labelling path produced
        an ordered frame. polars' paths do and verify it; a duckdb relation
        promises no order at all, and two of the three here are views over a
        cross join. So the order is asked for on the way out, where it is paid
        only by a caller that reads a solution back.

        The dtypes are binding's own: a string dimension crosses into duckdb as
        ``VARCHAR`` and comes back as ``String``, which is what
        `Engine._read_back` hands a caller from either engine (#541, #593).
        """
        if name not in self._frames:
            self._frames[name] = self._con.table(self._tables[name]).order(q(self._label)).pl().lazy()
        return self._frames[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._tables)

    def __len__(self) -> int:
        return len(self._tables)

    def clear(self) -> None:
        """Drop the cached frames; the relations go with the connection."""
        self._frames.clear()


class DuckExecutor(Engine):
    """The duckdb engine: plan → relations → `sinks.ModelTables`.

    Everything past the four frames — both sinks, the solution read-back, the
    context manager — comes from `Engine` and is shared with the polars
    executor, because none of it is engine work.
    """

    _cols: pl.DataFrame | None
    _obj: pl.DataFrame | None
    _rows: pl.DataFrame | None
    _matrix: pl.DataFrame | None
    _matrix_starts: Any
    _con: duckdb.DuckDBPyConnection

    def __init__(self) -> None:
        self._con = duckdb.connect()
        self._bound: BoundSources | None = None
        self._compiler: DuckCompiler | None = None
        self._program: plan.Program | None = None
        self._var_tables: dict[str, str] = {}
        self._row_tables: dict[str, str] = {}
        #: the dims each declared name is read through — what `plan.free_prefix`
        #: needs to know which label path a mask allows
        self._name_dims: dict[str, tuple[str, ...]] = {}
        self._var_labels = _Labels(self._con, self._var_tables, 'var_label')
        self._row_labels = _Labels(self._con, self._row_tables, 'row')
        #: the contiguous run of labels each declaration was given — what
        #: `Engine._read_back` slices a solver vector by, on both engines
        self._blocks: dict[str, tuple[int, int]] = {}
        self._omitted: dict[str, int] = {}
        self._matrix_starts: Any = None
        #: `(head, kept, ranked table, width)` when the last label frame
        #: factored — `None` otherwise. Read only by `_repeating_bounds`.
        self._rectangle: tuple[tuple[str, ...], tuple[str, ...], str, int] | None = None
        self._n_cols = 0
        self._n_rows = 0
        self._obj_const = 0.0
        self._obj_sense = 'min'
        self._registered = 0

    # -- registration ---------------------------------------------------

    def _relation(self, rel: Relation, prefix: str, *, materialise: bool) -> str:
        """Name *rel* as a table or a view, and return the name.

        **Materialise when the derivation is the expensive part, not when the
        result is read often.** A label relation is read three or four times
        downstream, which argues for a table until you price the two sides: a
        counted label needs an ordered window over the whole coordinate product
        and is worth paying for once, while an arithmetic one is a cross join
        of two small relations and a multiply. Writing ten million rows to
        answer that a second time costs more than answering it three times.

        The polars engine collects in the same places, and for a reason that
        does not carry over: a `LazyFrame` read twice is *planned* twice, and
        the plan under a label reaches all the way back to the parquet.
        """
        self._registered += 1
        name = f'{prefix}_{self._registered}'
        if materialise:
            rel.create(name)
        else:
            rel.create_view(name)
        return name

    def _register(self, name: str, frame: pl.DataFrame) -> str:
        """Copy *frame* into a table of its own, and return the table's name.

        **Copied, not scanned in place.** A registered frame stays a Python
        object: duckdb reads it through the buffer protocol from whichever
        worker thread the scan lands on, and those threads need the GIL — which
        the thread that called `execute` is holding while it waits. On a plan
        with several scans of registered frames feeding one pipeline, that
        deadlocks outright: `transport/l` reproduces it as a build that sits at
        0% CPU indefinitely, while the same query over copies returns in 0.3 s.

        Copying is not free — the frame and the table are both resident for the
        length of one statement — but it buys a build that cannot stall, and
        duckdb's own storage for every scan after the first.

        The frame itself, not `to_arrow()`: duckdb reads polars natively, and
        the round-trip through a pyarrow table is what used to drag pandas into
        a runtime that declares it a bridge out and not a dependency.
        """
        source = f'__source {name}__'
        self._con.register(source, frame)
        try:
            self._con.sql(f'SELECT * FROM {q(source)}').create(name)
        finally:
            self._con.unregister(source)
        return name

    def build(self, program: plan.Program, sources: Mapping[str, Any]) -> None:
        """Bind *sources*, then build every declaration into the model frames.

        Binding is `relational/binding.py`, shared with the polars engine: it
        is `scan_parquet` plus dtype and duplicate-coordinate validation, which
        is what a caller's data has to survive whoever builds the model.
        """
        self._program = program
        self._name_dims = plan.name_dims(program)
        bound = self._bound = bind(program, sources)
        materialised = {n: f.collect() for n, f in bound.dimensions.items()}
        dims = {n: self._register(f'dim_{n}', f) for n, f in materialised.items()}
        params = {n: self._register(f'par_{n}', f.collect()) for n, f in bound.parameters.items()}
        self._compiler = DuckCompiler(
            self._con, program, dims, params, bound.cardinality, bound.boolean_parameters, self._var_tables
        )

        cols = [self._build_variable(v) for v in program.variables]
        built = [self._build_constraint(c) for c in program.constraints]
        objective = self._build_objective(program.objective)

        self._cols = _stack(cols, _COLS)
        self._rows = _stack([r for r, _ in built], _ROWS)
        stacked = _stack([m for _, m in built if m is not None], _MATRIX)
        self._matrix, self._matrix_starts = sinks.compress_rows(stacked, self._n_rows)
        self._obj = _stack([objective] if objective is not None else [], _OBJ)

    @property
    def _variables(self) -> Mapping[str, pl.LazyFrame]:
        return self._var_labels

    @property
    def _constraints(self) -> Mapping[str, pl.LazyFrame]:
        return self._row_labels

    @property
    def _q(self) -> DuckCompiler:
        assert self._compiler is not None, 'build() has not run'
        return self._compiler

    # -- labels ---------------------------------------------------------

    def _label_frame(
        self,
        dims: tuple[str, ...],
        where: plan.Predicate | None,
        label: str,
        start: int,
        restrictions: Sequence[tuple[tuple[str, ...], Relation]] = (),
    ) -> tuple[str, int]:
        """The masked coord product of *dims* with a dense *label* from *start*.

        The same three paths the polars engine has, chosen by the same
        function. **Unmasked**, a row's label is its position in the product —
        arithmetic on the ordinals, no sort and nothing to count. **Masked but
        factoring**, the survivors are a rectangle and the label is arithmetic
        again over a ranked survivor set (:meth:`_factored`). **Otherwise** it
        is counted, which costs the ordered window over the whole product.

        `plan.free_prefix` decides between them for both engines, because a
        label is the solver's own column index: two engines choosing routes
        independently is how they would come to build different models.
        """
        if not restrictions:
            if where is None:
                rows = math.prod(self._q.cardinality[d] for d in dims)
                rel = self._q.frame(dims, None).select(
                    *_coordinates(dims), self._row_major(dims, start).cast('BIGINT').alias(label)
                )
                return self._relation(rel, 'lbl', materialise=False), start + rows

            free = plan.free_prefix(dims, plan.predicate_dims(where, self._name_dims))
            if free:
                return self._factored(dims, free, where, label, start)

        carrier = self._q.frame(dims, where)
        for on, presence in restrictions:
            carrier = restrict_to(carrier, on, presence)
        return self._counted(carrier, dims, label, start)

    def _row_major(self, dims: tuple[str, ...], start: int, alias: str = '') -> Expression:
        """Row-major position in *dims*' coordinate product, offset by *start*.

        The trailing dim has stride 1 and every other is the product of the
        cardinalities to its right, so the position is a dot product against
        the ordinals the frame already carries — no ordering imposed, because
        the answer does not depend on the order rows arrive in. The polars twin
        is `Labeller.row_major`, and both arithmetic paths reach a label
        through it for the reason the label itself is written once.
        """
        offset = ConstantExpression(start)
        if not dims:
            return offset
        terms: list[Expression] = []
        stride = 1
        for d in reversed(dims):
            terms.append(col(_ordinal(d), of=alias) * ConstantExpression(stride))
            stride *= self._q.cardinality[d]
        return reduce(operator.add, reversed(terms)) + offset

    def _counted(self, carrier: Relation, dims: tuple[str, ...], label: str, start: int) -> tuple[str, int]:
        """Rank the surviving rows of *carrier*: the ordered window, and a count.

        The general answer and the expensive one — a global sort of the whole
        product, which is what the two arithmetic paths exist to avoid.
        """
        ranked = _ranked([_ordinal(d) for d in dims], start).alias(label)
        name = self._relation(carrier.select(*_coordinates(dims), ranked), 'lbl', materialise=True)
        return name, start + self._height(name)

    def _factored(
        self,
        dims: tuple[str, ...],
        free: int,
        where: plan.Predicate,
        label: str,
        start: int,
    ) -> tuple[str, int]:
        """Labels for a mask that reads none of the first *free* dims.

        A mask that cannot see the leading dims removes the same coordinates
        under every one of their values, so the survivors are a rectangle: the
        full product of the leading dims against one surviving set. Ranking
        that set costs a window over the *set*, not over the product — on
        `dispatch` it ranks 100 generators instead of 10M
        ``(snapshot, generator)`` pairs, and the window is what a global sort
        made the dominant cost of the build.

        The label is then arithmetic again, through the same
        :meth:`_row_major` the unmasked path uses: row-major over the leading
        dims, times the width of the surviving set, plus a survivor's rank
        within it. That is the same number the window would have counted,
        because for each leading coordinate the same survivors appear in the
        same order — which is what `tests/test_engine_parity.py` checks by
        comparing the built model against the other engine's.
        """
        head, kept = dims[:free], dims[free:]
        ranked = self._relation(
            self._q.frame(kept, where).select(
                *(col(d) for d in kept), _ranked([_ordinal(d) for d in kept], 0).alias('__rank')
            ),
            'srv',
            # the one relation here worth writing down: it is small, it is
            # counted, and its window is the work the rectangle exists to avoid
            materialise=True,
        )
        width = self._height(ranked)
        if width == 0:
            # nothing survived anywhere, so there is no rectangle to describe.
            # The counted path returns the right columns and dtypes for free.
            return self._counted(self._q.frame(dims, where), dims, label, start)

        survivors = self._con.table(ranked).set_alias('s')
        position = self._row_major(head, 0, alias='h')
        placed = (position * ConstantExpression(width) + col('__rank', of='s') + ConstantExpression(start)).cast(
            'BIGINT'
        )
        rel = (
            self._q.frame(head, None)
            .set_alias('h')
            .cross(survivors)
            .select(
                *(col(d, of='h') for d in head),
                *(col(d, of='s') for d in kept),
                placed.alias(label),
            )
        )
        rows = math.prod(self._q.cardinality[d] for d in head) * width
        #: the rectangle, kept for `_repeating_bounds` — a caller that only
        #: reads the trailing dims can be answered once and repeated
        self._rectangle = (head, kept, ranked, width)
        return self._relation(rel, 'lbl', materialise=False), start + rows

    def _height(self, table: str) -> int:
        return self._con.table(table).shape[0]

    # -- declarations ---------------------------------------------------

    def _build_variable(self, v: plan.VariableDeclaration) -> pl.DataFrame:
        """One variable's labelled relation, and its share of ``cols``."""
        start = self._n_cols
        self._rectangle = None
        name, self._n_cols = self._label_frame(v.dims, v.where, 'var_label', start)
        self._var_tables[v.name] = name
        self._blocks[v.name] = (start, self._n_cols - start)

        # **`cols` has no `col`, so a row's place in this frame is its solver
        # column index** (`sinks/tables.py`). The polars engine gets that order
        # from the emission order of a cross join and only *verifies* it; a
        # duckdb relation promises no order at all, so here it is produced
        # deliberately — cheaply where the bounds repeat, and by sorting where
        # they do not.
        repeating = self._repeating_bounds(v)
        bounds = repeating if repeating is not None else self._sorted_bounds(v, name)
        # `vtype` is attached here rather than selected as a literal: one word
        # per column is one *copy* of that word per row over the wire, and the
        # frame's stated dtype is an Enum holding four bytes.
        cols = bounds.pl().with_columns(pl.lit(v.variable_type, dtype=sinks.VTYPE).alias('vtype'))
        bad = cols.filter(pl.col('lb').is_null() | pl.col('ub').is_null()).height
        if bad:
            raise DataError(null_bounds_message(v.name, bad))
        return cols

    def _sorted_bounds(self, v: plan.VariableDeclaration, labels: str) -> Relation:
        """``(lb, ub)`` in label order, by sorting — always available.

        The projection is applied *over* the ordered relation rather than
        beside it, because ``var_label`` is what orders the frame and is not
        one of the two columns a sink wants. duckdb keeps a subquery's order,
        and the parity gate is what would notice if it stopped:
        `test_engine_parity.py` compares byte-for-byte LP files, where a column
        out of place moves every bound after it.
        """
        bounds = self._q.bounds(self._con.table(labels), v)
        return bounds.order(q('var_label')).select(
            col('lb').cast('DOUBLE').alias('lb'), col('ub').cast('DOUBLE').alias('ub')
        )

    def _repeating_bounds(self, v: plan.VariableDeclaration) -> Relation | None:
        """Bounds in label order without a sort, when the same run repeats.

        A factored label frame is a rectangle: every leading coordinate carries
        the *same* surviving set in the same order. So if the bounds read only
        the trailing dims, the whole column is one short run repeated once per
        leading coordinate — build the run, and expand it.

        `unnest` of a list is that expansion, and it is not merely cheaper than
        the sort it replaces, it is cheaper than the cross join: 0.12 s against
        0.43 s at 10M columns, where an *unordered* join of the same shape is
        0.11 s. Ordering stops costing anything at all.

        `None` when the shape does not apply — an unfactored frame, or a bound
        that reads a leading dim and therefore does not repeat. The caller
        sorts, which is always correct and sometimes all that is available.
        """
        if self._rectangle is None:
            return None
        head, kept, ranked, _ = self._rectangle
        reads = {p for e in (v.lower, v.upper) for p in _parameters(e)}
        if any(not set(self._q.program.parameter(p).dims) <= set(kept) for p in reads):
            return None

        run = self._q.bounds(self._con.table(ranked), v)
        # An ordered aggregate has no expression form; the two names it orders
        # and collects are the compiler's own, not the model's.
        lists = run.aggregate(f'list(lb ORDER BY {q("__rank")}) AS lbs, list(ub ORDER BY {q("__rank")}) AS ubs')
        return (
            self._q.frame(head, None)
            .cross(lists.set_alias('k'))
            .select(
                FunctionExpression('unnest', col('lbs', of='k')).cast('DOUBLE').alias('lb'),
                FunctionExpression('unnest', col('ubs', of='k')).cast('DOUBLE').alias('ub'),
            )
        )

    def _build_constraint(self, c: plan.ConstraintDeclaration) -> tuple[pl.DataFrame, pl.DataFrame | None]:
        """One constraint as its ``rows`` and its share of the matrix."""
        lhs = self._q.expression(c.lhs, f"constraint '{c.name}' lhs")
        rhs = self._q.expression(c.rhs, f"constraint '{c.name}' rhs")
        terms = [(p, 1.0) for p in lhs.terms] + [(p, -1.0) for p in rhs.terms]
        consts = [(p, 1.0) for p in rhs.consts] + [(p, -1.0) for p in lhs.consts]
        for p, _ in [*terms, *consts]:
            extra = set(p.dims) - set(c.dims)
            if extra:
                raise LanguageError(
                    f"constraint '{c.name}': expression has dims {sorted(extra)} outside "
                    f'foreach {list(c.dims)} — missing a Sum/GroupSum?'
                )

        restrictions = _absence_restrictions([p for p, _ in terms])
        start = self._n_rows
        name, self._n_rows = self._label_frame(c.dims, c.where, 'row', start, restrictions)
        self._row_tables[c.name] = name
        self._blocks[c.name] = (start, self._n_rows - start)
        frame = self._con.table(name)

        rows = self._constant_side(frame, consts, c.sense).pl()

        if not terms:
            self._omitted[c.name] = rows.height
            self._blocks[c.name] = (start, 0)
            self._n_rows = start
            self._row_tables[c.name] = self._relation(frame.filter(ConstantExpression(False)), 'lbl', materialise=False)
            return rows.clear(), None

        matrix = self._matrix_side(frame, terms).pl()
        rows, matrix, self._n_rows = self._drop_termless_rows(c.name, name, rows, matrix, start)
        return rows, matrix

    def _constant_side(self, frame: Relation, consts: list[tuple[TermFragment, float]], sense: str) -> Relation:
        """``(row, sense, rhs)`` — every constant fragment folded onto the frame."""
        carrier = frame
        accumulated: list[Expression] = []
        for i, (p, sign) in enumerate(consts):
            column = f'__const {i}__'
            aggregated = self._q.constant_scalar(p).set_alias('r')
            left = carrier.set_alias('l')
            joined = left.join(aggregated, matching(p.dims), how='left') if p.dims else left.cross(aggregated)
            carrier = joined.select(*(col(x, of='l') for x in carrier.columns), col('cval', of='r').alias(column))
            accumulated.append(ConstantExpression(sign) * CoalesceOperator(col(column), ConstantExpression(0.0)))
        total = reduce(operator.add, accumulated) if accumulated else ConstantExpression(0.0)
        return carrier.select(col('row'), ConstantExpression(sense).alias('sense'), total.cast('DOUBLE').alias('rhs'))

    def _matrix_side(self, frame: Relation, terms: list[tuple[TermFragment, float]]) -> Relation:
        """``(row, col, coeff)`` for one constraint, aggregated only when it must be."""
        pieces = []
        for p, sign in terms:
            left, right = frame.set_alias('l'), p.rel.set_alias('r')
            joined = left.join(right, matching(p.dims)) if p.dims else left.cross(right)
            pieces.append(
                joined.select(
                    col('row', of='l'),
                    col('var_label', of='r').cast('INTEGER').alias('col'),
                    (ConstantExpression(sign) * col('coeff', of='r')).cast('DOUBLE').alias('coeff'),
                )
            )
        stacked = union_all(pieces[0], pieces[1:])
        if not _needs_aggregate([f for f, _ in terms], self._q.may_share_a_column):
            return stacked
        # `sum` over `(row, col)` is the terminal aggregate — where duplicates
        # from Sum and GroupSum, which project rather than aggregate, collapse.
        # Unordered: every sink sorts the matrix into the order it needs
        # (`lp_file` by `(row, col)`, `solver_direct` by `row`), so ordering
        # here is a second sort of the largest frame in the model for nothing.
        total = FunctionExpression('sum', col('coeff')).alias('coeff')
        return stacked.aggregate([col('row'), col('col'), total], f'{q("row")}, {q("col")}')

    def _drop_termless_rows(
        self, constraint: str, labels: str, rows: pl.DataFrame, matrix: pl.DataFrame, start: int
    ) -> tuple[pl.DataFrame, pl.DataFrame, int]:
        """Rows that kept no variable term are not built, and the block closes up.

        The polars engine's twin, and the reason it is written twice rather
        than shared: the frames are polars on both sides, but the *label
        relation* is a duckdb table here, so the renumbering that follows the
        drop is a window over that table rather than a `replace_strict` on a
        frame. The rule itself is the language's (SPEC §6) and must not differ —
        a row with no variables asserts something about constants, which the
        solver cannot act on.

        Labels are dense, and the dual read-back reads a block by position, so
        a dropped row cannot leave a gap: the survivors are renumbered from
        *start* and the row counter rewinds to match.
        """
        kept = matrix.get_column('row').unique()
        if kept.len() == rows.height:
            return rows, matrix, start + rows.height

        surviving = rows.filter(pl.col('row').is_in(kept)).sort('row')
        renumber = surviving.select('row').with_row_index('__new__', offset=start)
        self._omitted[constraint] = rows.height - surviving.height
        self._blocks[constraint] = (start, surviving.height)
        remap = dict(zip(renumber.get_column('row'), renumber.get_column('__new__'), strict=True))
        dropped = rows.filter(~pl.col('row').is_in(kept)).select('row')
        rows = surviving.with_columns(pl.col('row').replace_strict(remap))
        matrix = matrix.with_columns(pl.col('row').replace_strict(remap))
        self._row_tables[constraint] = self._relation(
            self._surviving_labels(constraint, labels, dropped, start), 'lbl', materialise=True
        )
        return rows, matrix, start + surviving.height

    def _surviving_labels(self, constraint: str, labels: str, dropped: pl.DataFrame, start: int) -> Relation:
        """The label relation without the *dropped* rows, renumbered from *start*.

        Anti-joined against the labels that went rather than restricted to the
        ones that stayed: a row loses all its terms rarely, so the dropped side
        is the small frame and the survivors are most of the model.
        """
        table = self._con.table(labels)
        absent = self._con.table(self._register(f'__dropped {constraint}__', dropped)).set_alias('r')
        remaining = table.set_alias('l').join(absent, matching(('row',)), how='anti')
        coordinates = [c for c in table.columns if c != 'row']
        return remaining.select(*(col(c) for c in coordinates), _ranked(['row'], start).alias('row'))

    def _build_objective(self, o: plan.ObjectiveDeclaration) -> pl.DataFrame | None:
        """The objective as ``(col, coeff)``, or ``None`` if it has no terms."""
        comp = self._q.expression(o.expression, 'objective')
        for p in comp.consts:
            if p.dims:
                raise LanguageError(
                    'objective constant part has dims — wrap parameter terms in '
                    'Mul with a Var, or pre-aggregate to a scalar'
                )
            row = p.rel.aggregate([FunctionExpression('sum', col('cval')).alias('cval')]).fetchone()
            self._obj_const += (row[0] if row else None) or 0.0
        self._obj_sense = o.sense
        if not comp.terms:
            return None
        pieces = [p.rel.select(col('var_label').cast('INTEGER').alias('col'), col('coeff')) for p in comp.terms]
        stacked = union_all(pieces[0], pieces[1:])
        if _needs_aggregate(comp.terms, self._q.may_share_a_column, projected=True):
            total = FunctionExpression('sum', col('coeff')).alias('coeff')
            return stacked.aggregate([col('col'), total], q('col')).pl()
        return stacked.pl()

    # -- the sink seam --------------------------------------------------

    def _tables(self) -> sinks.ModelTables:
        assert self._cols is not None and self._obj is not None
        assert self._rows is not None and self._matrix is not None
        return sinks.ModelTables(
            cols=self._cols,
            obj=self._obj,
            rows=self._rows,
            matrix=self._matrix,
            row_starts=self._matrix_starts,
            column_count=self._n_cols,
            row_count=self._n_rows,
            objective_sense=self._obj_sense,
            objective_constant=self._obj_const,
        )

    def close(self) -> None:
        """Drop the built model. Calling twice is not an error."""
        self._cols = self._obj = self._rows = self._matrix = self._matrix_starts = None
        self._bound = None
        self._var_labels.clear()
        self._row_labels.clear()
        self._blocks.clear()
        self._omitted.clear()
        self._compiler = None
        self._con.close()


def _coordinates(dims: tuple[str, ...]) -> list[Expression]:
    """The columns a label frame carries beside its label.

    A scalar declaration has none, and a relation with an empty projection is
    not a relation — so it carries the unit column the empty coordinate product
    is made of.
    """
    return [col(d) for d in dims] if dims else [col(UNIT)]


def _parameters(expression: plan.Expression) -> set[str]:
    """Every parameter name under *expression*."""
    found = {expression.name} if isinstance(expression, plan.Parameter) else set()
    for child in plan.children(expression):
        found |= _parameters(child)
    return found


def _stack(frames: list[pl.DataFrame], columns: tuple[str, ...]) -> pl.DataFrame:
    kept = [f for f in frames if f.height]
    if not kept:
        return pl.DataFrame(schema={name: sinks.DTYPES[name] for name in columns})
    return pl.concat([f.select(columns) for f in kept])


def _needs_aggregate(
    terms: Sequence[TermFragment],
    may_share: Callable[[TermFragment, TermFragment], bool],
    *,
    projected: bool = False,
) -> bool:
    """Whether stacking *terms* can put two rows on one solver column.

    Named for the answer, not the condition: an inverted test here is a wrong
    model rather than a slow one.

    Two things can put a label twice into the stack, asked separately. A
    fragment that is not `keyed` already holds one twice on its own. Whether a
    *pair* can is *may_share*, which answers no for distinct variables and
    otherwise asks whether two fragments of one variable send a label to one
    **row** — see `DuckCompiler.may_share_a_column`. That second half is what
    makes the ordinary multi-term constraint free: reading only a fragment
    count says the aggregate is reachable for ``reserve_up + reserve_down <=
    p_max``, which sorts every nonzero in the model to collapse nothing. Worth
    0.90-0.94x of build on the three ladder cases with multi-term constraints,
    and nothing on the other four (#638).

    *projected* is what the two call sites do not share. The matrix keeps a
    fragment's dims, so `keyed` — one row per ``(dims…, var_label)`` — carries
    straight into ``(row, col)``. The objective keeps only ``var_label``, so it
    asks the stronger question: does the key survive losing *all* dims?

    **This is not how the polars engine answers any more.** #520 replaced the
    same reasoning there with three linear probes over the built matrix — is it
    ordered, does any cell repeat — which is exact where this is conservative,
    and costs a pass rather than a dimension-table query. Whether that is the
    better trade here is unmeasured: this engine builds its matrix in duckdb
    and would have to fetch it unaggregated to probe it.
    """
    if any(not t.survives_dropping(set(t.dims) if projected else set()) for t in terms):
        return True
    return any(may_share(a, b) for i, a in enumerate(terms) for b in terms[i + 1 :])


def _absence_restrictions(terms: Sequence[TermFragment]) -> list[tuple[tuple[str, ...], Relation]]:
    """Where a masked variable says a constraint row must not exist."""
    return [(p.presence_dims or p.dims, p.presence) for p in terms if p.presence is not None]

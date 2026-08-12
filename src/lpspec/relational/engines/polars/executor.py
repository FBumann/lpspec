"""Polars executor: fill the model frames, then hand them to a sink.

Owns the *assembly* — turning each declaration into rows of the four model
frames, and holding them until a sink drains them. Owns none of the three
questions it asks on the way: what the data is
(:mod:`lpspec.relational.engines.polars.binding`), what a query over it looks like
(:mod:`lpspec.relational.engines.polars.compiler`), which coordinate gets which solver index
(:mod:`lpspec.relational.engines.polars.labels`). The lane is described in
docs/ARCHITECTURE.md.

The two registries it does own are the ones that fill *during* assembly — the
variable and constraint frames — because a declaration built later has to see
what earlier ones produced. Everything binding produced is frozen by contrast,
which is what :class:`~lpspec.relational.engines.polars.binding.BoundSources` says.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, get_args

import polars as pl

from lpspec.errors import (
    DataError,
    LanguageError,
    LpspecError,
    null_bounds_message,
    sparse_divisor_message,
    uncovered_constant_message,
)
from lpspec.relational import plan, sinks
from lpspec.relational.binding import BoundSources, bind
from lpspec.relational.engine import Engine
from lpspec.relational.engines.polars import labels
from lpspec.relational.engines.polars.compiler import PolarsCompiler, TermFragment

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from polars._typing import MaintainOrderJoin


#: The four frames a sink reads, as schemas. Stated here because the executor
#: is what fills them and an empty model still has to have them.
_COLS = ('lb', 'ub', 'vtype')
_OBJ = ('col', 'coeff')
_ROWS = ('row', 'sense', 'rhs')
_MATRIX = ('row', 'col', 'coeff')

#: The dtype of each of those columns. ``vtype`` is an ``Enum`` over the
#: variable types the plan declares, rather than a string: it holds one word
#: per column and the same handful of words for the whole model, so as a string
#: it stores that word once per row, where an Enum stores a code: on a wide
#: model that is most of the ``cols`` frame (#189). The Enum also makes the
#: vocabulary
#: explicit, so a fourth variable type added to
#: :data:`~lpspec.relational.plan.VariableType` and not reaching here fails
#: where the column is built rather than in whichever sink first compares
#: against a name it does not know.
#:
#: ``col`` is ``Int32`` — the solver's own index width, HiGHS and Gurobi both
#: being 32-bit indexed, so a count past 2^31 has no sink that could take it
#: and the strict cast raises there rather than wrapping. The cast sits where
#: the column is *produced*, inside the per-declaration streaming collect, and
#: must stay there: narrowing the stacked frame instead allocates the narrow
#: copy while the wide one is still alive, a transient visible in
#: `dispatch/l`'s peak RSS. A *label* stays ``Int64``: the arithmetic path
#: computes it as a position in the full pre-mask coordinate product, which
#: can pass 2^31 while every survivor fits.
_DTYPES = {
    'col': pl.Int32, 'row': pl.Int64,
    'lb': pl.Float64, 'ub': pl.Float64, 'rhs': pl.Float64, 'coeff': pl.Float64,
    'sense': pl.String, 'vtype': pl.Enum(get_args(plan.VariableType)),
}  # fmt: skip


class PolarsExecutor(Engine):
    """Build a :class:`Program` into polars frames, then sink it."""

    def __init__(self) -> None:
        self._program: plan.Program | None = None
        self._compiler: PolarsCompiler | None = None
        self._bound: BoundSources | None = None
        self._var_frames: dict[str, pl.LazyFrame] = {}
        self._row_frames: dict[str, pl.LazyFrame] = {}
        #: ``name -> rows not built``, because every term they had vanished.
        #: Empty for a model whose every declared row reached the solver.
        self._omitted: dict[str, int] = {}
        #: ``name -> (first label, how many)``.
        #: :func:`~lpspec.relational.engines.polars.labels.frame` hands a
        #: declaration a *contiguous, dense* run of labels, so a declaration's
        #: share of a solver vector is a slice of it.
        self._blocks: dict[str, tuple[int, int]] = {}
        self._cols: pl.DataFrame | None = None
        self._obj: pl.DataFrame | None = None
        self._rows: pl.DataFrame | None = None
        self._matrix: pl.DataFrame | None = None
        self._matrix_starts: Any = None
        self._n_cols = 0
        self._n_rows = 0
        self._obj_const = 0.0
        self._obj_sense: str = 'min'

    # ------------------------------------------------------------------
    # build
    # ------------------------------------------------------------------

    def build(self, program: plan.Program, sources: Mapping[str, Any]) -> None:
        """Bind *sources*, then build every declaration into the model frames.

        The compiler comes after binding, two of its answers being read off the
        data — a dim's size, whether a parameter is boolean. Declarations build
        one at a time and concatenate at the end; their rows are independent,
        which is what lets the model be four frames rather than a graph.

        The matrix leaves in ``(row, col)`` order, as ``ModelTables`` promises
        its sinks. The stack already has it — each share leaves sorted and owns
        the next run of rows — so :func:`_in_row_order` *checks* with one
        linear scan rather than sorting the model's largest frame at the peak
        of the build. :func:`_row_starts` reads the CSR index off that order,
        after which ``row`` is dropped: 8 bytes per entry no sink reads.
        """
        self._program = program
        self._bound = bind(program, sources)
        self._compiler = PolarsCompiler(program, self._bound, self._var_frames)

        cols = [self._build_variable(v) for v in program.variables]
        built = [self._build_constraint(c) for c in program.constraints]
        objective = self._build_objective(program.objective)

        self._cols = _stack(cols, _COLS)
        self._rows = _stack([r for r, _ in built], _ROWS)
        ordered = _in_row_order(_stack([m for _, m in built if m is not None], _MATRIX))
        self._matrix_starts = _row_starts(ordered, self._n_rows)
        self._matrix = ordered.select('col', 'coeff').rechunk()
        self._obj = _stack([objective] if objective is not None else [], _OBJ)

    @property
    def _variables(self) -> Mapping[str, pl.LazyFrame]:
        return self._var_frames

    @property
    def _constraints(self) -> Mapping[str, pl.LazyFrame]:
        return self._row_frames

    @property
    def _q(self) -> PolarsCompiler:
        assert self._compiler is not None, 'build() has not run'
        return self._compiler

    def _refuse_undefined_divisors(self, stacked: pl.DataFrame, name: str, *expressions: plan.Expression) -> None:
        """A null coefficient means a divisor had no value where the model divided.

        A quotient left-joins its divisor (:func:`_join_mul`), so a missing
        value leaves a null — and a term whose row was masked out, or whose
        numerator variable is absent, never gets this far. That is what keeps
        the refusal from becoming a wall on ordinary sparse data. Asked of the
        stack before any cell collapses, since ``sum`` reads a null as zero.
        """
        undefined = int(stacked.get_column('coeff').null_count())
        if undefined:
            params = sorted(plan.divisor_parameters(*expressions))
            raise DataError(f'{name}: {sparse_divisor_message(", ".join(params), undefined)}')

    def _matrix_share(self, pieces: list[pl.LazyFrame], name: str, *expressions: plan.Expression) -> pl.DataFrame:
        """One constraint's share: in ``(row, col)`` order, repeated cells summed.

        Nothing runs unconditionally except three linear probes — the null
        count, whether the stack arrives in order, whether any cell repeats. It
        usually is in order and usually repeats nothing (a cell repeats only
        when one variable reaches a row twice), so the sort and the aggregate
        run only when a probe says they would change something, which is far
        cheaper than the unconditional hash aggregate it replaced. Read off the
        data, so nothing here knows *why* a cell repeats (#520).

        **Cheap probes are why the share is rechunked first.** A streaming
        collect returns morsels as chunks, and ``shift(1)`` pays at every
        boundary where ``is_sorted`` does not, so rechunking and probing
        together beats probing as it arrives (#576). Not an extra copy, since
        the assembly needs a contiguous matrix anyway (#550). The objective's
        stack is left fragmented — measured, and it costs peak for no wall.
        """
        stacked = pl.concat(pieces).collect(engine='streaming').rechunk()
        self._refuse_undefined_divisors(stacked, name, *expressions)
        row, col = pl.col('row'), pl.col('col')
        tied, ahead = row == row.shift(1), row > row.shift(1)
        repeat = tied & (col == col.shift(1))
        probes = stacked.select(
            (ahead | (tied & (col >= col.shift(1)))).all().alias('#ordered'),
            repeat.any().alias('#repeated'),
        )
        ordered, repeated = probes.row(0)
        if not ordered:
            stacked = stacked.sort('row', 'col')
            repeated = stacked.select(repeat.any()).item()
        if not repeated:
            return stacked
        matrix = stacked.lazy().group_by('row', 'col').agg(pl.col('coeff').sum()).sort('row', 'col')
        return matrix.collect(engine='streaming')

    # ------------------------------------------------------------------
    # declarations
    # ------------------------------------------------------------------

    def _build_variable(self, v: plan.VariableDeclaration) -> pl.DataFrame:
        """One variable's labelled frame, and its share of ``cols``.

        The share leaves in label order, ``cols`` carrying no ``col`` of its
        own: a row's *position* is its solver column index. The bounds joins
        usually keep that order, so it is verified with one linear scan and
        re-established only when a join lost it
        (:func:`labels.in_position_order`).

        Only the label and the two bounds are collected, keeping the dim
        columns and joined bound parameters inside the lazy pipeline rather
        than materialising them to be dropped. The label then goes too, having
        been the order's witness and nothing else.
        """
        start = self._n_cols
        labelled, self._n_cols = labels.frame(self._q, v.dims, v.where, 'var_label', start)
        self._var_frames[v.name] = labelled.lazy()
        self._blocks[v.name] = (start, labelled.height)

        bounded = labels.in_position_order(
            self._q.bounds(labelled.lazy(), v)
            .select('var_label', pl.col('lb').cast(pl.Float64), pl.col('ub').cast(pl.Float64))
            .collect(engine='streaming'),
            'var_label',
        )
        cols = bounded.select('lb', 'ub', pl.lit(v.variable_type, dtype=_DTYPES['vtype']).alias('vtype'))

        bad = cols.filter(pl.col('lb').is_null() | pl.col('ub').is_null()).height
        if bad:
            raise DataError(null_bounds_message(v.name, bad))
        return cols

    def _build_constraint(self, c: plan.ConstraintDeclaration) -> tuple[pl.DataFrame, pl.DataFrame | None]:
        """One constraint as its ``rows`` and its share of the matrix.

        Terms normalise to the left, constants to the right. Each constant
        fragment is aggregated to its own coordinates and left-joined, so a
        coordinate it has no row for contributes zero.

        Duplicates from ``Sum`` and ``GroupSum`` — which project rather than
        aggregate — and from ``x + 2 * x`` collapse in :meth:`_matrix_share`'s
        terminal ``SUM(coeff) GROUP BY row, col``, read off the data rather
        than reasoned from how the fragments were reshaped (#520). The share
        leaves in ``(row, col)`` order, which is how every sink reads it.

        The labelled frame is kept for the dual read-back, and its block
        narrows when rows go termless: the run of labels a declaration owns is
        what survived, not what it declared (#561).
        """
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
        labelled, self._n_rows = labels.frame(self._q, c.dims, c.where, 'row', start, restrictions)
        frame = labelled.lazy()
        self._row_frames[c.name] = frame
        self._blocks[c.name] = (start, labelled.height)

        accumulated = pl.lit(0.0, dtype=pl.Float64)
        uncovered: pl.Expr | None = None
        carrier = frame
        for i, (p, sign) in enumerate(consts):
            column = f'__const {i}__'
            aggregated = self._q.constant_scalar(p).rename({'cval': column})
            carrier = (
                carrier.join(aggregated, on=list(p.dims), how='left')
                if p.dims
                else carrier.join(aggregated, how='cross')
            )
            accumulated = accumulated + sign * pl.col(column).fill_null(0.0)
            gap = pl.col(column).is_null()
            uncovered = gap if uncovered is None else uncovered | gap

        if uncovered is not None:
            gaps = int(carrier.select(uncovered.sum()).collect(engine='streaming').item())
            if gaps:
                names = ', '.join(sorted(plan.parameters_of(c.lhs, c.rhs)))
                raise DataError(uncovered_constant_message(names, gaps, f"constraint '{c.name}'"))

        rows = carrier.select(
            'row',
            pl.lit(c.sense, dtype=pl.String).alias('sense'),
            accumulated.cast(pl.Float64).alias('rhs'),
        ).collect(engine='streaming')

        if not terms:
            self._omitted[c.name] = rows.height
            self._blocks[c.name] = (start, 0)
            self._n_rows = start
            self._row_frames[c.name] = self._row_frames[c.name].clear()
            return rows.clear(), None

        pieces = []
        carried_order: MaintainOrderJoin | None = 'left_right' if len(terms) == 1 else None
        for p, sign in terms:
            placed = (
                frame.join(p.frame, on=list(p.dims), how='inner', maintain_order=carried_order)
                if p.dims
                else frame.join(p.frame, how='cross')
            )
            pieces.append(
                placed.select(
                    'row',
                    pl.col('var_label').cast(_DTYPES['col']).alias('col'),
                    (sign * pl.col('coeff')).cast(pl.Float64).alias('coeff'),
                )
            )
        matrix = self._matrix_share(pieces, f"constraint '{c.name}'", c.lhs, c.rhs)
        rows, matrix, self._n_rows = self._drop_termless_rows(c.name, rows, matrix, start)
        return rows, matrix

    def _drop_termless_rows(
        self, name: str, rows: pl.DataFrame, matrix: pl.DataFrame, start: int
    ) -> tuple[pl.DataFrame, pl.DataFrame, int]:
        """Rows that kept no variable term are not built, and the block closes up.

        A row with no variables is not a constraint — it asserts something
        about constants, which the solver cannot act on. Three provenances
        reach that shape (an absent variable, an empty reduction, a missing
        coefficient) and all three drop the row, so the rule is stated once
        here rather than per provenance.

        Labels are dense and the dual read-back reads a block by position, so a
        dropped row may not leave a gap: survivors renumber from *start* and
        the row counter rewinds. Costs one ``n_unique`` on an in-memory frame
        when nothing is dropped, which is every correct model.
        """
        kept = matrix.get_column('row').unique()
        if kept.len() == rows.height:
            return rows, matrix, start + rows.height

        surviving = rows.filter(pl.col('row').is_in(kept)).sort('row')
        renumber = surviving.select('row').with_row_index('__new__', offset=start)
        self._omitted[name] = rows.height - surviving.height
        self._blocks[name] = (start, surviving.height)
        remap = dict(zip(renumber.get_column('row'), renumber.get_column('__new__'), strict=True))
        rows = surviving.with_columns(pl.col('row').replace_strict(remap))
        matrix = matrix.with_columns(pl.col('row').replace_strict(remap))
        self._row_frames[name] = (
            self._row_frames[name].filter(pl.col('row').is_in(kept)).with_columns(pl.col('row').replace_strict(remap))
        )
        return rows, matrix, start + surviving.height

    def _build_objective(self, o: plan.ObjectiveDeclaration) -> pl.DataFrame | None:
        """The objective as ``(col, coeff)``, or ``None`` if it has no terms.

        This projection drops the dims, so a dim that arrived by broadcast puts
        several rows on one column and their **sum** is the coefficient.
        Nothing downstream computes it — the hand-off scatters with
        ``dense[at] = values``, which keeps the *last* write — so the aggregate
        here is what makes the objective the one the file wrote.

        The aggregate runs only when a column repeats, probed by ``n_unique``
        — the only sound probe here, the stack arriving unordered so adjacency
        proves nothing. Buying order to probe linearly is a dead end twice
        over: the mul join's ``maintain_order`` holds the label order on some
        shapes and loses it on others differing only in data (`dispatch` keeps
        it, `nodal` and `profiled` do not, all three the same lone masked
        ``p * cost``), so no static gate can say when the tax will pay; and
        paid for nothing it multiplies the objective phase several times over,
        against a best case that is a wash (#581). ``obj`` carries no order
        contract anyway.
        """
        comp = self._q.expression(o.expression, 'objective')
        for p in comp.consts:
            if p.dims:
                raise LanguageError(
                    'objective constant part has dims — wrap parameter terms in '
                    'Mul with a Var, or pre-aggregate to a scalar'
                )
            self._obj_const += p.frame.select(pl.col('cval').sum()).collect().item() or 0.0
        self._obj_sense = o.sense
        if not comp.terms:
            return None
        pieces = [
            p.frame.select(pl.col('var_label').cast(_DTYPES['col']).alias('col'), pl.col('coeff')) for p in comp.terms
        ]
        stacked = pl.concat(pieces).collect(engine='streaming')
        self._refuse_undefined_divisors(stacked, 'objective', o.expression)
        if stacked.get_column('col').n_unique() == stacked.height:
            return stacked
        return stacked.lazy().group_by('col').agg(pl.col('coeff').sum()).collect(engine='streaming')

    # ------------------------------------------------------------------
    # sinks — see relational/sinks/; the executor only supplies the frames
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------

    def close(self) -> None:
        """Drop the built model. Optional — see :class:`Result`.

        ``BoundSources`` is frozen, so it cannot be emptied in place the way
        the registries can: what frees the bound frames is dropping every
        reference to them, and the compiler holds one.
        """
        self._cols = self._obj = self._rows = self._matrix = self._matrix_starts = None
        self._var_frames.clear()
        self._row_frames.clear()
        self._blocks.clear()
        self._bound = None
        self._compiler = None


def _spanning(solver: str, quantity: str, values: pl.Series | None, expected: int) -> None:
    """Refuse a solver vector that does not span the model.

    Reading a solution back is positional, so a wrong length is an answer about
    a *different* model. Checked where the solver hands it over rather than
    where it is read: the objective comes back directly, so a ``Result`` built
    on a broken vector would report a plausible number and only fail if someone
    asked for a coordinate.

    ``None`` is not a wrong length — a mixed-integer model has no duals, and
    neither does a run stopped short of a simplex basis.
    """
    if values is not None and len(values) != expected:
        raise LpspecError(
            f'{solver} returned {len(values)} {quantity} values for a model with {expected}. '
            f'Reading a solution back is positional, so a vector that does not span the model '
            f'describes a different one. This is an engine bug rather than a problem with the '
            f'model — please report it.'
        )


def _in_row_order(matrix: pl.DataFrame) -> pl.DataFrame:
    """*matrix* with ``row`` known to ascend — checked, not assumed.

    Ordered by construction, each constraint's share sorted and stacked in
    ascending row ranges. Worth *checking* because :func:`_row_starts` reads one
    run per row off this column and scatters the lengths by value: a row whose
    entries arrived in two runs would have the first overwritten and the CSR
    spans would be silently wrong. The correctness floor of the layout, not a
    speed choice — ``is_sorted`` is a linear scan and the sort never runs.
    """
    if matrix.height and not matrix['row'].is_sorted():
        return matrix.sort('row')
    return matrix


def _row_starts(ordered: pl.DataFrame, row_count: int) -> Any:
    """Each row's first entry in the row-ordered *ordered* — CSR's own index.

    Run-length, scatter, cumulative sum — robust to the model's shape where the
    alternatives are not: ``bincount`` pays per entry — several times rle's
    cost on a large matrix (#550) — and ``searchsorted`` per row times log
    entries.
    Computed here so ``row`` can then be dropped, since every consumer either
    slices by these starts or asks
    :meth:`~lpspec.relational.sinks.tables.ModelTables.matrix_block` to spell
    the labels back out.

    The kept matrix is then **rechunked, once**: a sink slices it per row
    block, and against a chunked frame every block's ``to_numpy`` is a
    gather-copy where against one contiguous buffer it is a view (#550).
    """
    import numpy as np

    runs = ordered['row'].rle()
    starts = np.zeros(row_count + 1, dtype=np.int64)
    starts[runs.struct.field('value').to_numpy() + 1] = runs.struct.field('len').to_numpy()
    return np.cumsum(starts, out=starts)


def _stack(frames: list[pl.DataFrame], columns: tuple[str, ...]) -> pl.DataFrame:
    """Concatenate *frames*, or an empty frame of *columns* when there are none.

    Named rather than inferred because a model may legitimately have nothing to
    stack, and a sink still has to find what it reads.
    """
    if frames:
        return pl.concat(frames)
    return pl.DataFrame(schema={name: _DTYPES[name] for name in columns})


def _absence_restrictions(terms: Sequence[TermFragment]) -> list[tuple[tuple[str, ...], pl.LazyFrame]]:
    """The presence frames a constraint's rows have to be contained in.

    Absence propagates into a comparison and drops the row (v1
    ``convention.rst`` §6, §12): ``x + y >= 10`` where ``y`` is masked is not
    ``x >= 10``, it is no constraint at all.

    Only *variable* absence counts — a sparse parameter's missing rows mean a
    zero coefficient (SPEC §8) — which is why the fragment carries
    :attr:`TermFragment.presence` separately from its frame, and why this reads
    that. A fragment with nothing to restrict is skipped, an unmasked variable
    existing at every coordinate of its foreach.

    *Having* no dims is not *having nothing to restrict*: a masked scalar
    variable restricts every row of every constraint naming it, all or nothing.
    Each restriction is keyed by ``presence_dims`` where the fragment states
    them — narrower than ``dims`` for an acyclic shift.
    """
    out: list[tuple[tuple[str, ...], pl.LazyFrame]] = []
    for p in terms:
        if p.presence is None:
            continue
        out.append((p.presence_dims or p.dims, p.presence))
    return out

"""What every sink reads, and nothing more.

Four frames plus the scalars a writer needs to size its batching, and the
projections more than one sink needs — the dense column and row vectors, the
matrix a block at a time. Those belong to the contract rather than to either
solver: two sinks computing them separately could disagree about the model
they loaded, which is the one thing neither may do.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, get_args

import polars as pl

from lpspec.relational import chunking, plan

if TYPE_CHECKING:
    from collections.abc import Iterator

    import numpy as np
    import numpy.typing as npt

    #: What a solver sink is handed: three float vectors and an integrality
    #: mask, each as long as the model has columns.
    DenseColumns = tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.bool_]
    ]

    #: A sense code and a right-hand side, each as long as the model has rows.
    DenseRows = tuple[npt.NDArray[np.uint8], npt.NDArray[np.float64]]

    #: One chunk of rows: the half-open label range, the matrix entries those
    #: rows own, and the offset of each row's entries within them.
    RowBlock = tuple[int, int, pl.DataFrame, npt.NDArray[np.int64]]


#: ``sense`` as a number, so a row's comparison crosses into numpy as one byte
#: rather than as a boxed Python string. The order is arbitrary and shared: a
#: solver indexes its own spelling with these, so the two cannot drift.
SENSE_CODES = {'<=': 0, '>=': 1, '==': 2}


#: The columns of each frame, in order.
COLS = ('lb', 'ub', 'vtype')
OBJ = ('col', 'coeff')
ROWS = ('row', 'sense', 'rhs')
MATRIX = ('row', 'col', 'coeff')

#: And the dtype of each column. Here rather than in an executor because it is
#: what a *sink* reads: two engines filling the same four frames with different
#: types is a difference no sink asked for and none can see coming.
#:
#: ``vtype`` is an ``Enum`` over the variable types the plan declares, rather
#: than a string: it holds one word per column and the same handful of words
#: for the whole model, so as a string it stores that word once per row —
#: 0.098 GB of the ``cols`` frame's 0.333 at 9.8M columns, against 0.010 as an
#: Enum. The Enum also makes the vocabulary explicit, so a fourth variable type
#: added to :data:`~lpspec.relational.plan.VariableType` and not reaching here
#: fails where the column is built rather than in whichever sink first compares
#: against a name it does not know.
#:
#: ``col`` is ``Int32`` — the solver's own index width, HiGHS and Gurobi both
#: being 32-bit indexed, so a count past 2^31 has no sink that could take it
#: and the strict cast raises there rather than wrapping. An engine casts where
#: the column is *produced*, not on the stacked frame: narrowing afterwards
#: allocates the narrow copy while the wide one is still alive, a transient
#: visible in `dispatch/l`'s peak RSS. A *label* stays ``Int64``: it is a
#: position in the full pre-mask coordinate product, which can pass 2^31 while
#: every survivor fits.
DTYPES = {
    'col': pl.Int32, 'row': pl.Int64,
    'lb': pl.Float64, 'ub': pl.Float64, 'rhs': pl.Float64, 'coeff': pl.Float64,
    'sense': pl.String, 'vtype': pl.Enum(get_args(plan.VariableType)),
}  # fmt: skip

#: The variable-type column's dtype, which an engine builds a literal against.
VTYPE = DTYPES['vtype']


def compress_rows(matrix: pl.DataFrame, row_count: int) -> tuple[pl.DataFrame, npt.NDArray[np.int64]]:
    """A ``(row, col, coeff)`` matrix as the CSR pair `ModelTables` takes.

    Here rather than in an engine because it is the *contract's* layout: both
    engines stack their constraints' shares in declaration order, which is
    ascending row ranges, and both then owe a sink the same compressed form.
    Two engines compressing separately could disagree about which entries a row
    owns — the one thing neither may do.

    ``row`` is known to ascend, and that is **checked rather than assumed**.
    polars cannot see the ordering through a ``concat``, and a sink that finds
    the flag missing orders the whole matrix again; ``is_sorted`` is a linear
    scan over a column the frame already holds, and the sort behind it is the
    correctness floor, expected never to run.

    The starts are a run-length over that sorted column, then a scatter and a
    cumulative sum — robust to the model's shape where the obvious alternatives
    are not: ``bincount`` pays per entry (26 ms to rle's 7 at 10M entries over
    100k rows), ``searchsorted`` per row times the log of the entries, and
    either is the wrong one on some ladder case. Computed here so ``row`` can
    then be dropped: a label repeated once per nonzero is 8 bytes per entry no
    sink reads, since every consumer either slices by these starts or asks
    :meth:`ModelTables.matrix_block` to spell the labels back out.

    The kept matrix is then **rechunked, once**. A streaming collect leaves it
    in chunks, and a sink slices it per row block — against a chunked frame
    every block's ``to_numpy`` is a gather-copy, where against one contiguous
    buffer it is a view (codspeed caught the difference as -6.9% on
    `profiled-m`, ~150 blocks over 16 chunks).
    """
    import numpy as np

    if not matrix.height:
        ordered = matrix
    elif not matrix['row'].is_sorted():
        ordered = matrix.sort('row')
    else:
        ordered = matrix.with_columns(pl.col('row').set_sorted())

    runs = ordered['row'].rle()
    starts = np.zeros(row_count + 1, dtype=np.int64)
    starts[runs.struct.field('value').to_numpy() + 1] = runs.struct.field('len').to_numpy()
    return ordered.select('col', 'coeff').rechunk(), np.cumsum(starts, out=starts)


@dataclass(frozen=True)
class ModelTables:
    """The built model, as a sink sees it.

    ``cols`` (lb, ub, vtype), ``obj`` (col, coeff), ``rows`` (row, sense, rhs)
    and ``matrix`` in CSR: ``(col, coeff)`` in row-major order, with
    ``row_starts[r] : row_starts[r + 1]`` the half-open span row ``r`` owns.
    The objective constant lives outside the frames, having no column to
    attach to.

    ``col`` and ``row`` are dense ``0..n-1``, so they *are* the solver's own
    indices and no sink builds a mapping. **``cols`` carries no ``col`` and
    ``matrix`` no ``row``**: a ``cols`` row's position is its index and a
    matrix entry's row is where it sits between two starts, which is what both
    solvers' matrix APIs take — where a row label per nonzero would hold
    8 bytes an entry for the model's lifetime. :meth:`matrix_block` spells them
    back out for the one consumer that renders them. ``obj`` keeps its ``col``,
    being genuinely sparse (0.71 of ``cols`` on `transport`).
    """

    cols: pl.DataFrame
    obj: pl.DataFrame
    rows: pl.DataFrame
    matrix: pl.DataFrame
    row_starts: npt.NDArray[np.int64]
    column_count: int
    row_count: int
    objective_sense: str
    objective_constant: float

    def _spans(self, budget: int | None) -> Iterator[tuple[int, int]]:
        """The row ranges a block reader walks — one rule, for both of them.

        Width is the average row, since a reader pays in nonzeros: 100k rows is
        900k entries in one model and 10M in another. There is deliberately no
        row-counted twin to reach for by mistake.

        ``budget=None`` is one span, and a real answer: whether splitting pays
        is a property of the API being fed, not the model. HiGHS takes a chunk
        at a time and its budget bounds the temporary; Gurobi's ``addMConstr``
        charges per *model column* per call whatever the block holds, so
        splitting a matrix into many blocks costs it dearly (#434).

        Private, so no caller can pair spans and entries that disagree.
        """
        if budget is None:
            return iter([(0, self.row_count)])
        return chunking.ranges(self.row_count, budget, self.matrix.height / max(1, self.row_count))

    def _span(self, lo: int, hi: int) -> pl.DataFrame:
        """The matrix entries rows ``[lo, hi)`` own — the CSR arithmetic, once.

        Both block readers slice through here, so how a span is located, and
        the half-open ``hi`` bound, cannot drift between them.
        """
        first = int(self.row_starts[lo])
        return self.matrix.slice(first, int(self.row_starts[hi]) - first)

    def col_chunks(self, budget: int) -> Iterator[tuple[int, int]]:
        """Column ranges of roughly ``budget`` columns each.

        Width 1: a column *is* one row of the batch a sink hands over, stated
        rather than assumed (:mod:`~lpspec.relational.chunking`).
        """
        return chunking.ranges(self.column_count, budget, 1.0)

    def dense_columns(self, infinity: float) -> DenseColumns:
        """``(lb, ub, cost, integral)`` as numpy vectors over the solver's index.

        *infinity* is the solver's own spelling of an absent bound — the one
        thing the two disagree on — so it is asked for and the vectors come
        back ready to hand over unedited.

        ``cols`` already arrives one row per column in ``col`` order, so its
        three vectors are the frame's own. Only ``cost`` is scattered, ``obj``
        being genuinely sparse: a variable in no objective term is left free
        rather than holding whatever the allocator returned.

        The bound vectors are rewritten with ``copy=True``, being views of the
        frame — in place, an infinity would edit the built model to suit
        whichever solver asked last.

        **Nothing textual crosses into numpy**: a polars ``String`` converts by
        boxing every value as a Python object, so the test against
        ``'continuous'`` is made in polars and only its answer crosses — an
        order of magnitude apart at the top of the ladder (#418).
        """
        import numpy as np

        count = self.column_count
        lb = np.nan_to_num(self.cols['lb'].to_numpy(), copy=True, neginf=-infinity, posinf=infinity)
        ub = np.nan_to_num(self.cols['ub'].to_numpy(), copy=True, neginf=-infinity, posinf=infinity)
        integral = self.cols.select(pl.col('vtype') != 'continuous').to_series().to_numpy()
        cost = _scattered(count, self.obj['col'].to_numpy(), self.obj['coeff'].to_numpy(), 0.0)
        return lb, ub, cost, integral

    def dense_rows(self, infinity: float) -> DenseRows:
        """``(sense, rhs)`` as numpy vectors over the solver's row index.

        The row half of :meth:`dense_columns`: a chunk of rows is a slice
        rather than a search. Sorting and filtering the row frame once per
        chunk read the same 6M rows nine times over on `fleet/l`.

        It stops at the sense because that is where the two solvers part —
        HiGHS wants ``lower``/``upper``, Gurobi a comparison and right-hand
        side, both this pair spelled differently. A row with no entry gets a
        comparison nothing can fail (``>=`` against ``-infinity``) rather than
        the ``== 0`` that would be an equality the model never stated.
        """
        sided = self.rows.select(
            'row',
            pl.col('sense').replace_strict(SENSE_CODES, return_dtype=pl.UInt8).alias('op'),
            'rhs',
        )
        at = sided['row'].to_numpy()
        sense = _scattered(self.row_count, at, sided['op'].to_numpy(), SENSE_CODES['>='])
        rhs = _scattered(self.row_count, at, sided['rhs'].to_numpy(), -infinity)
        return sense, rhs

    def row_blocks(self, budget: int | None) -> Iterator[RowBlock]:
        """Each chunk of rows with the matrix entries it owns — a solver's reader.

        A chunk is a ``slice``: ``row_starts`` already says where every row's
        entries sit, so nothing is sorted and nothing is searched.

        Yields:
            ``(lo, hi, entries, starts)`` for rows ``[lo, hi)``, where
            ``starts`` is each row's offset within the block — what both
            solvers' matrix APIs ask for. A row with no entries takes the
            next row's offset, and so occupies no span.
        """
        for lo, hi in self._spans(budget):
            yield lo, hi, self._span(lo, hi), self.row_starts[lo:hi] - self.row_starts[lo]

    def matrix_block(self, lo: int, hi: int) -> pl.DataFrame:
        """Rows ``[lo, hi)`` of the matrix with their ``row`` labels spelled out.

        The adjoint of what CSR compressed — ``np.repeat`` walks the start
        offsets back into one label per entry — at the cost of one label column
        per *block*, not per model.
        """
        import numpy as np

        labels = np.repeat(np.arange(lo, hi, dtype=np.int64), np.diff(self.row_starts[lo : hi + 1]))
        return self._span(lo, hi).with_columns(pl.Series('row', labels))

    def labeled_blocks(self, budget: int | None) -> Iterator[tuple[int, int, pl.DataFrame]]:
        """Each chunk of rows with its entries labeled — the LP writer's reader.

        One method per consumer shape — solvers take :meth:`row_blocks`, the
        writer this — so no caller pairs spans and entries that disagree.

        Yields:
            ``(lo, hi, entries)`` for rows ``[lo, hi)``, the entries labelled
            as :meth:`matrix_block` labels them.
        """
        for lo, hi in self._spans(budget):
            yield lo, hi, self.matrix_block(lo, hi)


def solver_vector(values: Any) -> pl.Series:
    """One quantity a solver produced, in its own index — every sink's read-back.

    A series rather than a ``(label, value)`` frame: the read-back takes a
    declaration's share by slicing, so an index column beside it is an
    ``arange`` nothing reads — 8 bytes a column for as long as the result is
    held. The argument that took ``col`` off ``cols`` (#433).
    """
    import numpy as np

    return pl.Series('value', np.asarray(values, dtype=np.float64))


def _scattered(count: int, at: Any, values: Any, absent: Any) -> Any:
    """*values* written at the label each one belongs to, *absent* elsewhere."""
    import numpy as np

    dense = np.full(count, absent, dtype=values.dtype)
    dense[at] = values
    return dense

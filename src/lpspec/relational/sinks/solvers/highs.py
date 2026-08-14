"""The ``highs`` solver: COO batches straight into HiGHS.

The default, and the only one whose dependency ships with the package. Columns
and rows arrive as numpy slices, in batches, with no float→text→parse round
trip — which is why this exists beside
:mod:`~lpspec.relational.sinks.writers.lp_file`.

**Nothing textual crosses into numpy**: a row's ``'<='`` becomes a
:data:`~lpspec.relational.sinks.tables.SENSE_CODES` byte before it is read
here, the rule
:meth:`~lpspec.relational.sinks.tables.ModelTables.dense_columns` measured.

``highspy`` is imported inside the function, being optional: importing this
module stays free for callers that only write LP files.

:class:`Highs` is the same hand-off held open — what a driver that re-solves
one model with new numbers uses, and where the warm basis lives.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lpspec.errors import LpspecError
from lpspec.relational.sinks.solvers.base import SolveAnswer, Solver
from lpspec.relational.sinks.tables import SENSE_CODES, solver_vector
from lpspec.relational.status import SolveStatus

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lpspec.relational.sinks.tables import ModelTables, RowVectors


#: Elements per hand-off chunk. No *build-side* pass batches any more, labels
#: having become positional, so this is the sink's own budget rather than a
#: copy of a build knob — as the LP writer's ``EMIT_BUDGET`` is its. Spent
#: through
#: :mod:`~lpspec.relational.chunking`, which asks a caller to state what one
#: unit costs: a column is one element, a constraint row is as many as it has
#: nonzeros.
#:
#: Deliberately small. Both columns and rows are numpy slices, so more chunks
#: cost almost nothing and only residency scales with the budget — where an
#: engine whose every chunk re-ran an ordered query would want the opposite. A
#: wider budget buys a fraction of a second on a hand-off that precedes a
#: minute of simplex, and pays for it in a large fraction of the invariant this
#: budget exists to hold (#189).
HANDOFF_BUDGET = 100_000

#: HiGHS model status -> termination condition. Copied from linopy's own
#: ``Highs.CONDITION_MAP``; ``tests/test_solve_status.py`` asserts it still
#: matches, so a HiGHS release that adds a status shows up as a failure here
#: rather than as a silent ``unknown``.
_CONDITION_OF_HIGHS_STATUS = {
    'kNotset': 'unknown',
    'kLoadError': 'internal_solver_error',
    'kModelError': 'internal_solver_error',
    'kPresolveError': 'internal_solver_error',
    'kSolveError': 'internal_solver_error',
    'kPostsolveError': 'internal_solver_error',
    'kModelEmpty': 'unknown',
    'kMemoryLimit': 'resource_interrupt',
    'kOptimal': 'optimal',
    'kInfeasible': 'infeasible',
    'kUnboundedOrInfeasible': 'infeasible_or_unbounded',
    'kUnbounded': 'unbounded',
    'kObjectiveBound': 'terminated_by_limit',
    'kObjectiveTarget': 'terminated_by_limit',
    'kTimeLimit': 'time_limit',
    'kIterationLimit': 'iteration_limit',
    'kSolutionLimit': 'terminated_by_limit',
    'kInterrupt': 'user_interrupt',
    'kUnknown': 'unknown',
}


def build_highs(
    model: ModelTables,
    batch_rows: int | None = None,
    solver_options: Mapping[str, Any] | None = None,
) -> Any:
    """Load the model into a :class:`highspy.Highs` and stop there.

    The hand-off without the simplex, which is the same work whoever filled the
    model — so a measurement including it says nothing about the lane that
    filled it. `bench/` ends here, as linopy's ``Model.to_highspy()`` does on
    that side.

    ``batch_rows`` is the budget in *elements*, spent through
    :mod:`~lpspec.relational.chunking`; the parameter stays so tests can force
    ragged chunks.
    """
    import highspy
    import numpy as np

    batch = HANDOFF_BUDGET if batch_rows is None else batch_rows
    inf = highspy.kHighsInf
    h = highspy.Highs()
    h.setOptionValue('output_flag', False)
    for option, value in (solver_options or {}).items():
        h.setOptionValue(option, value)

    empty_i = np.empty(0, dtype=np.int32)
    empty_f = np.empty(0, dtype=np.float64)
    cols = model.dense_columns(inf)
    for lo, hi in model.col_chunks(batch):
        _loaded(
            h,
            h.addCols(hi - lo, cols.cost[lo:hi], cols.lb[lo:hi], cols.ub[lo:hi], 0, empty_i, empty_i, empty_f),
            'a batch of columns',
        )
        noncontinuous = np.flatnonzero(cols.integral[lo:hi] | cols.semi_continuous[lo:hi]).astype(np.int32) + np.int32(
            lo
        )
        if len(noncontinuous):
            integrality = np.where(
                cols.semi_continuous[noncontinuous],
                np.uint8(int(highspy.HighsVarType.kSemiContinuous)),
                np.uint8(int(highspy.HighsVarType.kInteger)),
            )
            h.changeColsIntegrality(len(noncontinuous), noncontinuous, integrality)

    rlb, rub = _row_bounds(model.dense_rows(inf), inf)
    for block in model.row_blocks(batch):
        _loaded(
            h,
            h.addRows(
                block.height,
                rlb[block.lo : block.hi],
                rub[block.lo : block.hi],
                block.entries.height,
                block.starts.astype(np.int32),
                block.entries['col'].to_numpy().astype(np.int32, copy=False),
                block.entries['coeff'].to_numpy(),
            ),
            'a batch of rows',
        )

    if model.objective_sense == 'max':
        h.changeObjectiveSense(highspy.ObjSense.kMaximize)
    return h


class Highs(Solver):
    """HiGHS, holding one model — :class:`Solver`'s member for the default sink.

    What makes an iterative driver cheap. The second solve of a rebound model
    changes bounds, costs and right-hand sides on the model HiGHS already
    holds and starts from the basis the last solve ended on, where loading
    again would hand over the matrix a second time and start cold.

    **Values are re-pushed, never diffed** — the previous model is *gone* by
    the time the new one exists, so there is nothing held to diff against;
    the trade is argued once, in ``../README.md``. Pushing the whole vectors
    costs a pass over the columns and the rows, against the matrix pass that
    loading would cost.
    """

    #: The loaded model. Declared rather than inferred, ``close`` dropping it.
    _handle: Any

    requires = ('highspy',)
    unavailable_message = 'highspy ships with lpspec, so a build without it is broken rather than missing an extra'

    #: HiGHS has no SOS concept at all — its own LP reader refuses the section
    #: — so a set arrives here already written as binaries and linking rows.
    sos = 'reformulated'

    def _load(self, model: ModelTables, batch_rows: int | None) -> None:
        self._handle = build_highs(model, batch_rows, self._options)

    def push(self, model: ModelTables) -> None:
        """*model*'s bounds, costs and right-hand sides onto the loaded model.

        Everything a rebind may change without moving a label. The index
        vectors are built here rather than held, an ``arange`` being cheaper
        to make than to keep.
        """
        import highspy
        import numpy as np

        inf = highspy.kHighsInf
        cols = model.dense_columns(inf)
        columns = np.arange(model.column_count, dtype=np.int32)
        _loaded(self._handle, self._handle.changeColsCost(model.column_count, columns, cols.cost), 'new costs')
        _loaded(
            self._handle, self._handle.changeColsBounds(model.column_count, columns, cols.lb, cols.ub), 'new bounds'
        )

        rows = np.arange(model.row_count, dtype=np.int32)
        rlb, rub = _row_bounds(model.dense_rows(inf), inf)
        _loaded(self._handle, self._handle.changeRowsBounds(model.row_count, rows, rlb, rub), 'new right-hand sides')

    def _run(self, model: ModelTables) -> SolveAnswer:
        import highspy

        self._handle.run()
        status = _status_of(self._handle, highspy)
        if not status.is_readable:
            return SolveAnswer.unreadable(status)

        objective = self._handle.getInfo().objective_function_value + model.objective_constant
        solution = self._handle.getSolution()
        primal = solver_vector(solution.col_value)
        dual = solver_vector(solution.row_dual) if solution.dual_valid else None
        return SolveAnswer(status, objective, primal, dual)

    def close(self) -> None:
        """Release the loaded model. Idempotent."""
        if self._handle is not None:
            self._handle.clear()
        self._handle = None


def _row_bounds(rows: RowVectors, inf: float) -> tuple[Any, Any]:
    """HiGHS's ``(lower, upper)`` spelling of a sense code and right-hand side.

    The one rule for it, asked by the load and the push alike, so the two
    cannot drift: an inequality is open on the side its sense does not bound.
    """
    import numpy as np

    return (
        np.where(rows.sense == SENSE_CODES['<='], -inf, rows.rhs),
        np.where(rows.sense == SENSE_CODES['>='], inf, rows.rhs),
    )


def _loaded(h: Any, status: Any, what: str) -> None:
    """Raise unless the solver accepted the hand-off.

    HiGHS reports a rejected call by return value and carries on with whatever
    it had, so an unchecked call turns a malformed hand-off into a confident
    answer to a different problem — an unconstrained one, if it was the rows.

    Raises:
        LpspecError: If the batch was refused.
    """
    import highspy

    if status == highspy.HighsStatus.kError:
        raise LpspecError(
            f'the solver refused {what}: {h.modelStatusToString(h.getModelStatus())!r}. '
            f'The model it holds is not the one handed over, so any answer would describe a '
            f'different one. This is an engine bug rather than a problem with the model — '
            f'please report it.'
        )


def _status_of(h: Any, highspy: Any) -> SolveStatus:
    """What the solve concluded, on both axes.

    ``has_primal`` is the solver's own answer to "is there anything here",
    which the termination condition does not give: a run stopped at a time
    limit may or may not have found an incumbent.
    """
    model_status = h.getModelStatus()
    return SolveStatus(
        termination_condition=_CONDITION_OF_HIGHS_STATUS.get(str(model_status).rsplit('.', 1)[-1], 'unknown'),
        solver_wording=h.modelStatusToString(model_status),
        has_primal=h.getInfo().primal_solution_status == int(highspy.SolutionStatus.kSolutionStatusFeasible),
    )

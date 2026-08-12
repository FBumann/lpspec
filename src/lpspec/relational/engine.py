"""What an engine is, and everything an engine does not have to write.

`plan.py` is what an engine consumes and `sinks/tables.py` is what it produces.
This is the third side: given those two, most of an executor's surface is not
engine work at all. Sinking to an LP file, handing the model to HiGHS, and
slicing a solver's answer back onto coordinates are all written against
`ModelTables` and the label frames — never against how either was filled.

So they live here once. An engine supplies four things:

- `build(program, sources)` — bind and construct
- `_tables()` — the four frames plus the scalars
- `_variables` / `_constraints` — `(dims…, var_label)` and `(dims…, row)` per
  declaration, and `_blocks`, the contiguous run of labels each was given —
  which together are what a solution is read back through
- `_program` — the plan it built, for the dims a read-back projects to

and inherits the rest. That split is the actual claim `engines/` makes, and it
is why a second engine is a compiler and an assembler rather than a whole lane.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import polars as pl

from lpspec.errors import LpspecError
from lpspec.relational import sinks
from lpspec.relational.result import Result

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lpspec.relational import plan
    from lpspec.relational.binding import BoundSources


class Engine(ABC):
    """A relational LP builder: plan in, `ModelTables` out.

    The label registries are declared here rather than in each engine because
    the read-back below is written against them. They are polars frames on both
    engines: a label frame is `(dims…, label)` and nothing about it is engine
    work — an engine that holds its labels elsewhere materialises them here,
    which is the price of not writing this file twice.
    """

    _program: plan.Program | None

    #: ``name -> (first label, how many)``. Every labelling path on either
    #: engine hands a declaration a *contiguous, dense* run of labels, so a
    #: declaration's share of a solver vector is a slice of it — which is what
    #: :meth:`_read_back` relies on instead of a join.
    _blocks: dict[str, tuple[int, int]]

    #: ``name -> rows not built``, because every term they had vanished. Empty
    #: for a model whose every declared row reached the solver. Filled by the
    #: engine, read by :meth:`omissions`.
    _omitted: dict[str, int]

    #: What `bind` returned, for the dtypes :meth:`_read_back` undoes.
    _bound: BoundSources | None

    #: How many columns and rows the built model has. An engine counts them as
    #: it labels; :func:`_spanning` reads them to refuse a solver vector that
    #: describes a different model.
    _n_cols: int
    _n_rows: int

    @property
    @abstractmethod
    def _variables(self) -> Mapping[str, pl.LazyFrame]:
        """Per-variable `(dims…, var_label)`. Read-only here; an engine owns the storage."""

    @property
    @abstractmethod
    def _constraints(self) -> Mapping[str, pl.LazyFrame]:
        """Per-constraint `(dims…, row)`. Read-only here; an engine owns the storage."""

    @abstractmethod
    def build(self, program: plan.Program, sources: Mapping[str, Any]) -> None:
        """Bind *sources* and build every declaration. Raises rather than half-building."""

    @abstractmethod
    def _tables(self) -> sinks.ModelTables:
        """The built model as `cols`, `obj`, `rows`, `matrix` plus its scalars."""

    @abstractmethod
    def close(self) -> None:
        """Drop the built model. Optional for a caller — see `Result`."""

    # -- sinks: written against ModelTables, so neither engine owns them ---

    def omissions(self) -> pl.DataFrame:
        """``(constraint, rows_not_built)`` — every row that lost all its terms.

        A row with no variable terms is not built (SPEC §6). Counts, not
        coordinates; empty for a model whose every declared row reached the
        solver.
        """
        return pl.DataFrame(
            {'constraint': list(self._omitted), 'rows_not_built': list(self._omitted.values())},
            schema={'constraint': pl.String, 'rows_not_built': pl.UInt32},
        )

    def write(self, path: str | Path) -> None:
        """Stream the built model to *path*, in the format its suffix names.

        Raises:
            ValueError: A suffix nothing writes.
            NotImplementedError: A format that is planned and not here yet.
        """
        path = Path(path)
        sinks.writer(path.suffix.lower())(self._tables(), path)

    def solve(
        self,
        batch_rows: int | None = None,
        solver_options: Mapping[str, Any] | None = None,
        solver_name: str = 'highs',
    ) -> Result:
        """Hand the built model to a solver and solve it.

        Args:
            batch_rows: The hand-off budget in elements, defaulting to the
                sink's own
                (:data:`~lpspec.relational.sinks.solvers.highs.HANDOFF_BUDGET`).
            solver_options: Forwarded to the solver verbatim, in its own
                vocabulary (``{'time_limit': 60, 'mip_rel_gap': 0.01}``).
            solver_name: ``highs``, which ships with the package, or
                ``gurobi``, which needs the ``[gurobi]`` extra.

        Returns:
            The solution, holding this executor.
        """
        status, objective, primal, dual = sinks.solver(solver_name)(self._tables(), batch_rows, solver_options)
        _spanning(solver_name, 'primal', primal, self._n_cols)
        _spanning(solver_name, 'dual', dual, self._n_rows)
        return Result(
            _status=status,
            _objective=objective,
            _executor=self,
            _primal_values=primal,
            _dual_values=dual,
        )

    # -- read-back: a slice, and labels are frames on every engine ---------

    def _solution_frame(self, name: str, values: pl.Series | None) -> pl.LazyFrame:
        """The tidy solution of variable *name*: ``(dims…, value)``.

        A slice, never a dense array and never a join. *values* is the solver's
        column vector, held by the :class:`Result` that asks — labels are the
        build's and shared, values are one solve's and are not.

        **In label order**: a label *is* row-major position in the coordinate
        product, so the caller gets the model's own order rather than the order
        a hash join happened to finish in.
        """
        assert self._program is not None
        assert values is not None, 'no solve has stored a primal'
        return self._read_back(name, self._variables[name], self._program.variable(name).dims, values)

    def _read_back(
        self,
        name: str,
        coordinates: pl.LazyFrame,
        dims: tuple[str, ...],
        values: pl.Series,
    ) -> pl.LazyFrame:
        """One declaration's coordinates in label order, beside its values.

        **The order is not re-established here, because it was never lost**:
        :func:`labels.frame` numbers a sorted frame and hands back a
        label-ascending one.

        The declaration owns a contiguous, dense run of labels
        (:attr:`_blocks`) and the solver's vector is positional in the same
        index, so coordinates and values line up by construction. The slice is
        attached as a column rather than concatenated as a frame, so a
        mismatched length raises instead of padding with nulls.

        **Dim columns leave in ``String``**, where the build holds them as
        ``pl.Enum`` (#541). That encoding is internal and every gram of its win
        is upstream of here, but a returned frame is something a caller *joins
        against their own data* — and polars refuses ``Enum`` against
        ``String`` with a message about dtypes that names nothing about the
        cause. Two frames of one sweep will not even concatenate when their
        slices bound different members.

        The cast sits inside this projection rather than after it, so the
        string column is produced once instead of widened from an Enum that
        also exists, which is cheaper in both wall and peak (#593). Declaration
        order is the *row* order and survives, never having been the dtype's to
        carry.
        """
        start, height = self._blocks[name]
        labelled = coordinates.select(*dims).with_columns(values.slice(start, height))
        return labelled.with_columns(pl.col(d).cast(pl.String) for d in self._string_dims(dims))

    def _string_dims(self, dims: tuple[str, ...]) -> list[str]:
        """Those of *dims* the binder encoded as ``Enum`` — its string ones."""
        assert self._bound is not None, 'build() has not run'
        return [d for d in dims if self._bound.is_enum_encoded(d)]

    def _primal(self, name: str, values: pl.Series | None) -> pl.DataFrame:
        return self._solution_frame(name, values).collect(engine='streaming')

    def _dual(self, name: str, values: pl.Series) -> pl.DataFrame:
        """:meth:`_solution_frame` against row labels instead of column ones.

        Ordered and sliced the same way, for the same reason — a constraint
        row's label is its position in that constraint's coordinate product.
        """
        assert self._program is not None
        dims = self._program.constraint(name).dims
        return self._read_back(name, self._constraints[name], dims, values).collect(engine='streaming')

    def _no_duals_reason(self, termination_condition: str) -> str:
        """Why a solve that *did* leave values still has no duals.

        Integrality is decidable from the program, and naming the variable is
        actionable where "the solver reported none" is not.
        """
        assert self._program is not None
        discrete = sorted(v.name for v in self._program.variables if v.variable_type != 'continuous')
        if discrete:
            names = ', '.join(f"'{n}'" for n in discrete)
            return (
                f'duals are undefined for a mixed-integer model: {names} '
                f'{"is" if len(discrete) == 1 else "are"} not continuous. '
                f'Drop the integrality to price the LP relaxation instead.'
            )
        return (
            f'the solver returned no dual solution, though the solve terminated '
            f'{termination_condition!r}. Duals come from a simplex basis, which a '
            f'run stopped short of one does not have.'
        )

    def _solution_to_parquet(self, directory: Path, values: pl.Series | None) -> dict[str, Path]:
        assert self._program is not None
        directory.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}
        for v in self._program.variables:
            out = directory / f'{v.name}.parquet'
            self._solution_frame(v.name, values).sink_parquet(out)
            written[v.name] = out
        return written

    def __enter__(self) -> Engine:
        return self

    def __exit__(self, *exc: object) -> Literal[False]:
        self.close()
        return False


def _spanning(solver: str, quantity: str, values: pl.Series | None, expected: int) -> None:
    """Refuse a solver vector that does not span the model.

    Reading a solution back is positional, so a vector of the wrong length is
    an answer about a *different* model rather than a short answer about this
    one. Checked here, where the solver hands it over, rather than where it is
    read: the objective comes back from the solver directly, so a `Result`
    built on a broken vector would report a plausible number and only fail if
    someone asked for a coordinate.

    Here rather than in an executor because `solve` is here — the check belongs
    to the hand-off, and both engines hand off through this one.

    ``None`` is not a wrong length. A mixed-integer model has no duals at all,
    and neither does a run stopped short of a simplex basis.
    """
    if values is not None and len(values) != expected:
        raise LpspecError(
            f'{solver} returned {len(values)} {quantity} values for a model with {expected}. '
            f'Reading a solution back is positional, so a vector that does not span the model '
            f'describes a different one. This is an engine bug rather than a problem with the '
            f'model — please report it.'
        )

"""What a solve returned — the caller's end of the relational lane.

The object ``lps.solve`` hands back, so it is the one piece of this subpackage a
reader meets without going looking. It lives beside the executor rather than in
it because the two answer different questions: the executor *builds* a model,
and this *reads* one — the accessors are joins against the label frames plus
whatever vector the solver left.

Named for linopy's envelope (``Result`` = status + solution + report) rather
than for our own decomposition, because our audience arrives from linopy and a
second vocabulary for one fact is a tax on every one of them
(docs/ARCHITECTURE.md, "Where a concept is already linopy's").
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from lpspec.errors import LpspecError, NoSolutionError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import pandas as pd
    import polars as pl
    import xarray as xr

    from lpspec.relational.engine import Engine
    from lpspec.relational.status import SolveStatus


def tidy_to_pandas(frame: pl.DataFrame) -> pd.DataFrame:
    """A tidy polars frame as pandas, column by column.

    The three bridges below are shared by :class:`Result` and
    :class:`~lpspec.strategy.Runs`, which differ only in where the tidy frame
    comes from — a sweep's is the same frame one slice wider. Built column by
    column because polars' own ``to_pandas`` reaches for pyarrow; pandas itself
    ships with the ``[linopy]`` extra.
    """
    import pandas as pd

    return pd.DataFrame({column: frame[column].to_numpy() for column in frame.columns})


def tidy_to_dataarray(frame: pd.DataFrame, name: str) -> xr.DataArray:
    """The same, labelled by its non-``value`` columns.

    A scalar declaration has none and comes back 0-dimensional.
    """
    dims = [column for column in frame.columns if column != 'value']
    if not dims:
        return frame['value'].to_xarray().rename(name)
    return frame.set_index(dims).to_xarray()['value'].rename(name)


def tidy_to_dataset(names: Sequence[str], one: Callable[[str], xr.DataArray]) -> xr.Dataset:
    """*names* as one dataset, each array built by *one*."""
    first, *rest = names
    dataset = one(first).to_dataset(name=first)
    for name in rest:
        dataset[name] = one(name)
    return dataset


@dataclass
class Result:
    """What a solve returned — the outcome, and access to any values.

    Returned whatever the solve concluded: test :attr:`has_primal` before
    reading values, or catch :class:`~lpspec.errors.NoSolutionError`. The
    values are this result's own, so a later solve on the same model does not
    rewrite them, and there is no lifetime to manage — :meth:`close` releases
    a large model early, and nothing breaks without it.
    """

    _status: SolveStatus
    _objective: float
    _executor: Engine
    _primal_values: pl.Series | None = None
    _dual_values: pl.Series | None = None
    _closed: bool = False

    @property
    def status(self) -> str:
        """Coarse outcome: ``ok`` / ``warning`` / ``error`` / ``aborted`` / ``unknown``."""
        return self._status.status

    @property
    def termination_condition(self) -> str:
        """What the solver said — ``optimal``, ``infeasible``, ``time_limit`` and so on."""
        return self._status.termination_condition

    @property
    def is_ok(self) -> bool:
        """The linopy rollup: not an error, an abort or a refusal."""
        return self._status.is_ok

    @property
    def has_primal(self) -> bool:
        """Whether there are values to read — what the accessors gate on.

        Narrower than :attr:`is_ok`: a run stopped at a time limit before any
        incumbent is ``ok`` with nothing to read.
        """
        return self._status.is_readable

    @property
    def objective(self) -> float:
        """The objective value, or ``nan`` when there is no solution."""
        return self._objective

    def _require_solution(self, what: str) -> None:
        if self._closed:
            raise LpspecError(
                f'cannot read {what}: this result was closed, and closing releases the model the '
                f'readers join against. Frames already read stay valid — they are their own data — so '
                f'read what you need before close(), or drop the `with` and close when you are done.'
            )
        if self._status.is_readable:
            return
        raise NoSolutionError(
            f'cannot read {what}: the solve terminated {self.termination_condition!r} '
            f'({self._status.solver_wording}), so there are no values to read. Test '
            f'`has_primal` first. This raises rather than returning, because the solver '
            f'hands back a full-length vector of zeros either way and it is '
            f'indistinguishable from an answer.'
        )

    def primal(self, name: str) -> pl.DataFrame:
        """The tidy solution of variable *name* — ``(dims…, value)``.

        Rows come back in label order, row-major over the variable's coordinate
        product, so two reads and two runs agree.

        Raises:
            NoSolutionError: The solve left no values to read.
            KeyError: No variable is called *name*.
        """
        self._require_solution(f"the primal of '{name}'")
        return self._executor._primal(name, self._primal_values)

    def dual(self, name: str) -> pl.DataFrame:
        """Shadow prices of constraint *name* — ``(dims…, value)``.

        :meth:`primal`'s shape and order, over constraint rows.

        Raises:
            NoSolutionError: The solve left no values at all.
            LpspecError: It left primals but no duals — an integer variable
                makes them undefined.
            KeyError: No constraint is called *name*.
        """
        self._require_solution(f"the dual of '{name}'")
        if self._dual_values is None:
            raise LpspecError(self._executor._no_duals_reason(self.termination_condition))
        return self._executor._dual(name, self._dual_values)

    def to_pandas(self, name: str) -> pd.DataFrame:
        """:meth:`primal` as a tidy :class:`pandas.DataFrame`."""
        self._require_solution(f"the primal of '{name}'")
        return tidy_to_pandas(self._executor._primal(name, self._primal_values))

    def to_dataarray(self, name: str) -> xr.DataArray:
        """:meth:`primal` as a labelled :class:`xarray.DataArray`.

        Dense over the variable's dims: a masked coordinate comes back NaN.
        """
        return tidy_to_dataarray(self.to_pandas(name), name)

    def to_dataset(self, *names: str) -> xr.Dataset:
        """The named variables as one :class:`xarray.Dataset`; all by default.

        Each arrives dense over its own dims, all at once — on a large model
        name the few you need, or use :meth:`to_parquet`.
        """
        assert self._executor._program is not None
        wanted = names or tuple(v.name for v in self._executor._program.variables)
        return tidy_to_dataset(wanted, self.to_dataarray)

    def to_parquet(self, directory: str | Path) -> dict[str, Path]:
        """Write one parquet file per variable into *directory*.

        Streamed to disk in :meth:`primal`'s order, so the same model and data
        write the same bytes.

        Returns:
            Each variable's name, mapped to the file it was written to.
        """
        self._require_solution('the solution')
        return self._executor._solution_to_parquet(Path(directory), self._primal_values)

    def close(self) -> None:
        """Release the built model and this result's values. Optional.

        Frames already read stay valid; the readers stop working.
        """
        self._primal_values = self._dual_values = None
        self._closed = True
        self._executor.close()

    def __enter__(self) -> Result:
        return self

    def __exit__(self, *exc: object) -> Literal[False]:
        self.close()
        return False

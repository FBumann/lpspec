"""What a caller's ``sources`` become: the frames the engine reads by name.

The boundary between *what was passed in* — parquet paths, any table exposing
the Arrow PyCapsule protocol — and *what the query is written against*. Binding
is the only phase that touches a caller's data; everything downstream reads
:class:`BoundSources` and nothing else.

**It is frozen, and that is the point.** Written once by the passes below,
then read to construct the compiler and the labeller. Holding it as a value
rather than as four dicts on the executor is what separates it from the one
registry that is deliberately *live* — the variable frames, which appear as
declarations build and which a constraint compiled afterwards has to see.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

from lpspec.errors import DataError
from lpspec.relational import data_validation
from lpspec.relational.frames import as_frame

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lpspec.relational import plan

#: Scratch column carrying a source row's position while first-occurrence
#: order is computed. The spaces make it unrepresentable as a declared name, so
#: it cannot collide with a column the caller's index already has.
_ROW_POSITION = '__row position__'


@dataclass(frozen=True)
class BoundSources:
    """The data a program is built against, after binding.

    ``parameters`` are tidy ``(dims…, value)``; ``dimensions`` are
    ``(val, ord, coordinates…)``. The other two are answers *read off* the data
    that the query needs and cannot re-derive cheaply: ``sum`` over an absent
    dim scales by that dim's size, and ``defined`` on a boolean parameter tests
    the value rather than its finiteness.

    ``cardinality`` is a dimension frame's height, cached here because deriving
    it later means collecting the frame again.
    """

    parameters: Mapping[str, pl.LazyFrame]
    dimensions: Mapping[str, pl.LazyFrame]
    cardinality: Mapping[str, int]
    boolean_parameters: frozenset[str]

    def is_enum_encoded(self, dim: str) -> bool:
        """Whether :meth:`_Binder.encode_dimensions` gave *dim* an ``Enum``.

        Asked by both consumers of the encoding — the compiler reads the
        physical code as an ordinal for free, and the executor casts back to
        ``String`` on the way out — and answered here, where the encoding is
        decided, so the two cannot come to disagree about which dims have one.
        """
        return self.dimensions[dim].collect_schema()['val'] == pl.Enum


def bind(program: plan.Program, sources: Mapping[str, Any]) -> BoundSources:
    """Adapt *sources* to the frames *program* is written against.

    Four passes, and the order is load-bearing. Dimensions with an index of
    their own come first, so a parameter's labels are checked in the pass that
    binds it rather than a second one over the same rows. The remaining
    dimensions are *derived* from those parameters and cannot be built until
    they exist — and have no strangers to find, their labels being the union of
    what arrived. Encoding comes last: a dimension's ``Enum`` is built from its
    labels, which a derived dimension has only once the parameters have bound.

    Raises:
        DataError: A source missing, unreadable, or not carrying what its
            declaration needs.
    """
    binder = _Binder(program, sources)
    binder.sourced_dimensions()
    for p in program.parameters:
        binder.parameter(p)
    binder.remaining_dimensions()
    binder.encode_dimensions()
    return BoundSources(
        parameters=binder.parameters,
        dimensions=binder.dimensions,
        cardinality=binder.cardinality,
        boolean_parameters=frozenset(binder.boolean),
    )


class _Binder:
    """The three passes' shared accumulator; discarded once :func:`bind` returns."""

    def __init__(self, program: plan.Program, sources: Mapping[str, Any]) -> None:
        self.program = program
        self.sources = sources
        self.parameters: dict[str, pl.LazyFrame] = {}
        self.dimensions: dict[str, pl.LazyFrame] = {}
        self.cardinality: dict[str, int] = {}
        self.boolean: set[str] = set()

    # -- parameters --------------------------------------------------------

    def parameter(self, p: plan.ParameterDeclaration) -> None:
        """Bind one parameter's source and register it as a tidy frame.

        The one collect in this file on the streaming engine, its result being
        the one that is model-sized: switching every collect costs a double-digit
        percentage of wall on a small join-heavy model, to save the peak this one
        saves alone (#370).

        Validation runs before the string cast — a dictionary-encoded column
        compares on its codes, and widening to strings first doubles the check.
        """
        if p.name not in self.sources:
            raise DataError(f"no source bound for parameter '{p.name}'")
        frame = self._read(
            self.sources[p.name],
            f"source for parameter '{p.name}' must be a parquet path or a table polars can "
            f'read — polars, pyarrow, pandas (got {type(self.sources[p.name]).__name__})',
        )
        wanted = [*p.dims, 'value']
        available = frame.collect_schema().names()
        missing = set(wanted) - set(available)
        if missing:
            raise DataError(
                f"source for parameter '{p.name}' is missing columns {sorted(missing)} "
                f"(need dims {list(p.dims)} plus 'value'; has {available}). Rename them to "
                f'the declared dims, or drop the index names to bind positionally.'
            )
        frame = frame.select(wanted).collect(engine='streaming').lazy()
        data_validation.check_one_row_per_coordinate(p, frame, self.dimensions)
        frame = _plain_strings(frame, p.dims)
        if frame.collect_schema()['value'] == pl.Boolean:
            self.boolean.add(p.name)
        self.parameters[p.name] = frame

    def _read(self, source: Any, unreadable: str) -> pl.LazyFrame:
        """A caller's source as a lazy frame, or *unreadable* as a data error.

        The one place a path is told from a table: a path is scanned, so the
        engine reads it directly, and anything else has to be table-shaped.
        The message is the caller's because what the source was *for* — a
        parameter, a dimension's index — is what makes it actionable.
        """
        if isinstance(source, (str, Path)):
            return pl.scan_parquet(source)
        frame = as_frame(source)
        if frame is None:
            raise DataError(unreadable)
        return frame

    # -- dimensions --------------------------------------------------------

    def sourced_dimensions(self) -> None:
        """Every dimension carrying its own index, before any parameter binds."""
        for d in sorted(self._declared_dims()):
            if d in self.sources:
                self._register(d, self._explicit_frame(d, self.sources[d], self.program.dimension(d).carried))

    def remaining_dimensions(self) -> None:
        """Build every dimension's frame, then check its coordinates.

        A dimension with no explicit index has no declared order, so its labels
        are sorted. Dimensions already registered by :meth:`sourced_dimensions`
        are skipped. Containment runs once every frame exists: it stops a
        mistyped coordinate from vanishing in the join that places its terms,
        leaving a model that builds and solves without them.
        """
        dims = self._declared_dims()
        for d in sorted(dims):
            if d in self.dimensions:
                continue
            carried = self.program.dimension(d).carried
            if d in self.sources:
                table = self._explicit_frame(d, self.sources[d], carried)
            else:
                if carried:
                    raise DataError(
                        f"dimension '{d}' declares coordinates {carried} but has "
                        f"no index source. Pass one under key '{d}' (a parquet path or frame "
                        f'carrying columns {[d, *carried]}) — a coordinate cannot '
                        f'be inferred from the parameters that happen to use the dimension.'
                    )
                params = [p for p in self.program.parameters if d in p.dims]
                if not params:
                    raise DataError(
                        f"dimension '{d}' has no source: no parameter carries it and "
                        f"no explicit index was provided under key '{d}'"
                    )
                stacked = pl.concat([self.parameters[p.name].select(pl.col(d).alias('val')) for p in params])
                table = stacked.unique().sort('val').with_row_index('ord').with_columns(pl.col('ord').cast(pl.Int64))
            self._register(d, table)

        for d in sorted(dims):
            for cname, target in sorted(self.program.dimension(d).coordinates):
                if target not in self.cardinality:
                    raise DataError(
                        f"dimension '{d}' coordinate '{cname}' targets '{target}', which "
                        f'no declaration in this model uses, so it has no coordinate set '
                        f'to check against'
                    )
                data_validation.check_coordinate_containment(d, cname, target, self.dimensions)

    def _explicit_frame(self, d: str, source: Any, names: list[str]) -> pl.LazyFrame:
        """A dimension's ``(val, ord, coordinates…)`` from a caller's index.

        Ordinals follow the source's own order — a label's position is the row
        it first appears at — so a translation moves by position exactly as the
        eager lane does, even for string labels.

        Collected once, the frame being a scan: every pass over a lazy view
        re-reads the source (#273), and both the single-valued check and the
        grouping below read that one collect.
        """
        frame = self._read(
            source,
            f"explicit index for dimension '{d}' must be a table polars can read "
            f"with a '{d}' column, or a parquet path",
        )
        available = frame.collect_schema().names()
        if d not in available:
            raise DataError(
                f"explicit index for dimension '{d}' must be a table polars can read "
                f"with a '{d}' column, or a parquet path (has {available})"
            )
        missing = [c for c in names if c not in available]
        if missing:
            raise DataError(
                f"index for dimension '{d}' is missing declared coordinate column(s) {missing} (has {available})"
            )
        labelled = frame.select(d, *names).with_row_index(_ROW_POSITION).collect().lazy()
        data_validation.check_coordinates_single_valued(d, names, labelled)
        return (
            labelled.group_by(d)
            .agg(pl.col(_ROW_POSITION).min(), *(pl.col(c).first() for c in names))
            .sort(_ROW_POSITION)
            .with_row_index('ord')
            .select(pl.col(d).alias('val'), pl.col('ord').cast(pl.Int64), *names)
        )

    def _register(self, d: str, table: pl.LazyFrame) -> None:
        materialised = table.collect()
        self.dimensions[d] = materialised.lazy()
        self.cardinality[d] = materialised.height

    def encode_dimensions(self) -> None:
        """Every string dimension becomes an ``Enum`` over its labels, in ordinal order.

        One dictionary per dimension applied to every frame carrying it, so
        downstream joins meet ``Enum`` against ``Enum`` with equal categories by
        construction. A dim column then costs a code instead of a string for the
        model's lifetime, which shrinks the retained label frames and the emit
        alike for an encode that is cheap per row (#541).

        Running after every check is what makes the strict cast safe — each
        label was already probed against its dimension, so a failure here is an
        engine bug rather than a data error.
        """
        materialised = {d: table.collect() for d, table in self.dimensions.items()}
        enums = {d: pl.Enum(f['val']) for d, f in materialised.items() if f.schema['val'] == pl.String}
        if not enums:
            return
        for d, frame in materialised.items():
            casts = [pl.col('val').cast(enums[d])] if d in enums else []
            casts += [
                pl.col(cname).cast(enums[target])
                for cname, target in self.program.dimension(d).coordinates
                if target in enums
            ]
            if casts:
                self.dimensions[d] = frame.with_columns(casts).lazy()
        for p in self.program.parameters:
            casts = [pl.col(d).cast(enums[d]) for d in p.dims if d in enums]
            if casts:
                self.parameters[p.name] = self.parameters[p.name].collect().with_columns(casts).lazy()

    def _declared_dims(self) -> set[str]:
        dims: set[str] = set()
        for v in self.program.variables:
            dims.update(v.dims)
        for c in self.program.constraints:
            dims.update(c.dims)
        for p in self.program.parameters:
            dims.update(p.dims)
        return dims


def _plain_strings(frame: pl.LazyFrame, dims: tuple[str, ...]) -> pl.LazyFrame:
    """Dim columns as plain strings, whatever encoding the source used.

    A dictionary-encoded source (pandas ``Categorical``, dictionary parquet)
    carries a writer's own dictionary, and the label checks and the
    derived-dimension union need every arrival in one dtype before any
    dimension's own dictionary exists. So sources are decoded here, and
    :meth:`_Binder.encode_dimensions` re-encodes everything at once into the
    dimension's canonical ``Enum``.
    """
    categorical = [d for d, dtype in frame.collect_schema().items() if d in dims and dtype in (pl.Categorical, pl.Enum)]
    if not categorical:
        return frame
    return frame.with_columns(pl.col(d).cast(pl.String) for d in categorical)

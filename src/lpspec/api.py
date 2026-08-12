"""The runner: bind data to a YAML model and execute it. Not a modeling API.

Math is defined in YAML only — there is no Python API for constructing models,
and the logical plan is internal. Four verbs: ``check``, ``build`` (YAML +
sources → live executor), ``solve`` and ``write``.

This is the product path (docs/ARCHITECTURE.md): validated at load time,
lowered to the plan, executed relationally. linopy exists only in the optional
compatibility/oracle layer (``import lpspec.linopy``).

Example::

    import lpspec as lps

    result = lps.solve(
        'model.yaml',
        {'p_max': 'p_max.parquet', 'load': 'load.parquet'},
        coords={'snapshot': range(8760)},
    )
    result.objective
    result.primal('p')  # tidy polars.DataFrame (coords..., value)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lpspec.errors import LpspecWarning
from lpspec.language.validation import load_model
from lpspec.lowering import advice, lower_program
from lpspec.relational.engines import resolve as resolve_engine
from lpspec.relational.sinks import solver, writer
from lpspec.sources import tidy_sources

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lpspec.language.model import Model
    from lpspec.relational.engine import Engine
    from lpspec.relational.result import Result

#: Re-exported: parsing and validating a model is the *language's* job, and a
#: consumer that binds no data (``typeset``) must be able to reach it without
#: reaching the runner. Callers keep saying ``lps.load_model``.
__all__ = ['build', 'check', 'load_model', 'solve', 'write']


def check(model: str | Path | dict[str, Any] | Model) -> Model:
    """Parse, expand, validate and lower a model; bind no data.

    Args:
        model: A YAML path, a mapping, or a loaded :class:`Model`.

    Returns:
        The validated schema.

    Raises:
        LanguageError: A construct outside the streaming language.
        ValueError: A schema or expression that does not parse.

    Warns:
        LpspecWarning: Advice short of an error — a declared dimension nothing
            uses as an axis, say. Issued here and nowhere else.
    """
    schema = load_model(model)
    program = lower_program(schema)
    for note in advice(program):
        warnings.warn(note, LpspecWarning, stacklevel=2)
    return schema


def build(
    model: str | Path | dict[str, Any] | Model,
    sources: Mapping[str, Any],
    *,
    coords: dict[str, Any] | None = None,
) -> Engine:
    """Bind data to *model* and build it on the relational engine.

    Args:
        model: A YAML path, a mapping, or a loaded :class:`Model`.
        sources: Parameter names to parquet paths or in-memory tables, and
            optionally dimension names to index tables.
        coords: Dimension labels neither *sources* nor the YAML carries.

    Which engine builds it is set by ``LPSPEC_ENGINE`` and is deliberately not
    a parameter here: the engines produce the same model integer for integer,
    so the choice cannot change the answer, only what computing it costs. A
    knob that cannot change the answer does not belong in the call that
    produces one — see :mod:`lpspec.relational.engines`.

    Returns:
        The executor holding the built model. It feeds any number of sinks —
        ``ex.solve()`` and ``ex.write(path)`` on the same object — and
        ``ex.close()`` releases it.

    Raises:
        LanguageError: A construct outside the streaming language.
        DataError: A source that is missing, unreadable, or the wrong shape.
    """
    schema = load_model(model)
    program = lower_program(schema)
    ex = resolve_engine()()
    try:
        ex.build(program, tidy_sources(schema, dict(sources), coords))
    except BaseException:
        ex.close()
        raise
    return ex


def solve(
    model: str | Path | dict[str, Any] | Model,
    sources: Mapping[str, Any],
    solver_options: Mapping[str, Any] | None = None,
    solver_name: str = 'highs',
    **build_kwargs: Any,
) -> Result:
    """Build *model* and solve it in one call.

    Args:
        model: A YAML path, a mapping, or a loaded :class:`Model`.
        sources: As :func:`build` takes them.
        solver_options: Forwarded to the solver verbatim, in its own
            vocabulary (``{'time_limit': 60}``).
        solver_name: ``highs``, which ships with the package, or ``gurobi``,
            which needs the ``[gurobi]`` extra.
        **build_kwargs: Passed to :func:`build`.

    Returns:
        The solution, the built model still attached to it. ``result.close()``
        releases that model.

    Raises:
        LpspecError: A solver name nothing serves — checked before the build.
    """
    solver(solver_name)
    ex = build(model, sources, **build_kwargs)
    try:
        return ex.solve(solver_options=solver_options, solver_name=solver_name)
    except BaseException:
        ex.close()
        raise


def write(
    model: str | Path | dict[str, Any] | Model,
    sources: Mapping[str, Any],
    out: str | Path,
    **build_kwargs: Any,
) -> Path:
    """Build *model* and stream it to a file, in the format *out*'s suffix names.

    Args:
        model: A YAML path, a mapping, or a loaded :class:`Model`.
        sources: As :func:`build` takes them.
        out: Where to write; ``.lp`` is what ships.
        **build_kwargs: Passed to :func:`build`.

    Returns:
        The path written.

    Raises:
        ValueError: A suffix nothing writes — checked before the build.
        NotImplementedError: A format that is planned and not here yet.
    """
    out = Path(out)
    writer(out.suffix.lower())
    with build(model, sources, **build_kwargs) as ex:
        ex.write(out)
    return out

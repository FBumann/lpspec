"""The both-lanes harness: one model, two backends, one answer.

The differential test is this project's central claim — the same YAML must
mean the same thing on the eager linopy lane and on the streaming relational
one (docs/ARCHITECTURE.md, hard rule 3). Twelve tests made that claim by hand, in
seven files, each rebuilding the same fifteen lines: build eagerly, solve,
take the objective, re-parse the schema, lower it, bind sources, execute,
compare.

What the repetition cost was not correctness but *evenness*. Every copy
compared objectives and checked the status, but only five of the twelve also
wrote the LP file and re-solved it, and nothing recorded why the other seven
skipped that third opinion — so the strength of the claim varied with which
file you happened to be reading. Here it is one ``lp=True``, and a test that
does not ask for it is visibly choosing not to.

Importing this module is the ``[linopy]`` guard: it reaches the oracle
through ``tests.oracle``, so a bare install skips every module that uses the
harness at collection time, with no filename list to maintain.

Usage — the executor stays open for the length of the ``with`` block, so
per-variable primal checks live inside it::

    with differential(NONCONVEX_YAML, data, coords, lp=True) as run:
        assert run.result.to_pandas('op_cost') ...
"""

from __future__ import annotations

import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from lpspec.lowering import lower_program
from lpspec.relational import PolarsExecutor
from lpspec.sources import tidy_sources
from tests.conftest import raw_of, schema_of, solve_lp_file
from tests.oracle import linopy, lpspec_linopy

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from lpspec.language.model import Model
    from lpspec.relational.result import Result

#: Both lanes hand the same numbers to the same solver, so they must agree to
#: solver precision, not to a fudge factor. One tolerance, one place.
RTOL = 1e-9


@dataclass
class Agreement:
    """What the two lanes produced, for tests that assert past the objective."""

    oracle: float
    """The eager objective — the number both lanes had to reach."""

    model: linopy.Model
    """The eager model, for structural assertions (labels, masks, solution)."""

    result: Result
    """The relational solution; live until the ``with`` block exits."""

    schema: Model
    executor: PolarsExecutor
    lp: Path | None = None
    """The written LP file, when ``lp=True`` — already checked to agree."""


@contextmanager
def differential(
    model: str | Path | dict[str, Any],
    data: Mapping[str, Any],
    coords: Mapping[str, Any] | None = None,
    *,
    lp: bool = False,
    rel: float = RTOL,
) -> Iterator[Agreement]:
    """Build ``model`` on both lanes with the same inputs; assert they agree.

    ``model`` is a ``Path`` to a file, the YAML text itself, or a raw dict —
    the eager lane only takes paths, so text and dicts are written to a
    temporary file here rather than in every caller.

    Set ``lp=True`` to also write and re-solve the LP file, the third opinion.
    """
    schema = schema_of(model)

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        path = model if isinstance(model, Path) else _write(work / 'model.yaml', model)

        m = lpspec_linopy.build(path, data=dict(data), coords=dict(coords) if coords else None)
        m.solve(solver_name='highs', output_flag=False)
        oracle = float(m.objective.value)
        assert np.isfinite(oracle), 'the eager oracle is infeasible or unbounded — fix the data, not the tolerance'

        with PolarsExecutor() as ex:
            ex.build(lower_program(schema), tidy_sources(schema, data, coords))
            result = ex.solve()
            assert result.is_ok
            assert result.objective == pytest.approx(oracle, rel=rel)

            lp_path = None
            if lp:
                lp_path = work / 'model.lp'
                ex.write(lp_path)
                assert solve_lp_file(lp_path) == pytest.approx(oracle, rel=rel)

            yield Agreement(oracle=oracle, model=m, result=result, schema=schema, executor=ex, lp=lp_path)


def _write(path: Path, model: str | dict[str, Any]) -> Path:
    import yaml as pyyaml

    path.write_text(model if isinstance(model, str) else pyyaml.safe_dump(raw_of(model)))
    return path

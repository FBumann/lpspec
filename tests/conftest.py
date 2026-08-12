"""Shared fixtures and schema helpers for lpspec tests.

Everything here is linopy-free *and pandas-free at import*, so it loads on a
bare install. On a bare install (no [linopy] extra) the eager/oracle modules
skip themselves: they reach the oracle through ``tests.oracle``, whose
``importorskip`` guard fires at collection. There is no list of filenames to
keep in sync here — a module that needs the extra says so by importing it. The
differential harness lives in ``tests.differential`` for the same reason:
importing it *is* the guard.

pandas follows the same discipline one level down. It is no longer a runtime
dependency (it ships with the ``[linopy]`` extra, for the oracle and for
``Result.to_pandas``), so a fixture that hands out pandas objects imports it in
its own body: requesting the fixture is what asks for the dependency, and the
bare job never requests it. ``dispatch_inputs`` and ``dispatch_frame_inputs``
are the same numbers in the two shapes — the oracle lane is pandas-native, the
engine is frame-native, and the module constants are the single source of both.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import yaml as pyyaml

from lpspec.language.validation import load_model

if TYPE_CHECKING:
    from lpspec.language.model import Model

if TYPE_CHECKING:
    from collections.abc import Iterator

EXAMPLES_DIR = Path(__file__).parent.parent / 'examples'

#: The dispatch model as a dict, for tests that need to mutate a declaration
#: rather than read a file. Deliberately the same math as
#: ``examples/dispatch.yaml`` so a reader who knows one knows the other; use
#: :func:`override` to vary it.
DISPATCH_MODEL: dict[str, Any] = {
    'dimensions': {'snapshot': {'dtype': 'int'}, 'generator': {'values': ['wind', 'gas']}},
    'parameters': {
        'p_max': {'dims': ['generator']},
        'cost': {'dims': ['generator']},
        'load': {'dims': ['snapshot']},
    },
    'variables': {'p': {'foreach': ['snapshot', 'generator'], 'bounds': {'lower': 0, 'upper': 'p_max'}}},
    'constraints': {'balance': {'foreach': ['snapshot'], 'expression': 'sum(p, over=generator) == load'}},
    'objectives': {'total': {'sense': 'minimize', 'expression': 'sum(p * cost, over=generator)'}},
}


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        '--update-golden',
        action='store_true',
        default=False,
        help='rewrite committed golden output (examples/*.out) from this run instead of asserting on it',
    )
    parser.addoption(
        '--engine',
        default=None,
        help='run the whole suite on this engine instead of the default. Every test that reaches '
        '`lps.build` is then a test of that engine, which is what makes a second one a lane '
        'rather than a demo.',
    )


@pytest.fixture
def engine_internals(pytestconfig: pytest.Config) -> None:
    """Skip unless this run is on the polars engine.

    For the few tests that reach *inside* an engine — its compiler, its label
    strategies — rather than through `lps.build`. Another engine does not have
    to share those internals to be correct; it has to produce the same model,
    which every other test here already checks against the same YAML.

    `--engine` unset means `DEFAULT_ENGINE`, not polars, so it is resolved
    before the comparison: reading it as polars would run these against
    whichever engine the default names.
    """
    from lpspec.relational import engines

    name = pytestconfig.getoption('--engine') or engines.DEFAULT_ENGINE
    if name != 'polars':
        pytest.skip(f'reaches polars-engine internals; this run is on {name!r}')


@pytest.fixture(autouse=True, scope='session')
def _engine_under_test(pytestconfig: pytest.Config) -> Iterator[None]:
    """Point `lps.build`'s default at `--engine`, for the whole session.

    Through `LPSPEC_ENGINE`, the same switch a user has — so this fixture
    exercises the documented mechanism rather than a private hook only the
    tests know about. Setting the default rather than threading a parameter
    through every test: the claim being checked is that *the suite* passes on
    another engine, and a parameter each test opts into would only check the
    ones that remembered.
    """
    name = pytestconfig.getoption('--engine')
    if name is None:
        yield
        return
    import os

    from lpspec.relational import engines

    previous = os.environ.get(engines.ENV_VAR)
    os.environ[engines.ENV_VAR] = name
    try:
        yield
    finally:
        if previous is None:
            del os.environ[engines.ENV_VAR]
        else:
            os.environ[engines.ENV_VAR] = previous


# ---------------------------------------------------------------------------
# building schemas to test against
# ---------------------------------------------------------------------------


def override(base: dict[str, Any], **patch: Any) -> dict[str, Any]:
    """A deep copy of ``base`` with dotted paths replaced.

    ``override(DISPATCH_MODEL, **{'variables.p.where': 'p_max > 0'})``. Missing
    intermediate keys are created, so this both edits an existing declaration
    and adds a new one — which is what makes a whole family of "the base model
    but for one thing" tests a one-liner each.
    """
    raw = copy.deepcopy(base)
    for dotted, value in patch.items():
        node = raw
        *parents, leaf = dotted.split('.')
        for key in parents:
            node = node.setdefault(key, {})
        node[leaf] = value
    return raw


def schema_of(source: str | Path | dict[str, Any], **patch: Any) -> Model:
    """A ``Model`` from a YAML path, YAML text, or a raw dict.

    ``Path`` means a file, ``str`` means the YAML itself — the distinction is
    the type, never a guess about the content. ``**patch`` applies
    :func:`override` first, which is how a test says "this example, but with
    ``**`` in the objective".
    """
    raw = raw_of(source)
    return load_model(override(raw, **patch) if patch else raw)


def raw_of(source: str | Path | dict[str, Any]) -> dict[str, Any]:
    """The parsed mapping behind a path / YAML text / dict, unvalidated."""
    if isinstance(source, dict):
        return source
    text = source.read_text() if isinstance(source, Path) else source
    return pyyaml.safe_load(text)


def solve_lp_file(path: Path | str) -> float:
    """Objective HiGHS reaches reading the written LP file back from disk.

    The third opinion in a differential: the ``highs`` solver builds the model
    through the HiGHS API, this one round-trips it through text, and a sink
    that writes a wrong file is otherwise invisible. Lives here rather than in
    ``tests.differential`` because highspy is a core dependency — a bare
    install must still be able to check the LP sink.
    """
    import highspy

    h = highspy.Highs()
    h.setOptionValue('output_flag', False)
    h.readModel(str(path))
    h.run()
    assert h.getModelStatus() == highspy.HighsModelStatus.kOptimal
    return h.getInfo().objective_function_value


def by_coord(result: Any, name: str, dim: str) -> dict[Any, float]:
    """A variable's primal as ``{coordinate: value}``, for a one-dim variable.

    One ``primal`` call, then one zip — and that is the whole reason this is a
    function. ``primal`` is a label join and promises row *order* but not that
    two separate calls line up column-wise, so the idiom has to read the frame
    once and pair its columns in a single pass. Six tests do this and the
    caveat was written down at one of them; here it applies to all six by
    construction.
    """
    frame = result.primal(name)
    return dict(zip(frame[dim], frame['value'], strict=True))


def resolved(text, schema):
    """Parse + expand + resolve — exactly what a backend receives.

    Tests that call `_lower_expr` or `evaluate_where` directly must go through
    this: a raw `parse_expression` result still holds NameNodes, and both
    backends now assert those never reach them (resolution.py).
    """
    from lpspec.language.resolution import Namespace, expression_of

    return expression_of(text, schema, Namespace.of(schema), 't')


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


@pytest.fixture
def dispatch_yaml() -> Path:
    return EXAMPLES_DIR / 'dispatch.yaml'


#: Generators of ``examples/dispatch.yaml``, and the snapshot count. Distinct
#: costs, so the optimal vertex is unique and primals are comparable across
#: lanes.
DISPATCH_GENERATORS = ('wind', 'solar', 'gas')
DISPATCH_P_MAX = (100.0, 60.0, 200.0)
DISPATCH_COST = (1.0, 2.0, 50.0)
DISPATCH_SNAPSHOTS = 48


def _dispatch_load() -> np.ndarray:
    """The load series both shapes below carry — one draw, one seed."""
    rng = np.random.default_rng(3)
    return (rng.uniform(0.2, 0.8, DISPATCH_SNAPSHOTS) * sum(DISPATCH_P_MAX)).round(3)


@pytest.fixture
def dispatch_inputs():
    """Dispatch data as pandas — the shape the linopy oracle takes.

    Pairs with :func:`dispatch_frame_inputs`: same numbers, and a test picks
    the shape by which lane it exercises. Importing pandas here rather than at
    module scope is what keeps this file loadable on a bare install.
    """
    import pandas as pd

    data = {
        'p_max': pd.Series(dict(zip(DISPATCH_GENERATORS, DISPATCH_P_MAX, strict=True))),
        'cost': pd.Series(dict(zip(DISPATCH_GENERATORS, DISPATCH_COST, strict=True))),
        'load': pd.Series(_dispatch_load(), index=pd.RangeIndex(DISPATCH_SNAPSHOTS, name='snapshot')),
    }
    coords = {'snapshot': pd.RangeIndex(DISPATCH_SNAPSHOTS, name='snapshot')}
    return data, coords


@pytest.fixture
def dispatch_frame_inputs():
    """The same data as tidy frames — the shape the engine documents.

    Tests that assert the native API's behaviour use this one, so they stay
    runnable with no dataframe library beyond the engine's own installed.
    """
    import polars as pl

    generators = list(DISPATCH_GENERATORS)
    data = {
        'p_max': pl.DataFrame({'generator': generators, 'value': list(DISPATCH_P_MAX)}),
        'cost': pl.DataFrame({'generator': generators, 'value': list(DISPATCH_COST)}),
        'load': pl.DataFrame({'snapshot': list(range(DISPATCH_SNAPSHOTS)), 'value': _dispatch_load()}),
    }
    coords = {'snapshot': range(DISPATCH_SNAPSHOTS)}
    return data, coords


@pytest.fixture
def commitment_inputs():
    """Data for the unit-commitment MILP in ``tests.test_milp``.

    Here rather than beside the model because two modules need it: the MILP
    itself, and the duals refusal — a mixed-integer model is the case that has
    no dual solution to give.
    """
    import pandas as pd

    rng = np.random.default_rng(5)
    n_s = 24
    p_max = pd.Series({'coal': 120.0, 'gas': 80.0, 'peaker': 60.0})
    data = {
        'p_max': p_max,
        'cost': pd.Series({'coal': 10.0, 'gas': 30.0, 'peaker': 90.0}),
        'fix_cost': pd.Series({'coal': 400.0, 'gas': 150.0, 'peaker': 20.0}),
        'load': pd.Series(
            (rng.uniform(0.3, 0.9, n_s) * p_max.sum()).round(1),
            index=pd.RangeIndex(n_s, name='snapshot'),
        ),
    }
    coords = {
        'snapshot': pd.RangeIndex(n_s, name='snapshot'),
        'generator': pd.Index(p_max.index, name='generator'),
    }
    return data, coords


@pytest.fixture
def transport_data():
    """A four-bus network whose data is feasible by construction.

    Generation is dealt round-robin so every bus has some locally, the topology
    is a ring plus one chord so every bus is reachable, and loads sit below each
    bus's local capacity — feasible even with zero flow. The cost spread still
    makes cross-bus flows optimal, so the network is not decoration.
    """
    import pandas as pd

    rng = np.random.default_rng(11)
    n_s, n_b, n_g, n_l = 24, 4, 9, 5
    buses = [f'b{i}' for i in range(n_b)]
    gens = pd.DataFrame(
        {
            'generator': [f'g{i}' for i in range(n_g)],
            'bus': [buses[i % n_b] for i in range(n_g)],
            'p_max': rng.uniform(80, 150, n_g).round(3),
            'cost': rng.uniform(5, 100, n_g).round(3),
        }
    )
    pairs = [(buses[i], buses[(i + 1) % n_b]) for i in range(n_b)] + [(buses[0], buses[2])]
    lines = pd.DataFrame(
        {
            'line': [f'l{i}' for i in range(n_l)],
            'from_bus': [a for a, _ in pairs],
            'to_bus': [b for _, b in pairs],
            'cap': rng.uniform(60, 120, n_l).round(3),
        }
    )
    local_cap = gens.groupby('bus')['p_max'].sum().reindex(buses).to_numpy()
    factors = rng.uniform(0.3, 0.8, (n_s, n_b))
    load = pd.DataFrame(
        {
            'snapshot': np.repeat(np.arange(n_s), n_b),
            'bus': buses * n_s,
            'value': (factors * local_cap).round(3).ravel(),
        }
    )
    return gens, lines, load

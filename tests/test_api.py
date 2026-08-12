"""Native API: YAML → streaming engine → solver, with linopy never imported.

The linopy-free guarantee is asserted in a subprocess so conftest's optional
lpspec_linopy import cannot pollute the check.

This module is deliberately **pandas-free**: it is the bare install's proof
that the native path — frames in, build, solve, frames out — needs no
dataframe library beyond the engine's own. The tests that exercise the bridges
*out* (``to_pandas``, ``to_dataarray``) say so with an ``importorskip``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import polars as pl
import pytest

import lpspec as lps
from lpspec.language.model import Model
from tests.conftest import schema_of, solve_lp_file


def test_solve(dispatch_yaml, dispatch_frame_inputs):
    sources, coords = dispatch_frame_inputs
    result = lps.solve(dispatch_yaml, sources, coords=coords)
    try:
        assert result.is_ok
        assert np.isfinite(result.objective)
        balance = result.primal('p').group_by('snapshot').agg(pl.col('value').sum()).sort('snapshot')
        assert np.allclose(balance['value'], sources['load']['value'])
    finally:
        result.close()


def test_build_context_manager_and_write(dispatch_yaml, dispatch_frame_inputs, tmp_path):
    sources, coords = dispatch_frame_inputs
    with lps.build(dispatch_yaml, sources, coords=coords) as ex:
        result = ex.solve()
        assert result.is_ok
        objective_direct = result.objective

    lp = lps.write(dispatch_yaml, sources, tmp_path / 'm.lp', coords=coords)
    assert solve_lp_file(lp) == pytest.approx(objective_direct, rel=1e-9)


def test_parquet_path_sources(dispatch_yaml, dispatch_frame_inputs, tmp_path):
    sources, coords = dispatch_frame_inputs
    paths = {}
    for name, frame in sources.items():
        p = tmp_path / f'{name}.parquet'
        frame.write_parquet(p)
        paths[name] = str(p)

    result = lps.solve(dispatch_yaml, paths, coords=coords)
    try:
        assert result.is_ok
    finally:
        result.close()

    ref = lps.solve(dispatch_yaml, sources, coords=coords)
    try:
        assert result.objective == pytest.approx(ref.objective, rel=1e-9)
    finally:
        ref.close()


def test_runtime_is_linopy_free(dispatch_yaml):
    """Import the package, build and solve on Arrow sources — linopy never loads.

    The claim every engine keeps, which is why the subprocess below is pinned
    to the default rather than following ``--engine``: neither lane's shim is
    ever on the build path, whichever engine walks it.

    pandas is deliberately not on the list. The default engine reaches duckdb's
    dataframe interop, which imports pyarrow, and `pyarrow._dataset` imports
    pandas where pandas exists — so the narrower "no dataframe library but
    polars" claim belongs to the *polars* engine and is stated in
    :func:`test_the_polars_engine_needs_neither_pandas_nor_pyarrow`. It is worth
    having in both places: a bridge out (``to_pandas``, ``to_dataarray``) has to
    stay a bridge somewhere it can be checked.

    Distinct from, and weaker than, the claim that they need not be
    *installed*: the bare-install CI job is what proves that, running this
    suite with neither linopy nor pandas present at all.
    """
    absent = ('linopy', 'xarray')
    script = textwrap.dedent(f"""
        import sys
        assert "linopy" not in sys.modules

        import polars as pl
        import lpspec as lps
        for lib in {absent!r}:
            assert lib not in sys.modules, f"package import pulled in {{lib}}"

        result = lps.solve(
            {str(dispatch_yaml)!r},
            {{
                "p_max": pl.DataFrame({{"generator": ["wind", "solar", "gas"],
                                       "value": [100.0, 60.0, 200.0]}}),
                "cost": pl.DataFrame({{"generator": ["wind", "solar", "gas"],
                                      "value": [1.0, 2.0, 50.0]}}),
                "load": pl.DataFrame({{"snapshot": [0, 1, 2],
                                      "value": [80.0, 120.0, 150.0]}}),
            }},
            coords={{"snapshot": range(3)}},
        )
        assert result.is_ok
        assert isinstance(result.primal("p"), pl.DataFrame), "no dataframe on either side"
        assert result.primal("p").height == 9
        result.close()
        for lib in {absent!r}:
            assert lib not in sys.modules, f"solve pulled in {{lib}}"
        print("LINOPY_FREE_OK")
    """)
    # the default engine, whatever the session is running on: an env var that
    # leaked in would silently turn this into a claim about a different engine
    env = {k: v for k, v in os.environ.items() if k != 'LPSPEC_ENGINE'}
    out = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=300, env=env)
    assert out.returncode == 0, out.stderr
    assert 'LINOPY_FREE_OK' in out.stdout


#: The dispatch build both engine-footprint tests below run, as a script. One
#: model, one `_tables()`, and a line saying what got imported — the only thing
#: that differs between the two is which engine `LPSPEC_ENGINE` names, which is
#: exactly the comparison being made.
_FOOTPRINT = """
    import sys

    import polars as pl
    import lpspec as lps
    with lps.build({model!r}, {{
        "p_max": pl.DataFrame({{"generator": ["wind"], "value": [100.0]}}),
        "cost": pl.DataFrame({{"generator": ["wind"], "value": [1.0]}}),
        "load": pl.DataFrame({{"snapshot": [0], "value": [80.0]}}),
    }}, coords={{"snapshot": range(1)}}) as ex:
        ex._tables()
    print("IMPORTED", sorted(m for m in ("pandas", "pyarrow") if m in sys.modules))
"""


def _footprint(dispatch_yaml, engine: str) -> str:
    """Which of pandas/pyarrow *engine* pulled in, building in a clean process."""
    script = textwrap.dedent(_FOOTPRINT.format(model=str(dispatch_yaml)))
    env = {**os.environ, 'LPSPEC_ENGINE': engine}
    out = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=300, env=env)
    assert out.returncode == 0, out.stderr
    return out.stdout


def test_the_default_engine_needs_arrow(dispatch_yaml):
    """Why pyarrow is a runtime dependency and not an afterthought.

    duckdb and polars hand frames to each other through Arrow, so pyarrow is a
    requirement of the default engine rather than an accident. A declaration
    carrying only `duckdb` installs cleanly and then fails on the first build,
    which is the failure this pins.

    **pandas is not implied**, and that half cannot be checked here: pyarrow
    imports pandas whenever pandas is installed, and this environment has it
    for the linopy lane. Hiding it in-process does not work either — polars
    imports pandas itself on the way up. The bare-install CI job is what proves
    it, by not having pandas there at all.
    """
    assert 'pyarrow' in _footprint(dispatch_yaml, 'duckdb'), (
        'the default engine no longer needs pyarrow — narrow the dependencies in pyproject'
    )


def test_the_polars_engine_needs_neither_pandas_nor_pyarrow(dispatch_yaml):
    """No dataframe library but polars, on this engine.

    A per-engine claim rather than a runtime-wide one, because the default
    engine cannot keep it (:func:`test_the_default_engine_needs_arrow`). Stated
    here it keeps the polars engine's footprint a measured property rather than
    a thing that quietly widens: a build on it touches nothing else, so on this
    engine a bridge out stays a bridge.
    """
    assert 'IMPORTED []' in _footprint(dispatch_yaml, 'polars'), (
        'the polars engine reaches a dataframe library it does not need'
    )


def test_check_and_load_model_need_no_data(dispatch_yaml):
    """The model stands for itself: the schema is read from the file when
    wanted, never carried on a built model."""
    for schema in (lps.check(dispatch_yaml), lps.load_model(dispatch_yaml)):
        assert schema.variables['p'].foreach == ['snapshot', 'generator']
        assert schema.parameters['load'].dims == ['snapshot']


@pytest.mark.parametrize(
    ('expression', 'match'),
    [
        pytest.param('sum(p ** 2, over=generator)', r"operator '\*\*'", id='an-operator-outside-the-language'),
        pytest.param('sum(p * p, over=generator)', 'degree 2', id='degree-2-caught-with-no-data-bound'),
    ],
)
def test_check_reports_language_errors_before_any_data_is_bound(
    dispatch_yaml, dispatch_frame_inputs, expression, match
):
    """The CI verb enforces the ceiling with no data bound (docs/design/ceiling.md).

    ``build`` is asserted to say the same thing rather than defer it to the
    solver.
    """
    raw = schema_of(dispatch_yaml, **{'objectives.total_cost.expression': expression}).model_dump()

    with pytest.raises(lps.LanguageError, match=match):
        lps.check(raw)
    sources, coords = dispatch_frame_inputs
    with pytest.raises(lps.LanguageError, match=match):
        lps.build(raw, sources, coords=coords)


def test_error_hierarchy_is_one_catchable_tree():
    """One ``except`` covers the package, and the model/run split is real."""
    for cls in (lps.LanguageError, lps.DataError):
        assert issubclass(cls, lps.LpspecError)
    for cls in (lps.SchemaError, lps.DimensionError, lps.PiecewiseExpansionError):
        assert issubclass(cls, lps.LanguageError)
    assert not issubclass(lps.DataError, lps.LanguageError)
    assert issubclass(lps.LpspecError, ValueError)


def test_an_unknown_solver_is_refused_with_the_alternatives(dispatch_yaml, dispatch_frame_inputs):
    """The set of solvers is closed, and a name outside it never falls back to
    the default — solving with a solver other than the one asked for is the one
    answer that cannot be right. Here rather than in ``test_gurobi_sink.py``,
    which skips without the extra: the closed set is a property of the package,
    not of gurobi. Refused before the build, as an unwritable suffix is."""
    from lpspec.relational.sinks import SOLVERS

    sources, coords = dispatch_frame_inputs
    with pytest.raises(lps.LpspecError, match='unknown solver'):
        lps.solve(dispatch_yaml, sources, solver_name='cplex', coords=coords)
    assert set(SOLVERS) == {'highs', 'gurobi'}


def test_a_list_of_models_is_refused(dispatch_yaml):
    """Composition is merging declarations, not passing several models.

    The message points at the dict, because a caller holding two files has
    somewhere to go — #30 declined the native merge rather than deferring it.
    """
    with pytest.raises(TypeError, match='merge the declarations'):
        lps.check([dispatch_yaml, dispatch_yaml])


def test_write_suffix_dispatch(dispatch_yaml, dispatch_frame_inputs, tmp_path):
    sources, coords = dispatch_frame_inputs
    out = lps.write(dispatch_yaml, sources, tmp_path / 'm.lp', coords=coords)
    assert out.stat().st_size > 0
    with pytest.raises(NotImplementedError, match='mps'):
        lps.write(dispatch_yaml, sources, tmp_path / 'm.mps', coords=coords)
    with pytest.raises(ValueError, match='unsupported output format'):
        lps.write(dispatch_yaml, sources, tmp_path / 'm.nc', coords=coords)


def test_solution_to_parquet(dispatch_yaml, dispatch_frame_inputs, tmp_path):
    """One file per variable, tidy, streamed straight to disk."""
    sources, coords = dispatch_frame_inputs
    result = lps.solve(dispatch_yaml, sources, coords=coords)
    assert result.is_ok
    written = result.to_parquet(tmp_path / 'solution')
    assert set(written) == {'p'}
    frame = pl.read_parquet(written['p'])
    assert set(frame.columns) == {'snapshot', 'generator', 'value'}
    assert frame.height == result.primal('p').height


def test_read_back_is_in_label_order_and_stays_there(dispatch_yaml, dispatch_frame_inputs, tmp_path):
    """A read is a join, and a join settles no order — so the read states one.

    Was: every call came back in whatever order the hash join finished in, so
    two reads of one unchanged result disagreed and five writes of one solution
    produced five different files. Nothing was wrong with the numbers, which is
    what made it worth stating rather than leaving to the planner.

    Label order is row-major over the coordinate product, so it is checkable
    against the coordinates themselves: `snapshot` varies slowest, and within
    it `generator` follows the order the file declares.
    """
    sources, coords = dispatch_frame_inputs
    generators = list(sources['p_max']['generator'])
    with lps.solve(dispatch_yaml, sources, coords=coords) as result:
        first = result.primal('p')
        assert first.equals(result.primal('p')), 'a second read agrees, to the row'

        by_declaration = first.with_columns(
            pl.col('generator').replace_strict(generators, range(len(generators))).alias('ord')
        )
        assert by_declaration.equals(by_declaration.sort('snapshot', 'ord'))

        written = [result.to_parquet(tmp_path / f'solution{i}')['p'].read_bytes() for i in range(3)]
        assert len(set(written)) == 1, 'the same solution writes the same bytes'


def test_a_result_stays_readable_until_it_is_closed(dispatch_yaml, dispatch_frame_inputs):
    """No lifetime to manage: reading is valid until you say otherwise.

    The built model is frames this process owns, so nothing expires on its own
    and a caller who never closes loses nothing but memory. `close()` is there
    to hand a large model back early, and it means what it says — after it,
    there is nothing left to read.
    """
    sources, coords = dispatch_frame_inputs
    result = lps.solve(dispatch_yaml, sources, coords=coords)
    height = result.primal('p').height
    assert height > 0
    assert result.primal('p').height == height, 'still readable, with no close in sight'

    result.close()
    with pytest.raises(lps.LpspecError, match='this result was closed'):
        result.primal('p')


def test_a_second_solve_does_not_rewrite_the_first_result(dispatch_yaml, dispatch_frame_inputs):
    """A result reports its own solve, not the executor's latest.

    Was: the values lived on the executor and every reader went back to them,
    so `objective` was a snapshot while `primal` was live — one result
    disagreeing with itself after a second solve, silently and with plausible
    numbers. Nothing supported re-binds data yet, so the bound has to be moved
    the way the planned in-place update will (#382: `changeColsBounds`
    against labels that are already solver indices).
    """
    key = ['snapshot', 'generator']  # a read is a join, so compare on coordinates
    sources, coords = dispatch_frame_inputs
    with lps.build(dispatch_yaml, sources, coords=coords) as ex:
        first = ex.solve()
        before = first.primal('p').sort(key)
        assert first.is_ok

        assert ex._obj is not None
        ex._obj = ex._obj.with_columns(-pl.col('coeff'))
        second = ex.solve()

        assert not second.primal('p').sort(key).equals(before), 'the second solve really moved'
        assert first.primal('p').sort(key).equals(before), 'and the first still reports its own'
        assert first.objective != pytest.approx(second.objective)


def test_primal_is_a_frame_and_to_pandas_is_the_bridge(dispatch_yaml, dispatch_frame_inputs):
    """A frame is the shape results come in; pandas is an exit, not a shape.

    The two must describe the same table — the bridge is a conversion, not a
    second query with its own opinion about column order or dtypes.
    """
    sources, coords = dispatch_frame_inputs
    result = lps.solve(dispatch_yaml, sources, coords=coords)
    frame = result.primal('p')
    assert isinstance(frame, pl.DataFrame)
    assert frame.columns == ['snapshot', 'generator', 'value']

    pandas = pytest.importorskip('pandas')
    converted = result.to_pandas('p')
    assert isinstance(converted, pandas.DataFrame)
    assert list(converted.columns) == frame.columns
    assert len(converted) == frame.height
    assert frame['value'].sum() == pytest.approx(converted['value'].sum())


def test_no_helper_registry_anywhere():
    """The helper set is closed — there is no way to register more, on any
    surface (#38's ``escape:`` island replaces the idea).

    This is what makes the two lanes accept the same language, and hence what
    makes the differential tests an oracle rather than a comparison of
    dialects (docs/ARCHITECTURE.md, "The expressive ceiling").
    """
    import lpspec.language.helpers as helpers

    assert not hasattr(lps, 'register')
    assert not hasattr(helpers, 'register')
    assert not hasattr(helpers, '_REGISTRY')


def test_solution_to_dataarray(dispatch_yaml, dispatch_frame_inputs):
    """Long tables are right for joining, wrong for the array math that
    post-processing is mostly made of. `to_dataarray` is the bridge."""
    pytest.importorskip('xarray')
    sources, coords = dispatch_frame_inputs

    with lps.solve(dispatch_yaml, sources, coords=coords) as result:
        arr = result.to_dataarray('p')
        tidy = result.to_pandas('p')

    assert arr.name == 'p', "named for the variable, not 'value' — the tidy column it came from"
    assert sorted(arr.dims) == ['generator', 'snapshot']
    assert arr.sizes['generator'] == 3
    wind_0 = tidy.query("generator == 'wind' and snapshot == 0")['value'].iloc[0]
    assert float(arr.sel(generator='wind', snapshot=0)) == pytest.approx(wind_0), (
        'the labelled form is the tidy form, indexed'
    )


def test_solution_to_dataset(dispatch_yaml, dispatch_frame_inputs):
    """Several variables at once, each keeping its own dims."""
    pytest.importorskip('xarray')
    sources, coords = dispatch_frame_inputs

    with lps.solve(dispatch_yaml, sources, coords=coords) as result:
        ds = result.to_dataset('p')
        tidy = result.to_pandas('p')

    assert list(ds.data_vars) == ['p']
    assert sorted(ds['p'].dims) == ['generator', 'snapshot']
    first = tidy.iloc[0]
    assert float(ds['p'].sel(snapshot=first['snapshot'], generator=first['generator'])) == pytest.approx(first['value'])


TWO_VARIABLE_MODEL = {
    'dimensions': {'snapshot': {'dtype': 'int'}, 'generator': {'values': ['wind', 'gas']}},
    'parameters': {'p_max': {'dims': ['generator']}, 'load': {'dims': ['snapshot']}},
    'variables': {
        'p': {'foreach': ['snapshot', 'generator'], 'bounds': {'lower': 0, 'upper': 'p_max'}},
        'shed': {'foreach': ['snapshot'], 'bounds': {'lower': 0}},
    },
    'constraints': {
        'balance': {
            'foreach': ['snapshot'],
            'expression': 'sum(p, over=generator) + shed == load',
        }
    },
    'objectives': {'total': {'sense': 'minimize', 'expression': 'shed'}},
}


def test_to_dataset_defaults_to_every_variable():
    """A small model wants all of them at once, as linopy's model.solution
    gives you — naming them would be busywork."""
    pytest.importorskip('xarray')
    n = 4
    sources = {
        'p_max': pl.DataFrame({'generator': ['wind', 'gas'], 'value': [100.0, 200.0]}),
        'load': pl.DataFrame({'snapshot': list(range(n)), 'value': np.full(n, 90.0)}),
    }

    with lps.solve(TWO_VARIABLE_MODEL, sources, coords={'snapshot': range(n)}) as result:
        ds = result.to_dataset()
        subset = result.to_dataset('shed')

    assert set(ds.data_vars) == {'p', 'shed'}
    assert sorted(ds['p'].dims) == ['generator', 'snapshot']
    assert list(ds['shed'].dims) == ['snapshot'], 'each variable keeps its own dims'
    assert set(subset.data_vars) == {'shed'}


@pytest.mark.parametrize(
    ('mistake', 'raw'),
    [
        pytest.param('unknown key', {'dimensionz': {}}, id='unknown-key'),
        pytest.param('bad dtype', {'dimensions': {'g': {'dtype': 'complex'}}}, id='bad-dtype'),
        pytest.param('unknown version', {'version': 99}, id='unknown-version'),
        pytest.param(
            'undeclared name',
            {
                'dimensions': {'g': {'dtype': 'str', 'values': ['a']}},
                'constraints': {'c': {'foreach': ['g'], 'expression': 'nope <= 1'}},
            },
            id='undeclared-name',
        ),
    ],
)
def test_a_wrong_model_raises_one_tree(mistake: str, raw: dict[str, object], tmp_path):
    """Every documented door answers with `LpspecError` (#527).

    Model checking happens in two places — pydantic's validators and the
    language checkers — and they failed differently, so `except LpspecError`,
    the thing `docs/api.md` tells a caller to write, missed the majority of
    model mistakes and a caller had no way to know which.

    `Model.__init__` is *not* in this list, and cannot be: defining one makes
    pydantic route validation through it, which runs every after-validator
    twice and the first time with no context, breaking `extend()`.
    """
    doors = {
        'lps.load_model': lambda: lps.load_model(raw),
        'lps.check': lambda: lps.check(raw),
        'lps.solve': lambda: lps.solve(raw, {}),
        'lps.write': lambda: lps.write(raw, {}, str(tmp_path / 'm.lp')),
        'Model.model_validate': lambda: Model.model_validate(raw),
        'Model.model_validate_json': lambda: Model.model_validate_json(json.dumps(raw)),
    }
    for door, call in doors.items():
        with pytest.raises(lps.LpspecError):
            call()
        try:
            call()
        except lps.LpspecError as exc:
            assert 'errors.pydantic.dev' not in str(exc), f"{door}: {mistake} leaks pydantic's envelope"


def test_a_closed_result_says_it_was_closed(dispatch_yaml, dispatch_frame_inputs):
    """`close` releases the model the readers join against, and they should say so.

    The status gate cannot notice: closing frees the model, not the solve, so
    `is_readable` stays true and the reader used to fall through to a bare
    `AssertionError` from the executor. Frames read before the close are their
    own data and stay valid, which is the half worth stating in the message.
    """
    sources, coords = dispatch_frame_inputs
    sol = lps.solve(dispatch_yaml, sources, coords=coords)
    frame = sol.primal('p')
    objective = sol.objective
    sol.close()

    assert frame.height > 0, 'a frame read before the close is its own data'
    assert sol.objective == objective, 'and the outcome needs no model to report'
    for read in (lambda: sol.primal('p'), lambda: sol.dual('power_balance')):
        with pytest.raises(lps.LpspecError, match='this result was closed'):
            read()

"""The two engines, on the same YAML, must answer the same.

`--engine polars` already runs the whole suite on the other engine, which is
the broad check. This is the narrow one: both engines in **one process**, on
the same model, compared outright. It catches what a full-suite run cannot —
a difference that is stable, so both runs pass their own assertions while
disagreeing with each other.

What is compared is what a caller can observe: the objective, the primal, the
duals, and the LP file byte for byte. The four frames are checked too, through
`_tables()`, because `row` and `col` *are* the solver's own indices and an
off-by-one there is a different model that still solves.
"""

from __future__ import annotations

import contextlib
import inspect
import os
from pathlib import Path

import polars as pl
import pytest

import lpspec as lps
from lpspec.relational import engines

ROOT = Path(__file__).resolve().parent.parent
ENGINES = ('duckdb', 'polars')

DISPATCH = {
    'p_max': pl.DataFrame({'generator': ['wind', 'solar', 'gas'], 'value': [10.0, 5.0, 100.0]}),
    'cost': pl.DataFrame({'generator': ['wind', 'solar', 'gas'], 'value': [1.0, 2.0, 50.0]}),
    'load': pl.DataFrame({'snapshot': [0, 1, 2], 'value': [12.0, 8.0, 20.0]}),
}

#: The same model with a mask that *removes* something. `dispatch`'s
#: `where: p_max > 0` reads only `generator`, so both engines label it by
#: arithmetic over a ranked survivor set rather than by counting — and with
#: every `p_max` positive that arithmetic is never asked a question a wrong
#: answer would show up in. Zeroing one generator is what makes the label a
#: claim: `gas` must be column 1 under every snapshot, not column 2.
MASKED = DISPATCH | {'p_max': pl.DataFrame({'generator': ['wind', 'solar', 'gas'], 'value': [10.0, 0.0, 100.0]})}

#: `storage` carries a cyclic `shift` — the operator a second engine is most
#: likely to get subtly wrong, and the one the spike calls hardest to port.
MODELS = [
    ('examples/dispatch.yaml', DISPATCH),
    ('examples/dispatch.yaml', MASKED),
    ('examples/storage.yaml', DISPATCH),
]


@contextlib.contextmanager
def using(engine: str):
    """Build on *engine*, through the switch a caller actually has.

    `lps.build` takes no engine parameter, so a test wanting a specific one
    sets `LPSPEC_ENGINE`. That keeps these tests on the documented mechanism
    rather than an internal the public path never touches.
    """
    previous = os.environ.get(engines.ENV_VAR)
    os.environ[engines.ENV_VAR] = engine
    try:
        yield
    finally:
        if previous is None:
            del os.environ[engines.ENV_VAR]
        else:
            os.environ[engines.ENV_VAR] = previous


def _frames(tables) -> dict[str, pl.DataFrame]:
    """The four frames in a form two engines can be compared entry for entry.

    `matrix` is asked for its `row` labels back (`matrix_block`) rather than
    read as the CSR pair it is stored as. Comparing `row_starts` alone would
    pass on two engines that agree about how many entries each row owns and
    disagree about which — and comparing the compressed frame alone would pass
    on two that put the same entries under different rows.
    """
    return {
        # `cols` is positional — one row per column in label order — so
        # sorting it would hide exactly the disagreement this compares
        'cols': tables.cols,
        'rows': tables.rows.sort('row'),
        'matrix': tables.matrix_block(0, tables.row_count).sort('row', 'col'),
        'obj': tables.obj.sort('col'),
    }


@pytest.mark.parametrize(('model', 'sources'), MODELS)
def test_both_engines_build_the_same_model(model, sources):
    built = {}
    try:
        for name in ENGINES:
            with using(name):
                ex = lps.build(ROOT / model, sources)
            built[name] = (ex, ex._tables())

        left, right = (_frames(t) for _, t in (built['polars'], built['duckdb']))
        for frame in ('cols', 'rows', 'matrix', 'obj'):
            assert left[frame].equals(right[frame]), f'{frame} differs between engines'

        (_, a), (_, b) = built['polars'], built['duckdb']
        assert a.column_count == b.column_count
        assert a.row_count == b.row_count
        assert a.objective_sense == b.objective_sense
        assert a.objective_constant == pytest.approx(b.objective_constant)
    finally:
        for ex, _ in built.values():
            ex.close()


@pytest.mark.parametrize(('model', 'sources'), MODELS)
def test_both_engines_produce_the_declared_schema(model, sources):
    """Same columns *and* same dtypes — `.equals` above checks only values.

    Two engines can agree on every number and still hand a sink different
    types, and a sink reads the frames without asking who filled them. It cost
    the duckdb engine a `vtype` of 10M copies of the word `continuous` where
    the other holds an `Enum`, and an `obj.coeff` typed `DECIMAL(2,1)` because
    SQL reads `1.0` as a decimal — a different number from the double the plan
    holds, and one that overflows above 9.9.

    `matrix` is checked against `(col, coeff)` rather than `sinks.MATRIX`: the
    frame a sink reads is CSR, so `row` was compressed into `row_starts` and is
    not a column on either engine.
    """
    from lpspec.relational import sinks

    declared = {'cols': sinks.COLS, 'obj': sinks.OBJ, 'rows': sinks.ROWS, 'matrix': ('col', 'coeff')}
    for name in ENGINES:
        with using(name), lps.build(ROOT / model, sources) as ex:
            tables = ex._tables()
            for frame, columns in declared.items():
                schema = dict(getattr(tables, frame).schema)
                assert schema == {c: sinks.DTYPES[c] for c in columns}, f'{name}: {frame} is not the declared schema'


@pytest.mark.parametrize(('model', 'sources'), MODELS)
def test_both_engines_solve_to_the_same_answer(model, sources):
    answers = {}
    for name in ENGINES:
        with using(name), lps.solve(ROOT / model, sources) as result:
            assert result.is_ok
            primal = result.primal('p')
            answers[name] = (
                result.objective,
                primal.sort(primal.columns),
                result.dual('power_balance').sort('snapshot'),
            )

    (obj_a, primal_a, dual_a), (obj_b, primal_b, dual_b) = answers['polars'], answers['duckdb']
    assert obj_a == pytest.approx(obj_b)
    assert primal_a.equals(primal_b), 'primals differ between engines'
    assert dual_a.equals(dual_b), 'duals differ between engines'


@pytest.mark.parametrize(('model', 'sources'), MODELS)
def test_both_engines_write_the_same_lp_file(model, sources, tmp_path):
    written = {}
    for name in ENGINES:
        with using(name):
            out = lps.write(ROOT / model, sources, tmp_path / f'{name}.lp')
        written[name] = out.read_bytes()
    assert written['polars'] == written['duckdb'], 'the LP files differ byte for byte'


def test_the_env_var_is_the_whole_switch(monkeypatch):
    """`LPSPEC_ENGINE` selects the engine, and nothing else does.

    `build` deliberately takes no engine parameter: an engine cannot change the
    answer, only what computing it costs, so it does not belong in the call
    that produces one. The signature assertion is the part that would notice
    somebody adding it back.
    """
    assert 'engine' not in inspect.signature(lps.build).parameters

    monkeypatch.setenv(engines.ENV_VAR, 'polars')
    with lps.build(ROOT / 'examples/dispatch.yaml', DISPATCH) as ex:
        assert type(ex).__name__ == 'PolarsExecutor'

    monkeypatch.delenv(engines.ENV_VAR)
    with lps.build(ROOT / 'examples/dispatch.yaml', DISPATCH) as ex:
        assert type(ex).__name__ == 'DuckExecutor'


def test_the_engine_option_is_not_silently_a_no_op(pytestconfig):
    """`--engine X` must actually build on X, and this must fail if it stops.

    Everything the suite claims about a second engine rests on having *run* on
    it. If `resolve` ever read the environment at import time instead of call
    time, or the session fixture stopped being applied, every test would still
    pass and the claim would quietly become vacuous — a green suite proving
    nothing. So the switch is asserted outright, in whichever mode the run is in.
    """
    expected = {
        'polars': 'PolarsExecutor',
        'duckdb': 'DuckExecutor',
        None: 'DuckExecutor',
    }[pytestconfig.getoption('--engine')]
    with lps.build(ROOT / 'examples/dispatch.yaml', DISPATCH) as ex:
        assert type(ex).__name__ == expected


def test_a_typo_in_the_env_var_says_where_it_came_from(monkeypatch):
    """Otherwise an unknown name in a shell profile reads as a library bug."""
    monkeypatch.setenv(engines.ENV_VAR, 'ducdkb')
    with pytest.raises(ValueError, match=r"unknown engine 'ducdkb' \(from LPSPEC_ENGINE\)"):
        lps.build(ROOT / 'examples/dispatch.yaml', DISPATCH)


def test_an_unknown_engine_names_the_ones_that_exist(monkeypatch):
    monkeypatch.setenv(engines.ENV_VAR, 'nope')
    with pytest.raises(ValueError, match=r"available: 'duckdb', 'polars'"):
        lps.build(ROOT / 'examples/dispatch.yaml', DISPATCH)

"""What the duckdb engine owes its connection, rather than its caller.

`test_engine_parity.py` asks whether the two engines answer the same; this asks
about the one thing that is duckdb's alone — that a build's queries read
duckdb's own storage and never reach back into a Python object mid-plan.

Reached through `LPSPEC_ENGINE`, the switch a caller has, set explicitly rather
than left to the default: these assertions are about this engine whatever the
default happens to be, and the `--engine polars` pass must not turn them into
assertions about the other one.
"""

from __future__ import annotations

import polars as pl
import pytest

import lpspec as lps
from lpspec.relational import engines

MODEL = {
    'dimensions': {'i': {'dtype': 'int', 'values': [0, 1, 2]}},
    'parameters': {'b': {'dims': ['i']}},
    'variables': {'x': {'foreach': ['i'], 'bounds': {'lower': 0, 'upper': 'b'}}},
    'constraints': {'c': {'foreach': ['i'], 'expression': 'x >= b'}},
    'objectives': {'o': {'sense': 'minimize', 'expression': 'sum(x, over=i)'}},
}
SOURCES = {'b': pl.DataFrame({'i': [0, 1, 2], 'value': [1.0, 2.0, 3.0]})}


@pytest.fixture
def duck(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(engines.ENV_VAR, 'duckdb')
    with lps.build(MODEL, SOURCES) as ex:
        assert type(ex).__name__ == 'DuckExecutor', 'the env var did not select the engine'
        yield ex


def test_every_bound_source_is_a_table_and_not_a_registered_frame(duck) -> None:
    """A scan of a Python object needs the GIL the caller is holding.

    `con.register(name, frame)` leaves the frame a Python object that duckdb
    reads through the buffer protocol from whichever worker thread the scan
    lands on — and those threads need the GIL, which the thread inside
    `execute` is holding while it waits for them. A plan with several such
    scans in one pipeline deadlocks outright: `transport/l` reproduces it as a
    build sitting at 0% CPU indefinitely, while the same query over copies
    returns in 0.3 s.

    A registered frame is not in the catalog, so this is checkable without
    reproducing the hang — which as a test would be indistinguishable from CI
    wedging.
    """
    tables = {name for (name,) in duck._con.execute('SELECT table_name FROM duckdb_tables()').fetchall()}
    named = {*duck._q.dimensions.values(), *duck._q.parameters.values()}
    assert named <= tables, f'{sorted(named - tables)} are registered frames, not tables'


def _labels_in_catalog(ex) -> set[str]:
    """The label relations this build wrote down, as opposed to derived."""
    tables = {name for (name,) in ex._con.execute('SELECT table_name FROM duckdb_tables()').fetchall()}
    return {n for n in (*ex._var_tables.values(), *ex._row_tables.values()) if n in tables}


def test_an_arithmetic_label_is_derived_and_a_counted_one_is_written_down(duck, monkeypatch) -> None:
    """Materialise the window, not the rectangle.

    A label relation is read three or four times downstream, which sounds like
    a table. But the *counted* label needs an ordered window over the whole
    coordinate product and is worth paying for once, while an arithmetic one is
    a cross join of two small relations — cheaper to answer three times than to
    write ten million rows down once. Both are named relations and nothing
    above this can tell which.

    Pinned because the cost is invisible at test sizes: a table here builds the
    right model and costs 62% of the build on `dispatch/l`.
    """
    assert _labels_in_catalog(duck) == set(), 'an unmasked label was materialised'

    # a mask reading the leading dim leaves no free prefix, so it is counted
    counted = {
        **MODEL,
        'variables': {'x': {'foreach': ['i'], 'where': 'b > 0', 'bounds': {'lower': 0, 'upper': 'b'}}},
    }
    monkeypatch.setenv(engines.ENV_VAR, 'duckdb')
    with lps.build(counted, SOURCES) as ex:
        # the constraint counts too: a masked variable in the equation takes
        # its row with it, and which rows survive is not known until it is read
        assert _labels_in_catalog(ex) == {ex._var_tables['x'], ex._row_tables['c']}, (
            'a counted label was not materialised'
        )


def test_nothing_stays_registered_after_a_build(duck) -> None:
    """The copy is a step, not a second home for the data.

    A source left registered would keep the caller's frame alive for the life
    of the connection — a second copy of every parameter in a process that
    chose this engine to hold fewer.
    """
    import duckdb

    with pytest.raises(duckdb.CatalogException, match='does not exist'):
        duck._con.execute('SELECT * FROM "__source par_b__"')

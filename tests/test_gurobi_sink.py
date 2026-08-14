"""The ``gurobi`` sink, against the sink that was already here.

Two sinks loading one :class:`ModelTables` must produce the same model, so
HiGHS is the oracle for Gurobi the way linopy is the oracle for the math: the
interesting assertions here are agreements, not values. Where a value *is*
asserted it comes from ``examples/ports/references.json`` — somebody else's
published optimum, which neither sink can talk the other into.

Every test skips without ``gurobipy``. It ships a size-limited licence in its
own wheel, which is what makes this runnable in CI at all, so the models here
stay small enough for it — a few hundred columns, where the limit is 2000.
"""

from __future__ import annotations

import builtins
import gc
import weakref
from typing import Any

import polars as pl
import pytest

import lpspec as lps
from lpspec.errors import LpspecError, NoSolutionError
from lpspec.relational.sinks.solvers.gurobi import build_gurobi
from tests.conftest import port_sources

gurobipy = pytest.importorskip('gurobipy', reason='the gurobi sink needs the [gurobi] extra')


LP = {
    'dimensions': {'t': {'dtype': 'int', 'values': [0, 1, 2]}},
    'parameters': {'load': {'dims': ['t']}, 'price': {'dims': ['t']}},
    'variables': {'p': {'foreach': ['t'], 'bounds': {'lower': 0, 'upper': 100}}},
    'constraints': {'meet': {'foreach': ['t'], 'expression': 'p >= load'}},
    'objectives': {'cost': {'sense': 'minimize', 'expression': 'sum(p * price, over=t)'}},
}

#: Maximisation *and* an objective constant, which are the two things the
#: sink states outside the frames: ``ModelSense`` and ``ObjCon``.
MAX = {
    'dimensions': {'t': {'dtype': 'int', 'values': [0, 1]}},
    'parameters': {'cap': {'dims': ['t']}},
    'variables': {'p': {'foreach': ['t'], 'bounds': {'lower': 0, 'upper': 10}}},
    'constraints': {'lim': {'foreach': ['t'], 'expression': 'p <= cap'}},
    'objectives': {'profit': {'sense': 'maximize', 'expression': 'sum(p, over=t) + 5'}},
}

MIP = {
    'dimensions': {'i': {'dtype': 'int', 'values': [0, 1, 2]}, 'one': {'dtype': 'int', 'values': [0]}},
    'parameters': {'w': {'dims': ['i']}, 'cap': {'dims': ['one']}},
    'variables': {'x': {'foreach': ['i'], 'domain': 'binary'}},
    'constraints': {'budget': {'foreach': ['one'], 'expression': 'sum(x * w, over=i) <= cap'}},
    'objectives': {'o': {'sense': 'maximize', 'expression': 'sum(x * w, over=i)'}},
}

#: The optimum parks the semi-continuous variable at exactly 0 — infeasible
#: were its lower bound of 10 ordinary, and cheaper to run it at 5 were the
#: bound all there is — so agreement here is agreement on the zero-or-banded
#: semantics, not on the domain parsing.
SEMI = {
    'dimensions': {'one': {'dtype': 'int', 'values': [0]}},
    'parameters': {'load': {'dims': ['one']}},
    'variables': {
        'p': {'foreach': ['one'], 'bounds': {'lower': 10, 'upper': 100}, 'domain': 'semi_continuous'},
        'q': {'foreach': ['one'], 'bounds': {'lower': 0, 'upper': 100}},
    },
    'constraints': {'balance': {'foreach': ['one'], 'expression': 'p + q == load'}},
    'objectives': {'c': {'sense': 'minimize', 'expression': 'sum(p + 5 * q, over=one)'}},
}

INFEASIBLE = {
    'dimensions': {'t': {'dtype': 'int', 'values': [0]}},
    'parameters': {'load': {'dims': ['t']}},
    'variables': {'p': {'foreach': ['t'], 'bounds': {'lower': 0, 'upper': 1}}},
    'constraints': {'meet': {'foreach': ['t'], 'expression': 'p == load'}},
    'objectives': {'c': {'sense': 'minimize', 'expression': 'p'}},
}

#: Each case is the ``(model, data)`` pair a call site unpacks:
#: ``lps.solve(*CASES['MIP'])``.
CASES: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {
    'LP': (
        LP,
        {
            'load': pl.DataFrame({'t': [0, 1, 2], 'value': [1.0, 2.0, 3.0]}),
            'price': pl.DataFrame({'t': [0, 1, 2], 'value': [10.0, 20.0, 30.0]}),
        },
    ),
    'MAX': (MAX, {'cap': pl.DataFrame({'t': [0, 1], 'value': [3.0, 4.0]})}),
    'MIP': (
        MIP,
        {
            'w': pl.DataFrame({'i': [0, 1, 2], 'value': [2.0, 3.0, 4.0]}),
            'cap': pl.DataFrame({'one': [0], 'value': [5.0]}),
        },
    ),
    'SEMI': (SEMI, {'load': pl.DataFrame({'one': [0], 'value': [5.0]})}),
    'INFEASIBLE': (INFEASIBLE, {'load': pl.DataFrame({'t': [0], 'value': [99.0]})}),
}


# ---------------------------------------------------------------------------
# the two sinks answer the same question the same way
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('name', 'variable', 'constraint'),
    [('LP', 'p', 'meet'), ('MAX', 'p', 'lim'), ('MIP', 'x', None), ('SEMI', 'p', None)],
)
def test_gurobi_and_highs_agree(name: str, variable: str, constraint: str | None) -> None:
    """The claim the second solver has to earn, on all three quantities.

    Coordinates as well as values, since a sink that loaded the columns in a
    different order would still reach the same objective on these models — and
    duals under ``maximize``, where a sign convention could differ and nothing
    else in the suite would notice.
    """
    with lps.solve(*CASES[name]) as highs, lps.solve(*CASES[name], solver_name='gurobi') as gb:
        assert gb.termination_condition == highs.termination_condition
        assert gb.objective == pytest.approx(highs.objective)

        expected, got = highs.primal(variable), gb.primal(variable)
        assert got.columns == expected.columns
        assert got.drop('value').equals(expected.drop('value'))
        assert got['value'].to_list() == pytest.approx(expected['value'].to_list())

        if constraint:
            assert gb.dual(constraint)['value'].to_list() == pytest.approx(highs.dual(constraint)['value'].to_list())


def test_every_port_reaches_its_reference_optimum_on_gurobi(port: dict[str, Any]) -> None:
    """``test_ports.py``'s corpus, solved by the other solver.

    The one assertion here no part of this package produced. A sink that
    mis-loads the matrix — a block boundary off by a row, a sense inverted —
    still reaches *a* number; this is what that number is checked against.
    """
    with lps.solve(port['model'], port_sources(port['name']), solver_name='gurobi') as solution:
        assert solution.is_ok, f'{port["name"]} did not solve: {solution.status}'
        assert solution.objective == pytest.approx(port['objective'], rel=port['rtol'])


def test_block_boundaries_do_not_move_the_answer() -> None:
    """``batch_rows=1`` forces one block per row, so every CSR view is built at
    a boundary — where an off-by-one in ``indptr`` shifts coefficients into the
    neighbouring row rather than dropping them."""
    with lps.build(*CASES['LP']) as bound:
        whole = bound.solve(solver_name='gurobi')
        ragged = bound._engine.solve('gurobi', batch_rows=1)
        assert ragged.objective == pytest.approx(whole.objective)
        assert ragged.primal('p')['value'].to_list() == pytest.approx(whole.primal('p')['value'].to_list())


# ---------------------------------------------------------------------------
# what the sink says when there is nothing to read
# ---------------------------------------------------------------------------


def test_an_infeasible_solve_reports_both_axes_in_gurobis_wording() -> None:
    with lps.solve(*CASES['INFEASIBLE'], solver_name='gurobi') as solution:
        assert solution.status == 'warning'
        assert solution.termination_condition == 'infeasible'
        assert not solution.has_primal
        assert solution.objective != solution.objective, 'nan, not 0.0'
        with pytest.raises(NoSolutionError, match='INFEASIBLE'):
            solution.primal('p')


def test_a_mixed_integer_model_has_no_duals() -> None:
    """Gurobi refuses ``Pi`` rather than returning zeros; the sink passes the
    refusal on as the ``None`` that makes ``dual`` explain itself."""
    with lps.solve(*CASES['MIP'], solver_name='gurobi') as solution:
        assert solution.has_primal
        with pytest.raises(LpspecError, match='mixed-integer'):
            solution.dual('budget')


def test_solver_options_reach_gurobi() -> None:
    """Verbatim, in Gurobi's own vocabulary — ``TimeLimit``, not HiGHS'
    ``time_limit``. Forwarding is the contract; translating names is not, and
    an option the solver does not know reaches the caller as the solver's own
    complaint rather than as a guess at what was meant."""
    with lps.solve(*CASES['MIP'], solver_options={'TimeLimit': 0.0}, solver_name='gurobi') as solution:
        assert solution.termination_condition == 'time_limit'
    with pytest.raises(gurobipy.GurobiError, match='no_such_parameter'):
        lps.solve(*CASES['MIP'], solver_options={'no_such_parameter': 1}, solver_name='gurobi')


# ---------------------------------------------------------------------------
# the seams: the build without the search, and choosing a sink at all
# ---------------------------------------------------------------------------


def test_solver_options_land_on_the_environment() -> None:
    """Where a licence parameter has to go.

    ``WLSAccessID`` / ``ComputeServer`` / ``TokenServer`` can only be set
    before an environment starts, so applying options to the *model* — as this
    sink first did — locks out every Compute-Server and WLS user. Asserted
    through an ordinary parameter, since a licence one would need a licence:
    the model sees it as its default, which is what environment-level means.
    """
    with lps.build(*CASES['MIP']) as bound:
        assert build_gurobi(bound._engine._tables(), solver_options={'TimeLimit': 5.0}).Params.TimeLimit == 5.0


def test_build_gurobi_loads_the_model_and_stops() -> None:
    """`bench/`'s seam: the hand-off with no search behind it, so what it
    reports is what was loaded rather than what was solved."""
    with lps.build(*CASES['MIP']) as bound:
        tables = bound._engine._tables()
        m = build_gurobi(tables)
        assert (m.NumVars, m.NumConstrs) == (tables.column_count, tables.row_count)
        assert m.NumIntVars == tables.cols.filter(pl.col('vtype') != 'continuous').height
        assert m.ModelSense == gurobipy.GRB.MAXIMIZE
        assert m.SolCount == 0


def test_nothing_keeps_a_built_model_alive() -> None:
    """The precondition for releasing the licence a built model holds.

    :func:`build_gurobi` hands ownership over and disposes the environment
    through a finalizer on the model, so anything in this package still
    referencing that model would hold a Gurobi licence open for the life of
    the process. A held :class:`Gurobi` does not depend on this — its
    ``close()`` disposes both explicitly.
    """
    with lps.build(*CASES['MIP']) as bound:
        reference = weakref.ref(build_gurobi(bound._engine._tables()))
    gc.collect()
    assert reference() is None, 'a built gurobi model outlived its caller — its environment cannot be released'


def test_the_objective_constant_rides_on_the_model_not_the_answer() -> None:
    """Gurobi has ``ObjCon``, so the constant is part of the model it holds —
    which makes the build seam a complete hand-off rather than a model plus a
    number to remember."""
    with lps.build(*CASES['MAX']) as bound:
        assert build_gurobi(bound._engine._tables()).ObjCon == pytest.approx(5.0)


def test_the_missing_extra_is_named() -> None:
    """What a caller without gurobipy meets — both halves named, since the
    absent one is as often scipy."""
    real_import = builtins.__import__

    def refuse(name: str, *args: Any, **kwargs: Any) -> Any:
        if name in {'gurobipy', 'scipy.sparse'}:
            raise ModuleNotFoundError(f'No module named {name!r}')
        return real_import(name, *args, **kwargs)

    with lps.build(*CASES['LP']) as bound, pytest.MonkeyPatch.context() as patch:
        patch.setattr(builtins, '__import__', refuse)
        with pytest.raises(ModuleNotFoundError, match=r'\[gurobi\] extra \(gurobipy, scipy\)'):
            bound.solve(solver_name='gurobi')

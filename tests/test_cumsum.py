"""cumsum: the free half of the window family — running sums over data (#384).

`cumsum(p, over=d)` reduces a **variable-free** expression in the dimension's
declared coordinate order, emitting one data column with no term expansion
(SPEC §7). The variable-carrying case is a load error naming the state-variable
recurrence, and that refusal is the operator's whole design: over a variable
row *t* would carry *t* terms, O(T²) nonzeros for what a recurrence says in
O(T).
"""

from __future__ import annotations

import numpy as np
import pytest

import lpspec as lps
from lpspec.errors import LanguageError
from lpspec.language.helpers import cumsum_over_variable_message
from lpspec.lowering import _lower_expr
from lpspec.relational.plan import CumulativeSum, Parameter
from tests.conftest import by_coord, resolved, schema_of
from tests.differential import differential
from tests.oracle import lpspec_linopy, pd

CUMSUM_MODEL = {
    'dimensions': {'year': {'dtype': 'int', 'values': [2025, 2026, 2027, 2028, 2029]}},
    'parameters': {'build_rate': {'dims': ['year']}},
    'expressions': {'installed': 'cumsum(build_rate, over=year)'},
    'variables': {'p': {'foreach': ['year'], 'bounds': {'lower': 0, 'upper': 1000}}},
    'constraints': {'cap': {'foreach': ['year'], 'expression': 'p <= installed'}},
    'objectives': {'o': {'sense': 'maximize', 'expression': 'sum(p, over=year)'}},
}

CUMSUM_SCHEMA = schema_of(CUMSUM_MODEL)


# ---------------------------------------------------------------------------
# the reduction, end to end
# ---------------------------------------------------------------------------


def test_a_cumulative_capacity_matches_the_externally_computed_column():
    """The issue's own example: installed capacity is the running sum of builds.

    `p <= cumsum(build_rate, over=year)` under a maximizing objective pins `p`
    to the cumulative column exactly, so agreement with `np.cumsum` of the same
    data is the whole assertion — no pre-computed second parameter shipped, no
    data-prep round trip. Both lanes and the written LP file must agree.
    """
    years = [2025, 2026, 2027, 2028, 2029]
    rate = [5.0, 3.0, 0.0, 7.0, 2.0]
    data = {'build_rate': pd.Series(rate, index=pd.Index(years, name='year'))}

    with differential(CUMSUM_MODEL, data, lp=True) as run:
        got = by_coord(run.result, 'p', 'year')
        for year, expected in zip(years, np.cumsum(rate), strict=True):
            assert got[year] == pytest.approx(expected), f'p[{year}] must sit at the cumulative build'


def test_the_reduction_walks_declared_order_not_sorted_order():
    """String labels whose declared order differs from their sorted order.

    The running sum is over the dimension's declared coordinate order — the
    same order `shift` reads positionally (SPEC §8) — so a lane that sorted the
    labels first would accumulate in the wrong order and disagree here.
    """
    labels = ['t2', 't10', 't1']
    assert sorted(labels) != labels, 'the fixture is only a fixture if sorted != declared'
    rate = [5.0, 3.0, 2.0]
    expected = {'t2': 5.0, 't10': 8.0, 't1': 10.0}

    model = {
        'dimensions': {'t': {'dtype': 'str', 'values': labels}},
        'parameters': {'w': {'dims': ['t']}},
        'variables': {'x': {'foreach': ['t'], 'bounds': {'lower': 0, 'upper': 1000}}},
        'constraints': {'c': {'foreach': ['t'], 'expression': 'x <= cumsum(w, over=t)'}},
        'objectives': {'o': {'sense': 'maximize', 'expression': 'sum(x, over=t)'}},
    }
    data = {'w': pd.Series(rate, index=pd.Index(labels, name='t'))}

    with differential(model, data) as run:
        got = by_coord(run.result, 'x', 't')
        for label, value in expected.items():
            assert got[label] == pytest.approx(value), f'x[{label}] must follow declared order, not sorted order'


def test_a_single_element_dimension_is_its_own_running_sum():
    model = {
        'dimensions': {'t': {'dtype': 'int', 'values': [0]}},
        'parameters': {'w': {'dims': ['t']}},
        'variables': {'x': {'foreach': ['t'], 'bounds': {'lower': 0, 'upper': 1000}}},
        'constraints': {'c': {'foreach': ['t'], 'expression': 'x <= cumsum(w, over=t)'}},
        'objectives': {'o': {'sense': 'maximize', 'expression': 'sum(x, over=t)'}},
    }
    data = {'w': pd.Series([4.0], index=pd.Index([0], name='t'))}

    with differential(model, data) as run:
        assert by_coord(run.result, 'x', 't')[0] == pytest.approx(4.0)


def test_a_missing_operand_row_contributes_zero_in_a_coefficient_position():
    """A sparse operand reads as zero, the identity of the sum it feeds (law 8).

    `w` has no row at t=1, so the running sum holds flat there — [2, 2, 5] —
    and both lanes must agree without a fill shipped in the data.
    """
    model = {
        'dimensions': {'t': {'dtype': 'int', 'values': [0, 1, 2]}},
        'parameters': {'w': {'dims': ['t']}},
        'variables': {'x': {'foreach': ['t'], 'bounds': {'lower': 0, 'upper': 5}}},
        'constraints': {'c': {'foreach': ['t'], 'expression': 'x * cumsum(w, over=t) <= 10'}},
        'objectives': {'o': {'sense': 'maximize', 'expression': 'sum(x, over=t)'}},
    }
    data = {'w': pd.Series({0: 2.0, 2: 3.0})}

    with differential(model, data) as run:
        got = by_coord(run.result, 'x', 't')
        assert got[0] == pytest.approx(5.0), 't=0: 10/2 exceeds the bound, so the bound governs'
        assert got[1] == pytest.approx(5.0), 't=1: the running sum holds at 2, same row as t=0'
        assert got[2] == pytest.approx(2.0), 't=2: the running sum is 5, so 10/5'


# ---------------------------------------------------------------------------
# the variable-free restriction
# ---------------------------------------------------------------------------


def _cumsum_over(operand: str) -> dict[str, object]:
    return {
        'dimensions': {'t': {'dtype': 'int', 'values': [0, 1, 2]}},
        'parameters': {'w': {'dims': ['t']}},
        'variables': {'x': {'foreach': ['t'], 'bounds': {'lower': 0, 'upper': 5}}},
        'constraints': {'c': {'foreach': ['t'], 'expression': f'cumsum({operand}, over=t) <= 10'}},
        'objectives': {'o': {'sense': 'maximize', 'expression': 'sum(x, over=t)'}},
    }


@pytest.mark.parametrize(
    'operand',
    ['x', 'w * x', 'w + x'],
    ids=['a-bare-variable', 'a-product-carrying-a-variable', 'a-sum-carrying-a-variable'],
)
def test_cumsum_over_a_variable_carrying_subtree_is_a_load_error(operand):
    with pytest.raises(LanguageError, match='state variable'):
        lps.check(_cumsum_over(operand))


def test_the_refusal_names_the_recurrence_and_the_refused_window():
    """The message is the migration story, so its two alternatives are held as wording.

    The state-variable recurrence is the O(T) rewrite — a storage SOC balance
    is exactly that shape — and the unbounded window over a variable stays
    refused unless the budget question (#380) is ever answered.
    """
    with pytest.raises(LanguageError) as exc:
        lps.check(_cumsum_over('x'))

    message = str(exc.value)
    assert 'shift(total, over=d, by=1, edge=0)' in message, 'the recurrence rewrite has to be spelled out'
    assert 'SOC' in message, 'the reader arriving from storage modelling has to recognise the rewrite'
    assert '#380' in message, 'the refused variable case has to point at the budget question that holds it'
    assert 'O(T²)' in message, 'the cost being refused has to be named, not alluded to'


def test_the_eager_lane_speaks_the_same_refusal(tmp_path):
    """Hard rule 3: one language. The eager lane refuses at build with the identical wording."""
    import yaml as pyyaml

    path = tmp_path / 'model.yaml'
    path.write_text(pyyaml.safe_dump(_cumsum_over('x')))
    with pytest.raises(LanguageError) as exc:
        lpspec_linopy.build(path, data={'w': pd.Series([1.0, 1.0, 1.0], index=pd.Index([0, 1, 2], name='t'))})
    assert cumsum_over_variable_message() in str(exc.value), 'both lanes must speak the one wording'


# ---------------------------------------------------------------------------
# lowering
# ---------------------------------------------------------------------------


def test_cumsum_lowers_to_a_cumulative_sum_node():
    node = _lower_expr(resolved('cumsum(build_rate, over=year)', CUMSUM_SCHEMA), CUMSUM_SCHEMA, 't')
    assert node == CumulativeSum(Parameter('build_rate'), 'year')


def test_a_hand_built_call_whose_over_is_not_a_dimension_node_is_refused():
    """Purpose-built probe: resolution always types `over=`, so only a
    hand-built AST reaches lowering with anything else there — the same guard
    every helper's lowering case keeps."""
    from lpspec.language.expression_parser import FunctionCallNode, NameNode, ParameterNode

    call = FunctionCallNode('cumsum', [ParameterNode('build_rate')], {'over': NameNode('year')})
    with pytest.raises(LanguageError, match=r'cumsum\(over=\.\.\.\) must name a dimension'):
        _lower_expr(call, CUMSUM_SCHEMA, 't')


def test_a_hand_built_eager_call_with_a_scalar_operand_is_refused():
    """Purpose-built probe: `_eval_ast` hands the helper a DataArray always, so
    only a hand-built call reaches the operand-shape guard."""
    from lpspec.linopy.builder import _helper_cumsum

    with pytest.raises(TypeError, match=r'cumsum\(\) does not support'):
        _helper_cumsum(3.0, over='t')


@pytest.mark.parametrize(
    ('expression', 'match'),
    [
        pytest.param(
            'cumsum(build_rate, over=nope)',
            r'cumsum\(over=nope\) does not name a declared dimension',
            id='over-names-no-dimension',
        ),
        # A dimensionless operand cannot carry `year`, so the reduction has
        # nothing to walk — the same rule `shift` states for a missing dim.
        pytest.param(
            'cumsum(2, over=year)',
            'but the expression has dims',
            id='a-dim-the-expression-lacks',
        ),
        pytest.param(
            'cumsum(build_rate)',
            r'cumsum\(\) expects',
            id='the-over-kwarg-is-required',
        ),
    ],
)
def test_a_cumsum_neither_lane_can_honour_is_refused_at_load(expression, match):
    with pytest.raises(LanguageError, match=match):
        _lower_expr(resolved(expression, CUMSUM_SCHEMA), CUMSUM_SCHEMA, 't')

"""domain: semi_continuous — zero, or between the declared bounds, nothing in between.

The model is the shape the domain exists for: generators with a minimum stable
output of 10 that the optimum must switch *off* in one snapshot and dispatch
inside the band in another. Said with an ordinary lower bound the off snapshot
is infeasible, so agreement here is agreement on the semantics, not on the
domain parsing.
"""

from __future__ import annotations

import pytest

from tests.conftest import by_coord
from tests.differential import differential

SEMI_YAML = """
dimensions:
  snapshot: {dtype: int}
  generator: {dtype: str}

parameters:
  p_max: {dims: [generator]}
  cost: {dims: [generator]}
  load: {dims: [snapshot]}

variables:
  p:
    foreach: [snapshot, generator]
    bounds: {lower: 10, upper: p_max}
    domain: semi_continuous
  unserved:
    foreach: [snapshot]
    bounds: {lower: 0, upper: 1000}

constraints:
  balance:
    foreach: [snapshot]
    expression: sum(p, over=generator) + unserved == load

objectives:
  total_cost:
    sense: minimize
    expression: sum(p * cost, over=generator) + 50 * unserved
"""


@pytest.fixture
def semi_inputs():
    """Two generators with a shared minimum of 10, two regimes.

    Snapshot 0's load of 5 sits *under* the minimum, so the optimum must park
    both generators at exactly 0 and pay for unserved energy — infeasible if
    the lower bound were ordinary. Snapshot 1's load of 60 puts `base` inside
    its band.
    """
    import pandas as pd

    data = {
        'p_max': pd.Series({'base': 100.0, 'peaker': 100.0}),
        'cost': pd.Series({'base': 1.0, 'peaker': 5.0}),
        'load': pd.Series([5.0, 60.0], index=pd.RangeIndex(2, name='snapshot')),
    }
    coords = {
        'snapshot': pd.RangeIndex(2, name='snapshot'),
        'generator': pd.Index(['base', 'peaker'], name='generator'),
    }
    return data, coords


def test_the_optimum_switches_the_min_run_generators_off_or_dispatches_them(semi_inputs):
    """Both lanes agree, and the primal proves the zero-or-banded semantics."""
    data, coords = semi_inputs

    with differential(SEMI_YAML, data, coords, lp=True) as run:
        p = by_coord(run.result, 'p', 'snapshot', 'generator')
        unserved = by_coord(run.result, 'unserved', 'snapshot')

        assert p[(0, 'base')] == pytest.approx(0.0, abs=1e-6), (
            'load 5 sits under the minimum of 10, so semi-continuity must park base at exactly 0 '
            '— an ordinary lower bound makes this snapshot infeasible'
        )
        assert p[(0, 'peaker')] == pytest.approx(0.0, abs=1e-6)
        assert unserved[0] == pytest.approx(5.0, rel=1e-6)
        assert p[(1, 'base')] == pytest.approx(60.0, rel=1e-6), 'inside [10, 100], the band behaves as plain bounds'
        assert p[(1, 'peaker')] == pytest.approx(0.0, abs=1e-6)
        assert unserved[1] == pytest.approx(0.0, abs=1e-6)

        text = run.lp.read_text()
        assert '\nsemi-continuous\n' in text, 'the LP file carries the domain in its own section'
        assert text.index('\nbounds\n') < text.index('\nsemi-continuous\n'), (
            'the section follows bounds, where both HiGHS and linopy place it'
        )

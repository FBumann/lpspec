"""Phase-2 gate: two real models round-trip through solve on the relational backend.

Each model is built three ways and must agree on the objective:
  1. relational executor -> the `highs` solver (batched addCols/addRows)
  2. relational executor -> lp_file sink -> HiGHS reads and solves the file
  3. eager linopy build (the correctness oracle)
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import polars as pl
import pytest

import lpspec as lps
from lpspec.errors import DataError, LanguageError, LpspecError
from lpspec.language.model import Model
from lpspec.lowering import lower_program
from lpspec.relational import (
    PolarsExecutor,
    chunking,
)
from lpspec.relational.plan import (
    Constant,
    ConstraintDeclaration,
    DimensionDeclaration,
    GroupSum,
    ObjectiveDeclaration,
    Parameter,
    ParameterComparison,
    ParameterDeclaration,
    Program,
    Sum,
    Variable,
    VariableDeclaration,
)
from lpspec.relational.sinks import SOLVERS
from tests.conftest import by_coord, override, solve_lp_file
from tests.differential import RTOL, differential
from tests.oracle import linopy, pd, transport_eager_objective, xr

# ---------------------------------------------------------------------------
# model 1: dispatch (the spec example)
# ---------------------------------------------------------------------------


@pytest.fixture
def dispatch_data():
    """The spec's dispatch instance, with one generator retired to `p_max = 0`.

    That row is what the model's `where` masks out.
    """
    rng = np.random.default_rng(7)
    n_s, n_g = 40, 6
    gens = pd.DataFrame(
        {
            'generator': [f'g{i}' for i in range(n_g)],
            'p_max': rng.uniform(50, 200, n_g).round(3),
            'cost': rng.uniform(5, 100, n_g).round(3),
        }
    )
    gens.loc[1, 'p_max'] = 0.0
    load = pd.DataFrame(
        {
            'snapshot': np.arange(n_s),
            'value': (rng.uniform(0.2, 0.7, n_s) * gens['p_max'].sum()).round(3),
        }
    )
    return gens, load


def dispatch_program() -> Program:
    return Program(
        parameters=(
            ParameterDeclaration('p_max', ('generator',)),
            ParameterDeclaration('cost', ('generator',)),
            ParameterDeclaration('load', ('snapshot',)),
        ),
        variables=(
            VariableDeclaration(
                'p',
                ('snapshot', 'generator'),
                where=ParameterComparison('p_max', '>', 0),
                lower=Constant(0.0),
                upper=Parameter('p_max'),
            ),
        ),
        constraints=(
            ConstraintDeclaration(
                'power_balance',
                ('snapshot',),
                lhs=Sum(Variable('p'), over=('generator',)),
                sense='==',
                rhs=Parameter('load'),
            ),
        ),
        objective=ObjectiveDeclaration('min', Sum(Variable('p') * Parameter('cost'), over=('generator', 'snapshot'))),
    )


def dispatch_sources(gens: pd.DataFrame, load: pd.DataFrame) -> dict:
    return {
        'p_max': gens[['generator', 'p_max']].rename(columns={'p_max': 'value'}),
        'cost': gens[['generator', 'cost']].rename(columns={'cost': 'value'}),
        'load': load,
        'snapshot': load[['snapshot']],
    }


def dispatch_eager_objective(gens: pd.DataFrame, load: pd.DataFrame) -> float:
    gi = gens.set_index('generator')
    li = load.set_index('snapshot')['value']
    p_max = xr.DataArray.from_series(gi['p_max'])
    cost = xr.DataArray.from_series(gi['cost'])
    load_da = xr.DataArray.from_series(li)

    m = linopy.Model()
    mask = (p_max > 0).broadcast_like(load_da * p_max)
    p = m.add_variables(lower=0, upper=p_max, coords=[li.index, gi.index], name='p', mask=mask)
    m.add_constraints(p.sum('generator') == load_da, name='power_balance')
    m.add_objective((p * cost).sum())
    m.solve(solver_name='highs', output_flag=False)
    return float(m.objective.value)


def test_dispatch_roundtrip(dispatch_data, tmp_path):
    """Solver and LP file agree with the eager oracle, down to the dispatch."""
    gens, load = dispatch_data
    oracle = dispatch_eager_objective(gens, load)

    with PolarsExecutor() as ex:
        ex.build(dispatch_program(), dispatch_sources(gens, load))

        result = ex.solve()
        assert result.is_ok
        assert result.objective == pytest.approx(oracle, rel=RTOL)

        lp = tmp_path / 'dispatch.lp'
        ex.write(lp)
        assert solve_lp_file(lp) == pytest.approx(oracle, rel=RTOL)

        primal = result.to_pandas('p')
        n_active = int((gens['p_max'] > 0).sum())
        assert len(primal) == n_active * len(load), 'a masked variable row is absent, not zero'
        assert set(primal.columns) == {'snapshot', 'generator', 'value'}, 'primal joins back to the coords'
        balance = primal.groupby('snapshot')['value'].sum()
        expected = load.set_index('snapshot')['value']
        assert np.allclose(balance.sort_index(), expected.sort_index()), 'dispatch meets load in every snapshot'


# ---------------------------------------------------------------------------
# model 2: multi-bus transport (exercises GroupSum and signed flows)
# ---------------------------------------------------------------------------


def transport_program() -> Program:
    injection = (
        GroupSum(Variable('p'), over='generator', coordinate='bus', into='bus')
        + GroupSum(Variable('f'), over='line', coordinate='to', into='bus')
        - GroupSum(Variable('f'), over='line', coordinate='from', into='bus')
    )
    return Program(
        parameters=(
            ParameterDeclaration('p_max', ('generator',)),
            ParameterDeclaration('cost', ('generator',)),
            ParameterDeclaration('cap', ('line',)),
            ParameterDeclaration('load', ('snapshot', 'bus')),
        ),
        variables=(
            VariableDeclaration(
                'p',
                ('snapshot', 'generator'),
                lower=Constant(0.0),
                upper=Parameter('p_max'),
            ),
            VariableDeclaration(
                'f',
                ('snapshot', 'line'),
                lower=-Parameter('cap'),
                upper=Parameter('cap'),
            ),
        ),
        constraints=(
            ConstraintDeclaration(
                'balance',
                ('snapshot', 'bus'),
                lhs=injection,
                sense='==',
                rhs=Parameter('load'),
            ),
        ),
        objective=ObjectiveDeclaration('min', Sum(Variable('p') * Parameter('cost'), over=('generator', 'snapshot'))),
        dimensions=(
            DimensionDeclaration('generator', (('bus', 'bus'),)),
            DimensionDeclaration('line', (('from', 'bus'), ('to', 'bus'))),
        ),
    )


def transport_sources(gens, lines, load) -> dict:
    """The transport instance as tidy sources.

    A dim carrying declared coordinates needs an index source that has them,
    which is why `generator` and `line` ship their mapping columns.
    """
    return {
        'p_max': gens[['generator', 'p_max']].rename(columns={'p_max': 'value'}),
        'cost': gens[['generator', 'cost']].rename(columns={'cost': 'value'}),
        'cap': lines[['line', 'cap']].rename(columns={'cap': 'value'}),
        'load': load,
        'snapshot': load[['snapshot']],
        'bus': load[['bus']],
        'generator': gens[['generator', 'bus']],
        'line': lines[['line', 'from_bus', 'to_bus']].rename(columns={'from_bus': 'from', 'to_bus': 'to'}),
    }


def test_transport_roundtrip(transport_data, tmp_path):
    """Solver and LP file agree with the eager oracle."""
    gens, lines, load = transport_data
    oracle = transport_eager_objective(gens, lines, load)
    assert np.isfinite(oracle), 'oracle model must be feasible'

    with PolarsExecutor() as ex:
        ex.build(transport_program(), transport_sources(gens, lines, load))

        result = ex.solve()
        assert result.is_ok
        assert result.objective == pytest.approx(oracle, rel=RTOL)

        lp = tmp_path / 'transport.lp'
        ex.write(lp)
        assert solve_lp_file(lp) == pytest.approx(oracle, rel=RTOL)

        primal_f = result.to_pandas('f')
        caps = lines.set_index('line')['cap']
        limits = primal_f['line'].map(caps)
        assert (primal_f['value'].abs() <= limits + 1e-6).all()


# ---------------------------------------------------------------------------
# error paths
# ---------------------------------------------------------------------------


def test_nonlinear_product_rejected(dispatch_data):
    gens, load = dispatch_data
    prog = dispatch_program()
    bad = Program(
        parameters=prog.parameters,
        variables=prog.variables,
        constraints=prog.constraints,
        objective=ObjectiveDeclaration('min', Sum(Variable('p') * Variable('p'), over=('generator', 'snapshot'))),
    )
    with PolarsExecutor() as ex, pytest.raises(LanguageError, match='nonlinear'):
        ex.build(bad, dispatch_sources(gens, load))


def test_missing_source_rejected(dispatch_data):
    gens, load = dispatch_data
    sources = dispatch_sources(gens, load)
    del sources['cost']
    with PolarsExecutor() as ex, pytest.raises(DataError, match="no source bound for parameter 'cost'"):
        ex.build(dispatch_program(), sources)


#: A scalar parameter used in a bound and in the objective — the two places a
#: silent row multiplication is least visible.
SCALAR_MODEL = {
    'dimensions': {'i': {'dtype': 'int', 'values': [0, 1]}},
    'parameters': {'s': {'dims': []}},
    'variables': {'x': {'foreach': ['i'], 'bounds': {'lower': 0, 'upper': 's'}}},
    'constraints': {'floor': {'foreach': ['i'], 'expression': 'x >= 1'}},
    'objectives': {'total': {'sense': 'minimize', 'expression': 'sum(x * s, over=i)'}},
}


@pytest.mark.parametrize('rows', [2, 0])
def test_a_dimensionless_parameter_must_be_one_row(rows):
    """No dims means one value broadcast everywhere, and nothing used to check it.

    A dimensionless parameter is broadcast by joining on nothing, which is
    right for one row and a silent row multiplication for two — duplicate
    columns for one variable in a bound, duplicate mask rows in a where. The
    per-coordinate check skipped this case entirely, because a parameter with
    no dims has nothing to group by, which also left `keyed` claiming the
    opposite of what such a source held (#166).
    """
    data = {'s': pl.DataFrame({'value': [1.0] * rows}, schema={'value': pl.Float64})}
    with pytest.raises(DataError, match=f"parameter 's' .* its source has {rows} rows"):
        lps.build(SCALAR_MODEL, data)


def test_a_dimensionless_parameter_of_one_row_still_builds():
    """The control: the shape the check exists to let through."""
    data = {'s': pl.DataFrame({'value': [10.0]})}
    with lps.solve(SCALAR_MODEL, data) as result:
        assert result.objective == pytest.approx(20.0), 'x == 1 at both coordinates of i, times s'


def test_a_dimension_named_n_is_still_a_legal_dimension():
    """No check may claim a legal dim name for a column of its own.

    The duplicate-row check counts rows beside the dims it grouped by, and its
    count column once did: `n` collided with a dim of the same name, and every
    build of such a model — healthy data included — died on an internal polars
    DuplicateError naming no parameter.
    """
    model = {
        'dimensions': {'n': {'dtype': 'int', 'values': [1, 2]}},
        'parameters': {'cost': {'dims': ['n']}},
        'variables': {'x': {'foreach': ['n'], 'bounds': {'lower': 0, 'upper': 5}}},
        'constraints': {'c': {'foreach': ['n'], 'expression': 'x <= 5'}},
        'objectives': {'o': {'sense': 'maximize', 'expression': 'x * cost'}},
    }
    with lps.solve(model, {'cost': pl.DataFrame({'n': [1, 2], 'value': [1.0, 2.0]})}) as result:
        assert result.objective == pytest.approx(15.0)

    # the check the column belongs to still catches its own case
    doubled = {'cost': pl.DataFrame({'n': [1, 1, 2], 'value': [1.0, 9.0, 2.0]})}
    with pytest.raises(DataError, match="parameter 'cost'"):
        lps.build(model, doubled)


def test_out_of_foreach_dims_rejected(dispatch_data):
    """A term carrying a dim the constraint does not `foreach` is refused."""
    gens, load = dispatch_data
    prog = dispatch_program()
    bad = Program(
        parameters=prog.parameters,
        variables=prog.variables,
        constraints=(
            ConstraintDeclaration(
                'power_balance',
                ('snapshot',),
                lhs=Variable('p'),  # generator dim deliberately not summed
                sense='==',
                rhs=Parameter('load'),
            ),
        ),
        objective=prog.objective,
    )
    with PolarsExecutor() as ex, pytest.raises(LanguageError, match='missing a Sum'):
        ex.build(bad, dispatch_sources(gens, load))


def test_an_awkward_path_is_a_value_not_syntax(tmp_path):
    """Paths come from the calling program, so no language rule constrains them.

    ``o'brien`` is a legal directory name, and a quote in one must be as
    uninteresting as a quote in a label. Every path-carrying sink and source is
    exercised here: a parquet source, an explicit index source, the LP writer,
    and the parquet sink.
    """
    odd = tmp_path / "o'brien"
    odd.mkdir()
    pl.DataFrame({'snapshot': [0, 1], 'value': [1.0, 2.0]}).write_parquet(odd / 'load.parquet')
    pl.DataFrame({'snapshot': [0, 1]}).write_parquet(odd / 'index.parquet')

    model = {
        'dimensions': {'snapshot': {'dtype': 'int'}},
        'parameters': {'load': {'dims': ['snapshot']}},
        'variables': {'p': {'foreach': ['snapshot'], 'bounds': {'lower': 0}}},
        'constraints': {'meet': {'foreach': ['snapshot'], 'expression': 'p >= load'}},
        'objectives': {'c': {'sense': 'minimize', 'expression': 'sum(p, over=snapshot)'}},
    }
    sources = {'load': str(odd / 'load.parquet'), 'snapshot': str(odd / 'index.parquet')}

    lps.write(model, sources, odd / 'model.lp')
    result = lps.solve(model, sources)
    assert result.objective == pytest.approx(3.0)
    assert set(result.to_parquet(odd / 'solution')) == {'p'}


def test_a_variable_appearing_twice_in_a_row_is_summed_not_duplicated():
    """The case the skipped aggregate must not break.

    `x + 2 * x` is two term fragments landing on one solver column, so the
    assembly has to add them. Its coefficient must be 3, and the row must hold
    one entry for that column rather than two — a solver handed the same
    column twice in one row is entitled to reject the model.
    """
    model = {
        'dimensions': {'i': {'dtype': 'int', 'values': [0, 1]}},
        'parameters': {'rhs': {'dims': ['i']}},
        'variables': {'x': {'foreach': ['i'], 'bounds': {'lower': 0}}},  # no upper: +inf
        'constraints': {'c': {'foreach': ['i'], 'expression': 'x + 2 * x >= rhs'}},
        'objectives': {'o': {'sense': 'minimize', 'expression': 'sum(x, over=i)'}},
    }
    sources = {'rhs': pl.DataFrame({'i': [0, 1], 'value': [6.0, 9.0]})}
    with lps.build(model, sources) as ex:
        matrix = ex._tables().matrix
        assert matrix.height == 2, 'one entry per row, not one per fragment'
        assert sorted(matrix['coeff'].to_list()) == [3.0, 3.0]
        result = ex.solve()
    assert result.objective == pytest.approx(5.0), '6/3 + 9/3'


def test_a_masked_variable_is_labelled_in_declaration_order():
    """A label is the solver's own column index, so its order is the contract.

    Labels are dense ``0..n-1`` over the coordinates the mask leaves, assigned
    row-major on the dims' declared ordinals. An off-by-one here is a different
    model rather than a slower one, so it is asserted against the order spelled
    out by hand rather than against whatever the engine produced.
    """
    model = {
        'dimensions': {
            'snapshot': {'dtype': 'int', 'values': list(range(5))},
            'node': {'dtype': 'str', 'values': ['a', 'b', 'c']},
            'tech': {'dtype': 'str', 'values': ['wind', 'gas', 'coal', 'hydro']},
        },
        'parameters': {'cap': {'dims': ['node', 'tech']}, 'load': {'dims': ['snapshot']}},
        'variables': {
            'p': {'foreach': ['snapshot', 'node', 'tech'], 'where': 'cap > 0', 'bounds': {'lower': 0, 'upper': 'cap'}}
        },
        'constraints': {
            'balance': {
                'foreach': ['snapshot', 'node'],
                'expression': 'sum(p, over=tech) >= load',
            }
        },
        'objectives': {
            'o': {
                'sense': 'minimize',
                'expression': 'sum(sum(sum(p, over=tech), over=node), over=snapshot)',
            }
        },
    }
    caps = [
        {'node': n, 'tech': t, 'value': 0.0 if (n, t) in {('a', 'gas'), ('c', 'coal'), ('b', 'hydro')} else 10.0}
        for n in ['a', 'b', 'c']
        for t in ['wind', 'gas', 'coal', 'hydro']
    ]
    sources = {
        'cap': pl.DataFrame(caps),
        'load': pl.DataFrame({'snapshot': list(range(5)), 'value': [1.0] * 5}),
    }

    zero = {('a', 'gas'), ('c', 'coal'), ('b', 'hydro')}
    expected = [
        (s, n, t)
        for s in range(5)
        for n in ['a', 'b', 'c']
        for t in ['wind', 'gas', 'coal', 'hydro']
        if (n, t) not in zero
    ]

    with lps.build(model, sources) as ex:
        labelled = ex._variables['p'].collect()

    assert labelled['var_label'].to_list() == list(range(len(expected))), 'labels must be dense and ascending'
    assert list(labelled.select('snapshot', 'node', 'tech').iter_rows()) == expected


def test_a_dictionary_encoded_source_column_binds_like_a_plain_one():
    """A `Categorical` dim column is a source encoding, not a different model.

    Any writer that sees a 12M-row table of repeated node names will
    dictionary-encode it, and pandas does it by default for a `Categorical`.
    Each writer's dictionary is its own, and polars will not join dictionaries
    that disagree — so binding reads a source out of whatever encoding it
    arrived in and re-encodes every dim column into the one dictionary the
    dimension declares. Without that, the failure is a schema error from
    inside a join rather than anything a caller can act on.
    """
    model = {
        'dimensions': {'node': {'dtype': 'str', 'values': ['a', 'b']}},
        'parameters': {'cap': {'dims': ['node']}},
        'variables': {'x': {'foreach': ['node'], 'bounds': {'lower': 0, 'upper': 'cap'}}},
        'constraints': {'c': {'foreach': ['node'], 'expression': 'x >= cap'}},
        'objectives': {'o': {'sense': 'minimize', 'expression': 'sum(x, over=node)'}},
    }
    encoded = pl.DataFrame({'node': ['a', 'b'], 'value': [3.0, 4.0]}).with_columns(pl.col('node').cast(pl.Categorical))
    plain = pl.DataFrame({'node': ['a', 'b'], 'value': [3.0, 4.0]})

    with lps.build(model, {'cap': encoded}) as ex:
        from_encoded = ex.solve().objective
    with lps.build(model, {'cap': plain}) as ex:
        from_plain = ex.solve().objective

    assert from_encoded == pytest.approx(7.0)
    assert from_encoded == pytest.approx(from_plain), 'the encoding changed the model'


def test_a_string_dimension_leaves_the_read_back_as_string():
    """A string dim is plain ``String`` the moment it is handed back.

    What a caller gets is something they join against their own data, which an
    ``Enum`` refuses. Declaration order survives because it was never the dtype
    carrying it: it is the row order, and `to_pandas` puts the ordered
    categories back for the bridge out.

    **Asked of whichever engine is running.** How a dim is encoded *inside* a
    build is that engine's business — polars holds an `Enum` and casts on the
    way out, duckdb holds ``VARCHAR`` throughout — but what leaves is a
    contract, and `test_engine_parity` compares built models rather than
    returned dtypes, so nothing else here would catch the two disagreeing.
    """
    model = {
        'dimensions': {'node': {'dtype': 'str', 'values': ['c', 'a', 'b']}},
        'parameters': {'cap': {'dims': ['node']}},
        'variables': {'x': {'foreach': ['node'], 'bounds': {'lower': 0, 'upper': 'cap'}}},
        'constraints': {'k': {'foreach': ['node'], 'expression': 'x >= cap'}},
        'objectives': {'o': {'sense': 'minimize', 'expression': 'sum(x, over=node)'}},
    }
    cap = pl.DataFrame({'node': ['a', 'b', 'c'], 'value': [1.0, 2.0, 3.0]})

    with lps.build(model, {'cap': cap}) as ex:
        primal = ex.solve().primal('x')

    assert primal.schema['node'] == pl.String, 'what leaves is what a caller can join against'
    assert primal['node'].to_list() == ['c', 'a', 'b'], 'read-back follows label order, not source order'
    assert primal.join(cap, on='node').height == 3, "the caller's own frame is String, and it joins"


def test_a_string_dimension_is_enum_encoded_inside_the_build(engine_internals):
    """The encoding buys the joins and the label frames, and every gram of that
    is internal — so it is asserted where it is done, not where it is undone."""
    model = {
        'dimensions': {'node': {'dtype': 'str', 'values': ['c', 'a', 'b']}},
        'parameters': {'cap': {'dims': ['node']}},
        'variables': {'x': {'foreach': ['node'], 'bounds': {'lower': 0, 'upper': 'cap'}}},
        'constraints': {'k': {'foreach': ['node'], 'expression': 'x >= cap'}},
        'objectives': {'o': {'sense': 'minimize', 'expression': 'sum(x, over=node)'}},
    }
    cap = pl.DataFrame({'node': ['a', 'b', 'c'], 'value': [1.0, 2.0, 3.0]})

    with lps.build(model, {'cap': cap}) as ex:
        assert ex._variables['x'].collect_schema()['node'] == pl.Enum(['c', 'a', 'b'])


def test_a_where_orders_string_labels_bytewise_not_by_declaration():
    """`node >= 'b'` keeps {b, c} whatever order the labels were declared in
    (§6.1). Declared c, a, b so the Enum's declaration order would keep {b} alone."""
    model = {
        'dimensions': {'node': {'dtype': 'str', 'values': ['c', 'a', 'b']}},
        'parameters': {'cap': {'dims': ['node']}},
        'variables': {'x': {'foreach': ['node'], 'where': "node >= 'b'", 'bounds': {'lower': 0, 'upper': 'cap'}}},
        'constraints': {'k': {'foreach': ['node'], 'where': "node >= 'b'", 'expression': 'x >= cap'}},
        'objectives': {'o': {'sense': 'minimize', 'expression': 'sum(x, over=node)'}},
    }
    cap = pl.DataFrame({'node': ['a', 'b', 'c'], 'value': [1.0, 2.0, 3.0]})

    with lps.build(model, {'cap': cap}) as ex:
        assert sorted(ex._variables['x'].collect()['node'].to_list()) == ['b', 'c']
        assert ex.solve().objective == pytest.approx(5.0)


def test_a_where_naming_an_undeclared_label_masks_nothing_in():
    """A quoted label the dimension does not carry masks everything out (§6.1)
    — the Enum would refuse the stranger, so the comparison is in String space."""
    model = {
        'dimensions': {'node': {'dtype': 'str', 'values': ['a', 'b']}},
        'parameters': {'cap': {'dims': ['node']}},
        'variables': {'x': {'foreach': ['node'], 'bounds': {'lower': 0, 'upper': 'cap'}}},
        'constraints': {'k': {'foreach': ['node'], 'where': "node == 'zzz'", 'expression': 'x >= cap'}},
        'objectives': {'o': {'sense': 'minimize', 'expression': 'sum(x, over=node)'}},
    }
    cap = pl.DataFrame({'node': ['a', 'b'], 'value': [1.0, 2.0]})

    with lps.build(model, {'cap': cap}) as ex:
        assert ex._tables().rows.height == 0, 'the mask holds nowhere, so no constraint row is built'
        assert ex.solve().objective == pytest.approx(0.0)


def test_an_objective_naming_a_variable_twice_sums_its_coefficients():
    """Same argument, one dimension down: the objective is a column vector."""
    model = {
        'dimensions': {'i': {'dtype': 'int', 'values': [0]}},
        'parameters': {'lb': {'dims': ['i']}},
        'variables': {'x': {'foreach': ['i'], 'bounds': {'lower': 'lb'}}},
        'constraints': {'c': {'foreach': ['i'], 'expression': 'x >= lb'}},
        'objectives': {'o': {'sense': 'minimize', 'expression': 'x + 4 * x'}},
    }
    with lps.build(model, {'lb': pl.DataFrame({'i': [0], 'value': [2.0]})}) as ex:
        assert ex._tables().obj.height == 1
        assert ex._tables().obj['coeff'].to_list() == [5.0]
        assert ex.solve().objective == pytest.approx(10.0)


def test_a_mask_that_removes_nothing_labels_exactly_like_no_mask(dispatch_data):
    """A vacuous `where` must not shift a single solver index.

    Labels are the solver's own column numbers, so "the mask removed nothing"
    and "there was no mask" have to produce identical frames rather than merely
    the same row count.
    """
    gens, load = dispatch_data
    gens = gens.assign(p_max=gens['p_max'].where(gens['p_max'] > 0, 1.0))  # nothing left to mask out

    labels = []
    for where in (None, ParameterComparison('p_max', '>', 0)):
        base = dispatch_program()
        program = replace(base, variables=(replace(base.variables[0], where=where),))
        with PolarsExecutor() as ex:
            ex.build(program, dispatch_sources(gens, load))
            labels.append(ex._variables['p'].collect().sort('var_label'))
    assert labels[0].equals(labels[1])


def _objective_of(program, sources):
    """`obj` as `{col: coeff}`, plus whether the aggregate was skipped."""
    with PolarsExecutor() as ex:
        ex.build(program, sources)
        obj = ex._tables().obj
        return dict(zip(obj['col'].to_list(), obj['coeff'].to_list(), strict=True)), obj.height


def test_a_mask_a_missing_value_can_satisfy_keeps_the_rows_with_no_value():
    """`not` and `or` select rows the mask's own join must not have dropped.

    A mask's parameters are joined for, and under a plain conjunction that join
    can be an inner one: a coordinate the parameter has no row for fails the
    conjunct anyway, so keeping it only widens the frame the filter then
    narrows. Under `not` or `or` a missing value is what makes the mask *true*,
    and an inner join would have thrown the row away before the filter could
    say so.

    Every mask in the suite was a conjunction before this, so nothing else here
    distinguishes the two joins.

    """
    model = {
        'dimensions': {'i': {'dtype': 'int', 'values': [0, 1, 2, 3]}},
        'parameters': {'a': {'dims': ['i']}, 'b': {'dims': ['i']}},
        'variables': {
            'absent': {'foreach': ['i'], 'where': 'not a', 'bounds': {'lower': 0, 'upper': 1}},
            'either': {'foreach': ['i'], 'where': 'a > 0 or b > 0', 'bounds': {'lower': 0, 'upper': 1}},
            'both': {'foreach': ['i'], 'where': 'a and a > 0', 'bounds': {'lower': 0, 'upper': 1}},
            'mixed': {'foreach': ['i'], 'where': 'a and not b', 'bounds': {'lower': 0, 'upper': 1}},
        },
        'objectives': {'o': {'sense': 'minimize', 'expression': 'sum(absent, over=i)'}},
    }
    sources = {
        'a': pl.DataFrame({'i': [0, 1], 'value': [5.0, -1.0]}),
        'b': pl.DataFrame({'i': [2], 'value': [7.0]}),
    }
    with lps.build(model, sources) as ex:
        surviving = {
            name: sorted(ex._variables[name].select('i').collect().to_series().to_list())
            for name in ('absent', 'either', 'both', 'mixed')
        }
    assert surviving == {
        'absent': [2, 3],  # `a` is missing there, which is the whole condition
        'either': [0, 2],  # i=2 has no `a` at all and qualifies on `b`
        'both': [0],
        'mixed': [0, 1],  # `a` is certain, `b` is not
    }


def test_every_declaration_owns_a_contiguous_run_of_labels():
    """What reading a solve back by position rests on.

    A solver vector is positional in the same index the labels are, so a
    declaration's share of it is a slice — but only if its labels are a dense
    run and the runs tile the whole index. Both are how `labels.frame` numbers
    a declaration, and neither is visible from the frames: a block
    that started one late would report a neighbour's numbers under this
    declaration's coordinates, with nothing out of range and nothing null.
    """
    model = {
        'dimensions': {'i': {'dtype': 'int', 'values': [0, 1, 2]}, 'j': {'values': ['a', 'b']}},
        'parameters': {'cap': {'dims': ['i']}},
        'variables': {
            'x': {'foreach': ['i'], 'bounds': {'lower': 0, 'upper': 'cap'}},
            'y': {'foreach': ['i', 'j'], 'bounds': {'lower': 0}},
            'z': {'foreach': ['j'], 'bounds': {'lower': 0}},
        },
        'constraints': {
            'c1': {'foreach': ['i'], 'expression': 'x >= cap'},
            'c2': {'foreach': ['i', 'j'], 'expression': 'y >= 0'},
        },
        'objectives': {'o': {'sense': 'minimize', 'expression': 'sum(x, over=i)'}},
    }
    with lps.build(model, {'cap': pl.DataFrame({'i': [0, 1, 2], 'value': [1.0, 2.0, 3.0]})}) as ex:
        tables = ex._tables()
        for names, total, frames, label in (
            (['x', 'y', 'z'], tables.column_count, ex._variables, 'var_label'),
            (['c1', 'c2'], tables.row_count, ex._constraints, 'row'),
        ):
            at = 0
            for name in names:
                start, height = ex._blocks[name]
                assert start == at, f'{name} does not start where the previous declaration ended'
                labels = frames[name].select(label).collect().to_series()
                assert sorted(labels) == list(range(start, start + height)), f'{name} is not a dense run'
                at += height
            assert at == total, 'the runs do not tile the index'


def test_the_matrix_collapses_a_repeated_cell_and_leaves_the_rest_alone():
    """Both outcomes of the terminal aggregate, on models differing only in overlap.

    Two fragments over disjoint variables repeat nothing and must come out
    untouched; two over the same variable land on one cell and must be summed.
    A matrix holding the same `(row, col)` twice is not a slower model, it is
    one the sinks disagree about.
    """
    base = {
        'dimensions': {'i': {'dtype': 'int', 'values': [0, 1]}},
        'parameters': {'rhs': {'dims': ['i']}},
        'variables': {
            'x': {'foreach': ['i'], 'bounds': {'lower': 0}},
            'y': {'foreach': ['i'], 'bounds': {'lower': 0}},
        },
        'objectives': {'o': {'sense': 'minimize', 'expression': 'sum(x, over=i) + sum(y, over=i)'}},
    }
    sources = {'rhs': pl.DataFrame({'i': [0, 1], 'value': [4.0, 6.0]})}

    disjoint = dict(base, constraints={'c': {'foreach': ['i'], 'expression': 'x + y >= rhs'}})
    with lps.build(disjoint, sources) as ex:
        matrix = ex._tables().matrix
        assert matrix.height == 4, 'two variables per row, nothing to collapse'
        assert matrix['coeff'].to_list() == [1.0, 1.0, 1.0, 1.0]

    overlapping = dict(base, constraints={'c': {'foreach': ['i'], 'expression': 'x + 3 * x >= rhs'}})
    with lps.build(overlapping, sources) as ex:
        matrix = ex._tables().matrix
        assert matrix.height == 2, 'one cell per row after the collapse'
        assert matrix['coeff'].to_list() == [4.0, 4.0]


def _network(self_loop: bool) -> tuple[dict, dict]:
    """A balance of flows in minus flows out, with or without a line to itself."""
    model = {
        'dimensions': {
            'snapshot': {'dtype': 'int', 'values': [0, 1]},
            'bus': {'values': ['b0', 'b1']},
            'line': {'coords': {'from': 'bus', 'to': 'bus'}},
        },
        'parameters': {'cap': {'dims': ['line']}, 'load': {'dims': ['snapshot', 'bus']}},
        'variables': {'f': {'foreach': ['snapshot', 'line'], 'bounds': {'lower': 0, 'upper': 'cap'}}},
        'constraints': {
            'balance': {
                'foreach': ['snapshot', 'bus'],
                'expression': 'sum(f, over=line, group_by=to) - sum(f, over=line, group_by=from) == load',
            }
        },
        'objectives': {'o': {'sense': 'minimize', 'expression': 'sum(sum(f, over=line), over=snapshot)'}},
    }
    ends = ('b0', 'b1') if not self_loop else ('b0', 'b0')
    sources = {
        'line': pl.DataFrame({'line': ['l0', 'l1'], 'from': ['b0', ends[0]], 'to': ['b1', ends[1]]}),
        'cap': pl.DataFrame({'line': ['l0', 'l1'], 'value': [10.0, 10.0]}),
        'load': pl.DataFrame(
            {'snapshot': [0, 0, 1, 1], 'bus': ['b0', 'b1', 'b0', 'b1'], 'value': [0.0, 0.0, 0.0, 0.0]}
        ),
    }
    return model, sources


def test_two_sums_of_one_variable_collide_only_where_the_coordinates_meet():
    """`group_by=to` and `group_by=from` reach one cell exactly on a line to itself.

    Both fragments carry `f`, so counting variables says the aggregate is
    reachable and every nonzero in the model gets sorted to find out. Which
    labels they share is decided by the *line* table — two rows here, forty at
    the `l` rung, against 12.6M nonzeros — so it is asked there.

    The self-loop is the case the collapse exists for: `l1` leaves and arrives
    at `b0`, so `+f - f` lands twice on one cell. A matrix holding the same
    `(row, col)` twice is not a slower model, it is one the sinks disagree
    about — an LP reader sums the pair and a solver handed duplicate entries is
    entitled to do either.
    """
    for self_loop in (False, True):
        model, sources = _network(self_loop)
        with lps.build(model, sources) as ex:
            program = lower_program(Model(**model))
            terms = ex._q.expression(program.constraints[0].lhs, 'test').terms
            assert len(terms) == 2

            tables = ex._tables()
            cells = tables.matrix_block(0, tables.row_count).select('row', 'col')
            assert cells.height == cells.unique().height, f'a cell reached the sinks twice (self_loop={self_loop})'


def test_the_objective_sums_the_coefficients_that_land_on_one_column():
    """`p * cost` is one row per column; `p * cost + p * cost` is two, summed."""
    base = {
        'dimensions': {'i': {'dtype': 'int', 'values': [0, 1]}},
        'parameters': {'cost': {'dims': ['i']}, 'lb': {'dims': ['i']}},
        'variables': {'p': {'foreach': ['i'], 'bounds': {'lower': 'lb'}}},
        'constraints': {'c': {'foreach': ['i'], 'expression': 'p >= lb'}},
    }
    sources = {
        'cost': pl.DataFrame({'i': [0, 1], 'value': [2.0, 3.0]}),
        'lb': pl.DataFrame({'i': [0, 1], 'value': [1.0, 1.0]}),
    }
    once = lower_program(Model(**dict(base, objectives={'o': {'sense': 'minimize', 'expression': 'p * cost'}})))
    twice = lower_program(
        Model(**dict(base, objectives={'o': {'sense': 'minimize', 'expression': 'p * cost + p * cost'}}))
    )

    assert _objective_of(once, sources) == ({0: 2.0, 1: 3.0}, 2)
    assert _objective_of(twice, sources) == ({0: 4.0, 1: 6.0}, 2)


def test_the_objective_aggregate_survives_a_reduction_that_hides_extra_rows():
    """A fragment's dims can match the variable's while its rows do not.

    `sum(q * price, over=generator)` with `q` indexed by snapshot alone reduces
    to dims `('snapshot',)` — exactly `q`'s declaration — but `_sum_fragment`
    *projects*, so the fragment still carries one row per generator. Those rows
    all name one column, and `obj` must hold their sum: the LP file would
    quietly re-sum |generator| rows, while `cols` joined to `obj` in the HiGHS
    sink would hand the solver more columns than the model has.

    """
    model = {
        'dimensions': {'snapshot': {'dtype': 'int', 'values': [0, 1]}, 'generator': {'values': ['g0', 'g1', 'g2']}},
        'parameters': {'price': {'dims': ['snapshot', 'generator']}, 'load': {'dims': ['snapshot']}},
        'variables': {'q': {'foreach': ['snapshot'], 'bounds': {'lower': 0, 'upper': 10}}},
        'constraints': {'floor': {'foreach': ['snapshot'], 'expression': 'q >= load'}},
        'objectives': {'o': {'sense': 'minimize', 'expression': 'sum(q * price, over=generator)'}},
    }
    sources = {
        'price': pl.DataFrame(
            {'snapshot': [0, 0, 0, 1, 1, 1], 'generator': ['g0', 'g1', 'g2'] * 2, 'value': [1.0, 2.0, 3.0] * 2}
        ),
        'load': pl.DataFrame({'snapshot': [0, 1], 'value': [5.0, 5.0]}),
    }
    assert _objective_of(lower_program(Model(**model)), sources) == ({0: 6.0, 1: 6.0}, 2), (
        'one row per column, each carrying the summed price — not three rows of one'
    )


def test_infinite_bounds_survive_the_handoff(dispatch_data):
    """An absent upper bound must reach HiGHS as infinity, not as a number."""
    gens, load = dispatch_data
    base = dispatch_program()
    unbounded = replace(base, variables=(replace(base.variables[0], upper=Constant(float('inf'))),))
    with PolarsExecutor() as ex:
        ex.build(unbounded, dispatch_sources(gens, load))
        assert ex._tables().cols['ub'].is_infinite().all()
        assert ex.solve().is_ok


def test_a_solution_is_read_back_in_label_order_without_sorting_for_it():
    """The order is produced by the labeller, so the read-back only reads it.

    Seeding the vector with an ``arange`` makes every value its own label, so a
    read-back in label order comes out ascending — which is the contract
    `sol.primal` states, asserted directly rather than through the sort that
    used to establish it.

    The plan assertion is the other half: re-imposing the order is not wrong,
    it moved a full copy of the coordinates at the moment the solver's own
    model is still resident.
    """
    model = {
        'dimensions': {'t': {'dtype': 'int', 'values': [0, 1, 2, 3]}, 'g': {'values': ['a', 'b', 'c']}},
        'parameters': {'cap': {'dims': ['g']}, 'load': {'dims': ['t']}},
        'variables': {'p': {'foreach': ['t', 'g'], 'where': 'cap > 0', 'bounds': {'lower': 0, 'upper': 'cap'}}},
        'constraints': {'meet': {'foreach': ['t'], 'where': 'load > 0', 'expression': 'sum(p, over=g) >= load'}},
        'objectives': {'o': {'sense': 'minimize', 'expression': 'sum(sum(p, over=g), over=t)'}},
    }
    sources = {
        'cap': pl.DataFrame({'g': ['a', 'b', 'c'], 'value': [5.0, 0.0, 7.0]}),
        'load': pl.DataFrame({'t': [0, 1, 2, 3], 'value': [1.0, 0.0, 2.0, 3.0]}),
    }
    with lps.build(model, sources) as ex:
        primal = pl.Series('value', np.arange(ex._n_cols, dtype=np.float64))
        dual = pl.Series('value', np.arange(ex._n_rows, dtype=np.float64))
        variable = ex._solution_frame('p', primal)
        assert 'SORT' not in variable.explain(optimized=False), 'the labeller already ordered this'
        assert variable.collect()['value'].to_list() == list(range(len(primal))), 'primal not in label order'
        assert ex._dual('meet', dual)['value'].to_list() == list(range(len(dual))), 'dual not in label order'


#: Three columns and three rows, the smallest model whose solution vector has a
#: length worth disagreeing about.
SOLVER_VECTOR_MODEL = {
    'dimensions': {'t': {'dtype': 'int', 'values': [0, 1, 2]}},
    'parameters': {'load': {'dims': ['t']}},
    'variables': {'x': {'foreach': ['t'], 'bounds': {'lower': 0, 'upper': 10}}},
    'constraints': {'meet': {'foreach': ['t'], 'expression': 'x >= load'}},
    'objectives': {'o': {'sense': 'minimize', 'expression': 'sum(x, over=t)'}},
}
SOLVER_VECTOR_LOAD = {'load': pl.DataFrame({'t': [0, 1, 2], 'value': [1.0, 2.0, 3.0]})}


@pytest.mark.parametrize('length', [2, 5], ids=['short', 'long'])
def test_a_solver_vector_that_does_not_span_the_model_is_refused(monkeypatch, length):
    """A wrong length is a different model's answer, not a short one.

    Reading a solution back is positional. A short vector leaves the trailing
    declarations reading past the end; a long one leaves every declaration
    reading the right slice of the wrong vector. Neither is recoverable, so
    both are refused where the solver hands them over — not where they are
    read: the objective comes straight from the solver, so a `Result` built on
    a broken vector reports a plausible number and fails only if someone asks
    for a coordinate.
    """
    from lpspec.relational.engines.polars import executor as executor_module

    honest = executor_module.sinks.solver('highs')

    def crooked(tables, batch_rows, options):
        status, objective, primal, dual = honest(tables, batch_rows, options)
        stretched = pl.Series('value', list(primal) + [0.0] * length)
        return status, objective, stretched.head(length), dual

    monkeypatch.setattr(executor_module.sinks, 'solver', lambda _name: crooked)
    with (
        lps.build(SOLVER_VECTOR_MODEL, SOLVER_VECTOR_LOAD) as ex,
        pytest.raises(LpspecError, match=f'returned {length} primal values for a model with 3'),
    ):
        ex.solve()


@pytest.mark.parametrize('solver_name', sorted(SOLVERS))
def test_a_solver_hands_back_a_vector_and_not_an_index(solver_name):
    """A solution is positional, so there is nothing to key it by.

    Solver output is indexed by the solver's own index, which *is* our label —
    so a ``(label, value)`` frame carries an ``arange`` beside every value that
    the read-back never reads, 8 bytes a column for as long as the result is
    held. The same argument took ``col`` off ``cols`` in #433; this is the
    other half of it, and neither is visible from the numbers.
    """
    with lps.build(SOLVER_VECTOR_MODEL, SOLVER_VECTOR_LOAD) as ex:
        tables = ex._tables()
        solution = ex.solve(solver_name=solver_name)
        assert solution.is_ok
        for values, count in (
            (solution._primal_values, tables.column_count),
            (solution._dual_values, tables.row_count),
        ):
            assert isinstance(values, pl.Series), 'a frame here is an index column nothing reads'
            assert values.name == 'value'
            assert len(values) == count, 'the read-back slices it positionally, so it spans the model'


@pytest.mark.parametrize('solver_name', sorted(SOLVERS))
@pytest.mark.parametrize('batch_rows', [1, 2, 7, 100_000], ids=['one', 'two', 'odd', 'whole'])
def test_a_row_with_no_terms_is_not_built_and_is_reported(solver_name, batch_rows):
    """A row that lost every term is not a constraint, and the build says so.

    `where: "t > 0"` leaves `balance` at `t = 0` with nothing to sum. Three
    provenances reach that shape — an absent variable, an empty reduction, a
    missing coefficient — and the language used to answer them differently, so
    the same empty row meant different things depending on how it emptied. The
    rule is now at the level the property lives at: no variable terms, no row.

    **The omission is reported, and that is what makes dropping defensible.**
    An unenforced constraint the caller cannot see is the failure this used to
    guard against by keeping the row; `omissions()` answers it without asking
    the solver to carry a comparison nothing can fail.

    Ragged batches because the range loop is where a *surviving* seat would be
    lost — labels are compacted when a row goes, so the dense vector and the
    chunk ranges have to agree about the narrower block. Both solvers, because
    the seating is theirs jointly.
    """
    model = {
        'dimensions': {'t': {'dtype': 'int', 'values': [0, 1, 2]}, 'g': {'values': ['a', 'b']}},
        'parameters': {'load': {'dims': ['t']}},
        'variables': {'p': {'foreach': ['t', 'g'], 'where': 't > 0', 'bounds': {'lower': 0, 'upper': 100}}},
        'constraints': {'balance': {'foreach': ['t'], 'expression': 'sum(p, over=g) == load'}},
        'objectives': {'o': {'sense': 'minimize', 'expression': 'sum(sum(p, over=g), over=t)'}},
    }
    with lps.build(model, {'load': pl.DataFrame({'t': [0, 1, 2], 'value': [5.0, 4.0, 6.0]})}) as ex:
        tables = ex._tables()
        occupied = sorted(set(tables.matrix_block(0, tables.row_count)['row'].to_list()))
        assert occupied == [0, 1], 'the block closes up around the gap'
        assert ex.omissions().to_dicts() == [{'constraint': 'balance', 'rows_not_built': 1}]
        solution = ex.solve(batch_rows=batch_rows, solver_name=solver_name)
        assert solution.termination_condition == 'optimal'
        assert solution.objective == pytest.approx(4.0 + 6.0, rel=RTOL), 'the two built rows still bind'


def test_omissions_is_empty_when_every_declared_row_is_built():
    """The common case says nothing, so the report is a signal rather than noise."""
    model = {
        'dimensions': {'t': {'dtype': 'int', 'values': [0, 1]}},
        'parameters': {'load': {'dims': ['t']}},
        'variables': {'x': {'foreach': ['t'], 'bounds': {'lower': 0, 'upper': 10}}},
        'constraints': {'meet': {'foreach': ['t'], 'expression': 'x >= load'}},
        'objectives': {'o': {'sense': 'minimize', 'expression': 'sum(x, over=t)'}},
    }
    with lps.build(model, {'load': pl.DataFrame({'t': [0, 1], 'value': [1.0, 2.0]})}) as ex:
        assert ex.omissions().is_empty()


def test_row_chunks_are_bounded_by_nonzeros_not_by_rows():
    """A chunk of rows is a chunk of *entries*, and only entries are residency.

    The solver hand-off reads ``matrix`` a range at a time and holds that range
    while it works it. Sizing the range in rows bounds the wrong quantity: the
    same 100k-row range is 900k entries in ``transport`` and 10M in
    ``dispatch``, so what the sink actually holds is set by the model's shape
    rather than by the budget — which defeats the point of batching at all: a
    pass that holds a slice proportional to the model is a pass that holds the
    model.

    Wide rows are the case that separates the two, so this builds them: 50
    generators summed into each of 4 snapshots is 50 entries per row.

    """
    n_g, n_s = 50, 4
    gens = pd.DataFrame(
        {
            'generator': [f'g{i}' for i in range(n_g)],
            'p_max': np.full(n_g, 10.0),
            'cost': np.arange(1.0, n_g + 1.0),
        }
    )
    load = pd.DataFrame({'snapshot': np.arange(n_s), 'value': np.full(n_s, 100.0)})

    with PolarsExecutor() as ex:
        ex.build(dispatch_program(), dispatch_sources(gens, load))
        tables = ex._tables()
        assert tables.matrix.height == n_g * n_s

        def widest(ranges):
            return max(int(tables.row_starts[hi] - tables.row_starts[lo]) for lo, hi in ranges)

        budget = 100
        assert widest(tables._spans(budget)) <= budget

        assert widest(chunking.ranges(tables.row_count, budget, 1.0)) == n_g * n_s, (
            'the same budget spent as if a row cost one element puts every entry in one chunk'
        )


@pytest.mark.parametrize(
    ('total', 'budget', 'width'),
    [(0, 100, 1.0), (1, 100, 1.0), (10, 3, 1.0), (10, 100, 1.0), (10, 3, 7.0), (10, 3, 0.25), (10, 1, 1e9)],
)
def test_chunk_ranges_are_contiguous_gapless_and_cover_everything(total, budget, width):
    """The property the whole hand-off rests on, at every awkward size.

    ``addCols`` and ``addRows`` append: column *k* must be the *k*-th row handed
    over. That holds only if the ranges are ordered, consecutive and gapless —
    a dropped range silently shortens the model, an overlapping one relabels
    it, and neither shows up as an error. The widths here include one below 1
    and one far above the budget, the two ends where a ``//`` can produce a
    zero step or a chunk wider than asked for.
    """
    got = list(chunking.ranges(total, budget, width))

    assert all(lo < hi for lo, hi in got), 'an empty range means a wasted pass'
    assert [lo for lo, _ in got] == sorted(lo for lo, _ in got), 'ranges must ascend'
    assert [i for lo, hi in got for i in range(lo, hi)] == list(range(total)), 'gap, overlap, or short'
    if total:
        widest = max(hi - lo for lo, hi in got)
        assert widest * max(1.0, width) <= max(budget, max(1.0, width)), 'a chunk exceeded the budget'


SPARSE_COEFFICIENT_MODEL = {
    'dimensions': {'t': {'dtype': 'int', 'values': [0, 1, 2]}},
    'parameters': {'c': {'dims': ['t']}, 'w': {'dims': ['t']}},
    'variables': {'x': {'foreach': ['t'], 'bounds': {'lower': 0, 'upper': 10}}},
    'constraints': {'cap': {'foreach': ['t'], 'expression': 'w * x <= c'}},
    'objectives': {'o': {'sense': 'maximize', 'expression': 'sum(x, over=t)'}},
}


def test_a_sparse_coefficient_is_still_a_zero_coefficient():
    """A tidy parameter table is a compressed dense array — where it can be.

    Supplying rows only where a *coefficient* is nonzero stays the language's
    sparsity idiom (SPEC §8). The uncovered coordinate contributes no term, the
    row survives, and both lanes agree about it: nothing was invented, the term
    simply is not there.

    Only the *constant* side lost this reading, and the test below says why.

    """
    data = {'w': pd.Series({1: 1.0, 2: 1.0}), 'c': pd.Series({0: 0.0, 1: 4.0, 2: 5.0})}
    with differential(SPARSE_COEFFICIENT_MODEL, data, lp=True) as run:
        assert run.result.objective == pytest.approx(10.0 + 4.0 + 5.0, rel=RTOL), (
            't=0 carries `<= 0` with no term: a row that exists and constrains nothing'
        )


def test_a_sparse_constant_side_is_refused_on_both_lanes():
    """The same omission on the constant side is a `DataError`, not a zero.

    There the fill *is* the bound — `w * x <= c` with no `c` row reads `<= 0`,
    which binds rather than vanishing, and the solve reports optimal. Nothing in
    the model said so: a table left sparse is compression, not a claim.

    Refused on both lanes, in the same words, because a rule the eager lane did
    not share would be a parity break rather than a language rule (hard rule 3).
    """
    data = {'w': pd.Series({0: 1.0, 1: 1.0, 2: 1.0}), 'c': pd.Series({1: 4.0, 2: 5.0})}
    with pytest.raises(DataError, match="parameter 'c' covers 1 fewer"), differential(SPARSE_COEFFICIENT_MODEL, data):
        pass


def test_a_where_is_the_escape_from_the_constant_side_check():
    """Masking the coordinate answers the question, so it is not refused.

    The check is keyed to the rows a declaration builds, not to the coordinate
    product — the same property that keeps #312's divisor check from becoming a
    wall. Without it the remedy the error names would not work.

    """
    masked = {**SPARSE_COEFFICIENT_MODEL}
    masked['constraints'] = {'cap': {**SPARSE_COEFFICIENT_MODEL['constraints']['cap'], 'where': 'c'}}
    data = {'w': pd.Series({0: 1.0, 1: 1.0, 2: 1.0}), 'c': pd.Series({1: 4.0, 2: 5.0})}
    with differential(masked, data) as run:
        assert run.result.objective == pytest.approx(10.0 + 4.0 + 5.0, rel=RTOL), (
            't=0 has no row at all, so x runs to its bound there'
        )


ABSENT_VARIABLE_MODEL = {
    'dimensions': {'f': {'values': ['a', 'b']}},
    'parameters': {'gate': {'dims': ['f'], 'dtype': 'bool'}, 'relmax': {'dims': ['f']}, 'cost': {'dims': ['f']}},
    'variables': {
        'x': {'foreach': ['f'], 'bounds': {'lower': 0, 'upper': 100}},
        'size': {'foreach': ['f'], 'where': 'gate', 'bounds': {'lower': 0, 'upper': 50}},
    },
    'constraints': {'envelope': {'foreach': ['f'], 'expression': 'x - relmax * size <= 0'}},
    'objectives': {'total': {'sense': 'maximize', 'expression': 'sum(x * cost, over=f)'}},
}


def test_a_term_whose_variable_is_absent_drops_the_row_on_both_lanes():
    """Absence propagates into the comparison; it does not zero the term.

    ``x - relmax * size <= 0`` where ``size`` is masked out used to build
    ``x <= 0`` — a row that silently pinned the flow to zero. Plausible answer,
    no error, which is goal 1 of linopy's v1 convention ("no silent wrong
    answers") and the whole of PyPSA/linopy#712. Under §6 the slot is absent and
    §12 drops the row instead, so ``x`` is left free at ``f=b`` and bounded only
    by its own declaration.

    The oracle is the point: the eager lane gets this from linopy's own v1
    semantics, the relational lane from carrying variable presence apart from
    the term stream. Two independent implementations, one answer.
    """
    data = {
        'gate': pd.Series({'a': True}),
        'relmax': pd.Series({'a': 0.5, 'b': 0.5}),
        'cost': pd.Series({'a': 1.0, 'b': 1.0}),
    }
    with differential(ABSENT_VARIABLE_MODEL, data, lp=True) as run:
        x = by_coord(run.result, 'x', 'f')
        assert x['a'] == pytest.approx(25.0, rel=RTOL), 'sized: x <= 0.5 * size, size <= 50'
        assert x['b'] == pytest.approx(100.0, rel=RTOL), 'unsized: the row is gone, so only the bound holds'


#: One rule per block, so the two regimes are two named constraints.
DEFINED_MODEL = {
    'dimensions': {'f': {'values': ['a', 'b']}},
    'parameters': {'gate': {'dims': ['f'], 'dtype': 'bool'}, 'relmax': {'dims': ['f']}, 'cost': {'dims': ['f']}},
    'variables': {
        'x': {'foreach': ['f'], 'bounds': {'lower': 0, 'upper': 100}},
        'size': {'foreach': ['f'], 'where': 'gate', 'bounds': {'lower': 0, 'upper': 50}},
    },
    'constraints': {
        'envelope_sized': {'foreach': ['f'], 'where': 'size', 'expression': 'x - relmax * size <= 0'},
        'envelope_unsized': {'foreach': ['f'], 'where': 'NOT size', 'expression': 'x <= 0'},
    },
    'objectives': {'total': {'sense': 'maximize', 'expression': 'sum(x * cost, over=f)'}},
}


def test_a_bare_variable_name_in_a_where_asks_whether_it_exists():
    """The escape hatch for a language where absence drops the row.

    Since a term whose variable is absent takes the row with it, a model that
    wanted the *other* reading — keep the row, treat the term as zero — needs a
    way to say which coordinates those are. A bare parameter name in a ``where``
    already asks "does this have a value here"; a bare variable name asks "does
    this exist here", and the two complementary clauses spell out both cases.

    Without it the only way to write this is a parameter mirroring the
    variable's own mask, which is two sources for one fact and drifts.
    """
    data = {
        'gate': pd.Series({'a': True}),
        'relmax': pd.Series({'a': 0.5, 'b': 0.5}),
        'cost': pd.Series({'a': 1.0, 'b': 1.0}),
    }
    with differential(DEFINED_MODEL, data, lp=True) as run:
        x = by_coord(run.result, 'x', 'f')
        assert x['a'] == pytest.approx(25.0, rel=RTOL), 'sized: the envelope binds'
        assert x['b'] == pytest.approx(0.0, abs=1e-9), 'unsized: the complementary clause pins it'


ABSENT_COEFFICIENT_MODEL = {
    'dimensions': {'f': {'values': ['a', 'b']}},
    'parameters': {'relmax': {'dims': ['f']}, 'cost': {'dims': ['f']}},
    'variables': {
        'x': {'foreach': ['f'], 'bounds': {'lower': 0, 'upper': 100}},
        'size': {'foreach': ['f'], 'bounds': {'lower': 0, 'upper': 50}},
    },
    'constraints': {'envelope': {'foreach': ['f'], 'expression': 'x - relmax * size <= 0'}},
    'objectives': {'total': {'sense': 'maximize', 'expression': 'sum(x * cost, over=f)'}},
}


def test_a_sparse_coefficient_on_the_bound_side_still_pins_the_variable():
    """The half of §6's hazard that survives absence propagation.

    Same expression as ``ABSENT_VARIABLE_MODEL`` above, one operand different:
    the thing missing at ``f=b`` is the *parameter* ``relmax``, not the variable
    ``size``. Absence is a property of variables, so nothing propagates — the
    row is kept, the term is dropped, and ``x <= 0`` is built.

    That is correct and it is the documented reading of a sparse coefficient
    table, but it is the same silently-wrong shape the v1 convention removed
    from the variable side, so SPEC §6 now names it and this pins the behaviour
    the prose describes. The benign case is
    ``test_a_parameter_covering_a_subset_of_its_dims_means_zero_on_both_lanes``:
    there the zero lands on a coefficient *and* a right-hand side, so the row
    constrains nothing. Here the right-hand side is a literal 0 and the missing
    coefficient was the whole bound.
    """
    data = {
        'relmax': pd.Series({'a': 0.5}),  # no row at 'b'
        'cost': pd.Series({'a': 1.0, 'b': 1.0}),
    }
    with differential(ABSENT_COEFFICIENT_MODEL, data, lp=True) as run:
        x = by_coord(run.result, 'x', 'f')
        assert x['a'] == pytest.approx(25.0, rel=RTOL), 'sized: x <= 0.5 * size, size <= 50'
        assert x['b'] == pytest.approx(0.0, abs=1e-9), 'the row survived the missing coefficient and pins x'


def _reindexed_parameter_model(op: str) -> dict:
    return {
        'dimensions': {'t': {'dtype': 'int', 'values': [0, 1, 2]}},
        'parameters': {'dt': {'dims': ['t']}},
        'variables': {'x': {'foreach': ['t'], 'bounds': {'lower': 0, 'upper': 100}}},
        'constraints': {'r': {'foreach': ['t'], 'expression': f'x <= {op}'}},
        'objectives': {'o': {'sense': 'maximize', 'expression': 'sum(x, over=t)'}},
    }


@pytest.mark.parametrize(
    ('op', 'expected'),
    [
        pytest.param(
            "shift(dt, over=t, by=1, edge='wrap')",
            {0: 7.0, 1: 5.0, 2: 6.0},
            id='cyclic-vacates-nothing-so-t0-reads-the-last-value',
        ),
        pytest.param(
            'shift(dt, over=t, by=1, edge=0)',
            {0: 0.0, 1: 5.0, 2: 6.0},
            id='the-vacated-position-contributes-zero-which-pins',
        ),
    ],
)
def test_roll_and_filled_shift_re_index_a_parameter_not_only_a_variable(op, expected):
    """``array`` in §7 is any node, so these operators read a parameter.

    Worth its own test because every example in SPEC took a variable, and a
    downstream consumer built and shipped a hand-shifted copy of a parameter
    table before probing revealed this works.

    ``fill=0`` is what a *bare* ``shift`` used to mean here, and the pin it
    produces at ``t=0`` is why it stopped being the default — see the refusal
    below. Spelled out, it is a legitimate thing to ask for, so it still works.
    """
    data = {'dt': pd.Series({0: 5.0, 1: 6.0, 2: 7.0})}
    with differential(_reindexed_parameter_model(op), data, lp=True) as run:
        x = by_coord(run.result, 'x', 't')
        for t, want in expected.items():
            assert x[t] == pytest.approx(want, abs=1e-9), f'{op} at t={t}'


def test_a_bare_shift_over_data_is_refused_rather_than_filled():
    """The pin, removed at its source (#289).

    ``x <= shift(dt, over=t, by=1)`` used to build ``x <= 0`` at the first coordinate:
    a bound invented from a slot that has no value. Absence would be the
    consistent answer, but a parameter has no absence to propagate — a missing
    row is a zero coefficient (§6) — so this follows linopy v1 and refuses,
    at load time, naming the three things the author might have meant.

    Decidable without data, so ``lps.check()`` catches it: the operand is
    variable-free by declaration, not by what arrives in ``sources``.
    """
    model = _reindexed_parameter_model('shift(dt, over=t, by=1)')
    with pytest.raises(LanguageError) as exc:
        lps.check(model)
    assert 'edge=0' in str(exc.value), 'the refusal must name the escape hatch'
    assert "edge='wrap'" in str(exc.value), 'and the policy for a genuinely cyclic horizon'


PINNED_MODEL = {
    'dimensions': {'f': {'values': ['fixed', 'sized']}},
    'parameters': {'relmax': {'dims': ['f']}, 'size_lb': {'dims': ['f']}, 'size_ub': {'dims': ['f']}},
    'variables': {
        'rate': {'foreach': ['f'], 'bounds': {'lower': 0, 'upper': 1000}},
        'size': {'foreach': ['f'], 'bounds': {'lower': 'size_lb', 'upper': 'size_ub'}},
    },
    'constraints': {'envelope': {'foreach': ['f'], 'expression': 'rate - relmax * size <= 0'}},
    'objectives': {'total': {'sense': 'maximize', 'expression': 'sum(rate, over=f)'}},
}


def test_equal_bounds_pin_a_variable_so_one_equation_covers_both_regimes():
    """A capacity that is data in one model and a decision in another.

    The alternative a consumer reaches for otherwise is a block per regime with
    pre-multiplied coefficients — ``rate_max_at_size``, ``rate_max_when_on`` —
    whose names encode which regime they belong to rather than what quantity
    they are. Pinning with equal bounds writes the row form once and lets
    presolve substitute the fixed column, so SPEC §2 documents it and this shows
    both regimes coming out of the single equation.
    """
    data = {
        'relmax': pd.Series({'fixed': 0.8, 'sized': 0.8}),
        'size_lb': pd.Series({'fixed': 10.0, 'sized': 0.0}),
        'size_ub': pd.Series({'fixed': 10.0, 'sized': 50.0}),
    }
    with differential(PINNED_MODEL, data, lp=True) as run:
        rate = by_coord(run.result, 'rate', 'f')
        assert rate['fixed'] == pytest.approx(8.0, rel=RTOL), 'pinned at 10, so the envelope is 0.8 * 10'
        assert rate['sized'] == pytest.approx(40.0, rel=RTOL), 'free to 50, so the envelope is 0.8 * 50'


SCALAR_MASKED_MODEL = {
    'dimensions': {'f': {'values': ['a', 'b']}},
    'parameters': {'cost': {'dims': ['f']}, 'budget': {'dims': []}},
    'variables': {
        'x': {'foreach': ['f'], 'bounds': {'lower': 0, 'upper': 100}},
        'slack': {'foreach': [], 'where': 'budget > 1000', 'bounds': {'lower': 0, 'upper': 10}},
    },
    'constraints': {'cap': {'foreach': [], 'expression': 'sum(x, over=f) - slack <= budget'}},
    'objectives': {'total': {'sense': 'maximize', 'expression': 'x * cost'}},
}


def test_a_masked_out_scalar_variable_drops_the_row_that_uses_it():
    """Law 7 holds at no dimension either (#340).

    Was: a scalar's presence was `select()` over no dims, and polars cannot hold
    rows with no columns — collecting reports (0, 0), so present and absent were
    one frame and nothing downstream could restrict on it. `cap` stayed enforced
    with its term gone, as `sum(x) <= budget` — a constraint the file does not
    contain. The presence frame now carries a marker column instead, and a
    keyless restriction is a cross join.

    Held here as well as in the parity suite because that suite needs the
    ``[linopy]`` extra, and this has to be true on the bare install too.

    """
    data = {'cost': pl.DataFrame({'f': ['a', 'b'], 'value': [1.0, 2.0]}), 'budget': 120.0}

    with lps.solve(SCALAR_MASKED_MODEL, data) as sol:
        assert sol.dual('cap').height == 0, 'the row is gone, not slackened — a dropped row has no dual'
        assert sol.objective == pytest.approx(300.0), 'unbudgeted, both generators run flat out'


#: A masked variable broadcast onto a wider frame, then reduced back. `p` is
#: over (node, tech); `produces` adds `carrier`; the sum removes `tech`. So the
#: constraint's dims are neither a subset nor a superset of the variable's.
BROADCAST_MASK_MODEL = {
    'dimensions': {
        'node': {'values': ['n1', 'n2']},
        'tech': {'values': ['t1', 't2']},
        'carrier': {'values': ['elec', 'heat']},
    },
    'parameters': {
        'produces': {'dims': ['tech', 'carrier']},
        'demand': {'dims': ['node', 'carrier']},
        'cost': {'dims': ['tech']},
        'installed': {'dims': ['node', 'tech']},
    },
    'variables': {
        'p': {'foreach': ['node', 'tech'], 'where': 'installed > 0', 'bounds': {'lower': 0, 'upper': 'installed'}},
    },
    'constraints': {
        'balance': {'foreach': ['node', 'carrier'], 'expression': 'sum(p * produces, over=tech) == demand'},
    },
    'objectives': {'total': {'sense': 'minimize', 'expression': 'p * cost'}},
}


def test_a_mask_survives_a_broadcast_into_a_reduction():
    """`presence_dims=None` means "keyed by dims", and a product may *widen*
    dims — so carrying it through the widening re-read `p`'s (node, tech)
    presence as keyed by (node, tech, carrier) and `_propagate_absence` selected
    a column it never had (#345).

    Unmasked the same model was fine, which is what made it look like a problem
    with the coordinate dim rather than with the mask. The whole benchmark
    `sector` case sat on this.

    """
    import pandas as pd
    import xarray as xr

    data = {
        # a tech produces exactly one carrier, which is what makes `produces` sparse
        'produces': xr.DataArray(
            [[1.0, 0.0], [0.0, 1.0]],
            coords={'tech': ['t1', 't2'], 'carrier': ['elec', 'heat']},
            dims=['tech', 'carrier'],
        ),
        'demand': xr.DataArray(
            [[10.0, 20.0], [10.0, 20.0]],
            coords={'node': ['n1', 'n2'], 'carrier': ['elec', 'heat']},
            dims=['node', 'carrier'],
        ),
        'cost': pd.Series({'t1': 1.0, 't2': 2.0}),
        'installed': xr.DataArray(
            [[100.0, 100.0], [100.0, 100.0]], coords={'node': ['n1', 'n2'], 'tech': ['t1', 't2']}, dims=['node', 'tech']
        ),
    }

    with differential(BROADCAST_MASK_MODEL, data) as run:
        assert run.result.objective == pytest.approx(100.0, rel=RTOL)


LABEL_MODEL = {
    'dimensions': {'f': {'values': ['a', 'b']}},
    'parameters': {'cost': {'dims': ['f']}, 'cap': {'dims': ['f']}},
    'variables': {'x': {'foreach': ['f'], 'bounds': {'lower': 0, 'upper': 'cap'}}},
    'constraints': {'k': {'foreach': ['f'], 'expression': 'x <= cap'}},
    'objectives': {'o': {'sense': 'maximize', 'expression': 'x * cost'}},
}
_CAP = pl.DataFrame({'f': ['a', 'b'], 'value': [5.0, 5.0]})


def test_a_label_the_dimension_does_not_have_is_refused():
    """A typo used to be worth two thirds of the objective (#350).

    `b` mistyped as `zz` left `b` with no cost row, which reads as a zero
    coefficient — so the model solved, reported optimal, and returned 5.0 where
    15.0 is right. The eager lane already refused it; this lane joined the
    stray row against nothing and carried on.
    """
    ok = {'cost': pl.DataFrame({'f': ['a', 'b'], 'value': [1.0, 2.0]}), 'cap': _CAP}
    assert lps.solve(LABEL_MODEL, ok).objective == pytest.approx(15.0)

    typo = {'cost': pl.DataFrame({'f': ['a', 'zz'], 'value': [1.0, 2.0]}), 'cap': _CAP}
    with pytest.raises(DataError) as exc:
        lps.solve(LABEL_MODEL, typo)
    assert "'zz'" in str(exc.value), 'the refusal must name the offending label'
    assert 'typo' in str(exc.value)


def test_a_missing_row_is_still_only_sparse():
    """The distinction the refusal above rests on. A row that is *absent* is
    ordinary — it reads as a zero coefficient (§8) — and only a row that is
    present and unaddressable is a typo. Refusing both would make sparsity,
    which is the common case, an error.
    """
    sparse = {'cost': pl.DataFrame({'f': ['a'], 'value': [1.0]}), 'cap': _CAP}
    assert lps.solve(LABEL_MODEL, sparse).objective == pytest.approx(5.0)


def test_a_derived_dimension_cannot_have_a_stranger():
    """`values: null` takes the dimension's labels *from* the parameters, so the
    union of what arrived is the definition and the check has nothing to ask.
    Running it anyway would refuse every such model.
    """
    derived = {**LABEL_MODEL, 'dimensions': {'f': {'values': None}}}
    data = {'cost': pl.DataFrame({'f': ['a', 'b'], 'value': [1.0, 2.0]}), 'cap': _CAP}
    assert lps.solve(derived, data).objective == pytest.approx(15.0)


#: A `line` whose two endpoints are *both* multi-valued for one label — the case
#: that used to be reported one coordinate at a time.
TWO_BAD_COORDS_MODEL = {
    'dimensions': {'bus': {'values': ['b1', 'b2']}, 'line': {'coords': {'from': 'bus', 'to': 'bus'}}},
    'parameters': {'cap': {'dims': ['line']}},
    'variables': {'f': {'foreach': ['line'], 'bounds': {'lower': 0, 'upper': 'cap'}}},
    'constraints': {'k': {'foreach': ['line'], 'expression': 'f <= cap'}},
    'objectives': {'o': {'sense': 'maximize', 'expression': 'f'}},
}


def test_every_multi_valued_coordinate_is_named_at_once():
    """The per-coordinate loop this replaced raised on the first offender, so a
    source with two bad coordinates was fixed, rebuilt, and refused again (#273).

    Folding the counts into the caller's `group_by(d)` is what makes naming all
    of them free — they arrive in one frame instead of one pass each.
    """
    data = {
        'cap': pl.DataFrame({'line': ['l1'], 'value': [1.0]}),
        'bus': pl.DataFrame({'bus': ['b1', 'b2']}),
        'line': pl.DataFrame({'line': ['l1', 'l1'], 'from': ['b1', 'b2'], 'to': ['b2', 'b1']}),
    }

    with pytest.raises(DataError) as exc:
        lps.solve(TWO_BAD_COORDS_MODEL, data)

    message = str(exc.value)
    assert "'from'" in message and "'to'" in message, f'both offenders must be named; got: {message}'


def test_dense_columns_does_not_edit_the_model_it_projects():
    """Two solvers, two spellings of infinity, one model — and it survives both.

    `cols` has no `col` column: it is one row per column in label order, so the
    vectors a solver sink is handed are *views* of the frame rather than the
    scatter's fresh arrays. Replacing an infinity in place through one would
    rewrite the built model to suit whichever solver asked last, and the second
    ask would read bounds the first had already edited.

    """
    model = {
        'dimensions': {'i': {'dtype': 'int', 'values': [0, 1]}},
        'parameters': {'rhs': {'dims': ['i']}},
        'variables': {'x': {'foreach': ['i'], 'bounds': {'lower': 0}}},
        'constraints': {'c': {'foreach': ['i'], 'expression': 'x >= rhs'}},
        'objectives': {'o': {'sense': 'minimize', 'expression': 'sum(x, over=i)'}},
    }
    with lps.build(model, {'rhs': pl.DataFrame({'i': [0, 1], 'value': [1.0, 2.0]})}) as ex:
        tables = ex._tables()
        first, _, _, _ = tables.dense_columns(1e30)
        ub_after_first = tables.cols['ub'].to_list()

        _, second_ub, _, _ = tables.dense_columns(1e100)

        assert ub_after_first == tables.cols['ub'].to_list(), 'the projection edited the model'
        assert all(v == float('inf') for v in tables.cols['ub'].to_list()), 'the frame lost its infinities'
        assert list(second_ub) == [1e100, 1e100], "the second solver got the first solver's infinity"
        assert list(first) == [0.0, 0.0]


@pytest.mark.parametrize('where', [None, 'cap > 0'])
def test_cols_is_positional_so_a_row_index_is_its_solver_column(where):
    """Every row of `cols` sits at its own label, masked or not.

    `cols` carries no `col`: it is one row per column in label order, so a
    row's position *is* the solver's index. Both arithmetic label paths get
    that order from the emission order of a **cross join**, which is a property
    of polars rather than of this package — so it is checked here against the
    label frame, which knows the answer independently.

    Two dims on purpose: one dim is a scan and cannot be out of order, so a
    single-dim model would pass whatever the product did. Bounds are distinct
    per coordinate for the same reason — a frame permuted by a mask, a join, or
    a change in someone else's engine still has the right *multiset* of bounds
    and the wrong row for every one of them.
    """
    caps = [
        {'i': i, 'j': j, 'value': 0.0 if where and (i, j) == (1, 'b') else float(10 * i + ord(j))}
        for i in range(4)
        for j in ('a', 'b', 'c')
    ]
    model = {
        'dimensions': {'i': {'dtype': 'int', 'values': list(range(4))}, 'j': {'values': ['a', 'b', 'c']}},
        'parameters': {'cap': {'dims': ['i', 'j']}},
        'variables': {'x': {'foreach': ['i', 'j'], 'bounds': {'lower': 0, 'upper': 'cap'}}},
        'constraints': {'c': {'foreach': ['i', 'j'], 'expression': 'x <= cap'}},
        'objectives': {'o': {'sense': 'maximize', 'expression': 'sum(sum(x, over=j), over=i)'}},
    }
    if where:
        model['variables']['x']['where'] = where

    with lps.build(model, {'cap': pl.DataFrame(caps)}) as ex:
        tables = ex._tables()
        assert 'col' not in tables.cols.columns, 'cols carries an index it does not need'
        assert tables.cols.height == tables.column_count

        labels = ex._variables['x'].collect().sort('var_label')
        raw = pl.DataFrame(caps).with_columns(pl.col('j').cast(labels['j'].dtype))
        expected = labels.join(raw, on=['i', 'j'], how='left')['value'].to_list()
        assert tables.cols['ub'].to_list() == expected, 'a bound is attached to the wrong column'


#: A bound parameter dense over the whole variable product — a profile per
#: node, per hour. `p`'s upper bound spans exactly `p`'s foreach, so alignment
#: is positional rather than a join (compiler `_aligned_bound`).
DENSE_BOUND_MODEL = {
    'dimensions': {'t': {'dtype': 'int', 'values': [0, 1, 2]}, 'n': {'values': ['a', 'b']}},
    'parameters': {'avail': {'dims': ['t', 'n']}, 'cost': {'dims': ['n']}},
    'variables': {'p': {'foreach': ['t', 'n'], 'bounds': {'lower': 0, 'upper': 'avail'}}},
    'constraints': {'cap': {'foreach': ['t'], 'expression': 'sum(p, over=n) <= 100'}},
    'objectives': {'total': {'sense': 'maximize', 'expression': 'sum(p * cost, over=n)'}},
}

#: The same numbers the oracle sees, with the coordinates deliberately out of
#: order — a positional attach that skipped the sort would pair every
#: coordinate with another one's bound.
SHUFFLED_BOUND = pl.DataFrame(
    {
        't': [2, 0, 1, 2, 0, 1],
        'n': ['b', 'b', 'a', 'a', 'a', 'b'],
        'value': [6.0, 2.0, 3.0, 5.0, 1.0, 4.0],
    }
)
FLAT_COST = pl.DataFrame({'n': ['a', 'b'], 'value': [1.0, 1.0]})


def _aligned_for(model, data):
    """Which bound parameters took the positional path building *model*."""
    from lpspec.relational.engines.polars import compiler as compiler_module

    real = compiler_module.PolarsCompiler._aligned_bound
    seen = {}

    def spy(self, frame, param, v, alias):
        out = real(self, frame, param, v, alias)
        seen[param] = out is not None
        return out

    compiler_module.PolarsCompiler._aligned_bound = spy
    try:
        lps.build(model, data).close()
    except LpspecError:
        pass  # the decision is recorded while the plan is built, before it runs
    finally:
        compiler_module.PolarsCompiler._aligned_bound = real
    return seen


#: One dimension, so the oracle can take it: the eager loader wants a
#: `pd.Series` for a 1-D parameter and refuses a polars frame for a 2-D one.
FLAT_MODEL = {
    'dimensions': {'n': {'values': ['a', 'b', 'c']}},
    'parameters': {'avail': {'dims': ['n']}, 'cost': {'dims': ['n']}},
    'variables': {'p': {'foreach': ['n'], 'bounds': {'lower': 0, 'upper': 'avail'}}},
    'constraints': {'cap': {'foreach': [], 'expression': 'sum(p, over=n) <= 100'}},
    'objectives': {'total': {'sense': 'maximize', 'expression': 'sum(p * cost, over=n)'}},
}


def test_a_bound_dense_over_the_product_is_attached_by_position_not_joined(engine_internals):
    """The shape xarray gets for free: position *is* the coordinate.

    `avail` spans exactly `p`'s foreach and has a row per coordinate, so the
    label frame and the parameter describe the same product — and the label
    frame is in label order by construction. Sorting the parameter by the
    dimension ordinals reproduces that order, so the value column is attached
    rather than joined, which on `profiled/l` is most of a 0.58 s join.

    Checked against the oracle, because a misaligned bound is a different model
    rather than an error.

    """
    data = {'avail': pd.Series({'a': 1.0, 'b': 2.0, 'c': 3.0}), 'cost': pd.Series({'a': 1.0, 'b': 1.0, 'c': 1.0})}

    assert _aligned_for(FLAT_MODEL, data) == {'avail': True}
    with differential(FLAT_MODEL, data, lp=True) as run:
        assert run.result.objective == pytest.approx(1 + 2 + 3, rel=RTOL), (
            'every p at its own upper bound; the cap of 100 never binds'
        )


def test_the_positional_attach_sorts_rather_than_trusting_the_source_order(engine_internals):
    """The same numbers arriving shuffled must give the same model.

    A parameter's rows come in whatever order its source had. Attaching without
    sorting would be right only for a file that happened to be written in label
    order and silently wrong for every other one — which is the failure this
    whole path has to be gated against.
    """
    shuffled = {'avail': SHUFFLED_BOUND, 'cost': FLAT_COST}
    assert _aligned_for(DENSE_BOUND_MODEL, shuffled) == {'avail': True}

    with lps.solve(DENSE_BOUND_MODEL, shuffled) as result:
        assert result.objective == pytest.approx(1 + 2 + 3 + 4 + 5 + 6, rel=RTOL)
        got = result.primal('p').sort('t', 'n')['value'].to_list()
        assert got == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), 'each coordinate got its own bound'


def test_a_mask_or_a_sparse_bound_keeps_the_join(engine_internals):
    """Both ways position stops meaning the coordinate, refused by the gate.

    A `where` makes the label frame a subset of the product, and a parameter
    with a missing row makes the *parameter* one — either way the two sides no
    longer line up row for row, and attaching would pair a coordinate with
    another coordinate's bound. Neither is a shape the join cannot handle, so
    the gate declines and nothing else changes.
    """
    masked = override(DENSE_BOUND_MODEL, **{'variables.p.where': 't > 0'})
    assert _aligned_for(masked, {'avail': SHUFFLED_BOUND, 'cost': FLAT_COST}) == {'avail': False}

    holey = {'avail': SHUFFLED_BOUND.head(5), 'cost': FLAT_COST}
    assert _aligned_for(DENSE_BOUND_MODEL, holey) == {'avail': False}


def test_an_empty_index_keeps_the_dimension_s_declared_dtype():
    """polars infers `Null` from no labels, and a `Null` key joins nothing.

    A dimension whose members a caller appends to between solves starts empty —
    a Benders cut set is the motivating case — and the parameter bound to it
    carries the right dtype from the first call. Only the declaration knows
    what the column should be, and it always answers: `dtype` defaults to `str`.
    """
    model = {
        'dimensions': {'cut': {'dtype': 'int'}},
        'parameters': {'c': {'dims': ['cut']}},
        'variables': {'x': {'foreach': ['cut'], 'bounds': {'lower': 0}}},
        'objectives': {'o': {'sense': 'minimize', 'expression': 'x * c'}},
    }
    empty = pl.DataFrame(schema={'cut': pl.Int64, 'value': pl.Float64})
    with lps.build(model, {'c': empty}, coords={'cut': []}) as ex:
        assert ex._tables().column_count == 0

    grown = pl.DataFrame({'cut': [0, 1], 'value': [1.0, 2.0]})
    with lps.build(model, {'c': grown}, coords={'cut': [0, 1]}) as ex:
        assert ex._tables().column_count == 2


def test_an_empty_index_of_a_string_dimension_is_a_string_column():
    """The default `dtype` carries the same weight as a declared one."""
    model = {
        'dimensions': {'cut': {}},
        'parameters': {'c': {'dims': ['cut']}},
        'variables': {'x': {'foreach': ['cut'], 'bounds': {'lower': 0}}},
        'objectives': {'o': {'sense': 'minimize', 'expression': 'x * c'}},
    }
    empty = pl.DataFrame(schema={'cut': pl.String, 'value': pl.Float64})
    with lps.build(model, {'c': empty}, coords={'cut': []}) as ex:
        assert ex._tables().column_count == 0


def test_two_solutions_over_different_members_concatenate():
    """An `Enum` column will not concatenate against different categories.

    A sweep holds one frame per slice and joins them on read, so slices that
    bound different members of a dimension used to meet `SchemaError: Enum
    mismatch` — for two answers to the same question.
    """
    model = {
        'dimensions': {'node': {'dtype': 'str'}},
        'parameters': {'cap': {'dims': ['node']}},
        'variables': {'x': {'foreach': ['node'], 'bounds': {'lower': 0, 'upper': 'cap'}}},
        'constraints': {'k': {'foreach': ['node'], 'expression': 'x >= cap'}},
        'objectives': {'o': {'sense': 'minimize', 'expression': 'sum(x, over=node)'}},
    }
    frames = []
    for members in (['a', 'b'], ['a', 'c']):
        cap = pl.DataFrame({'node': members, 'value': [1.0, 2.0]})
        with lps.build(model, {'cap': cap}) as ex:
            frames.append(ex.solve().primal('x'))

    assert pl.concat(frames).height == 4

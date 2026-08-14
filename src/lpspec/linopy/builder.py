"""Model builder: schema + data → linopy Model.

Also the eager evaluation of every built-in helper. The helper *names* are the
language (``helpers.py``, imported by the linopy-free lane); these
xarray/linopy evaluations are this backend's private business, mirrored on the
relational side by lowering cases rather than shared code.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, assert_never

import numpy as np
import xarray as xr

from lpspec._notes import note
from lpspec.errors import DataError, LanguageError, null_bounds_message
from lpspec.language import degree
from lpspec.language.expression_parser import (
    ArithmeticNode,
    BinaryOperatorNode,
    ComparisonNode,
    CoordinateNode,
    DimensionNode,
    EdgeNode,
    FunctionCallNode,
    KeywordNode,
    NameNode,
    NumberNode,
    ParameterNode,
    UnaryOperatorNode,
    VariableNode,
)
from lpspec.language.helpers import EDGE_WRAP, unknown_helper_message
from lpspec.language.resolution import Namespace, expression_of, where_of
from lpspec.language.where_parser import (
    AndNode,
    BooleanLiteralNode,
    DimensionComparisonNode,
    NotNode,
    OrNode,
    ParameterComparisonNode,
    ParameterDefinedNode,
    UnresolvedComparisonNode,
    UnresolvedNameNode,
    VariableDefinedNode,
    WhereNode,
)
from lpspec.linopy.loader import check_constant_side_covers, check_divisors_cover, gaps_under

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Mapping

    import linopy
    import pandas as pd

    from lpspec.language.model import Model

_SIGN_MAP = {'==': '=', '<=': '<=', '>=': '>='}

#: The language's arithmetic. ``**`` is absent on purpose — see ``_eval_ast``.
_ARITHMETIC_OPS: dict[str, Callable[[Any, Any], Any]] = {
    '+': operator.add,
    '-': operator.sub,
    '*': operator.mul,
    '/': operator.truediv,
}

#: Where-comparison operators, evaluated element-wise on a DataArray.
_PREDICATE_OPS: dict[str, Callable[[Any, Any], Any]] = {
    '==': operator.eq,
    '!=': operator.ne,
    '<': operator.lt,
    '>': operator.gt,
    '<=': operator.le,
    '>=': operator.ge,
}


@dataclass(frozen=True)
class EvaluationContext:
    """Everything expression evaluation needs to resolve names.

    Extend this rather than adding parameters to ``_eval_ast`` and every
    helper-facing seam.
    """

    model: linopy.Model
    dataset: xr.Dataset
    master_coords: dict[str, pd.Index]
    schema: Model
    ns: Namespace
    #: dim -> {coordinate name: values as a DataArray over that dim}
    dim_coords: dict[str, dict[str, xr.DataArray]] = field(default_factory=dict)


def build_model(
    model: linopy.Model,
    schema: Model,
    dataset: xr.Dataset,
    master_coords: dict[str, pd.Index],
    dim_coords: dict[str, dict[str, xr.DataArray]] | None = None,
) -> None:
    """Populate a linopy Model from a parsed schema and loaded parameters.

    This mutates *model* in-place, adding variables, constraints, and
    objectives as declared in *schema*.
    """
    ctx = EvaluationContext(
        model,
        dataset,
        master_coords,
        schema,
        Namespace.of(schema, list(model.variables)),
        dim_coords or {},
    )
    _build_variables(ctx)
    _build_sos(ctx)
    _build_constraints(ctx)
    _build_objectives(ctx)


# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------


def _build_variables(ctx: EvaluationContext) -> None:
    for vname, vdef in ctx.schema.variables.items():
        with note(f"while building variable '{vname}'"):
            coords = {d: ctx.master_coords[d] for d in vdef.foreach}

            lower = _resolve_bound(vdef.bounds.lower, ctx.dataset)
            upper = _resolve_bound(vdef.bounds.upper, ctx.dataset)

            where = where_of(vdef.where, ctx.ns, f"variable '{vname}'", self_variable=vname)
            mask = evaluate_where(where, ctx.dataset, ctx.master_coords, ctx.model)

            _check_bounds_are_defined(vname, vdef, ctx.dataset, mask)

            ctx.model.add_variables(
                lower=lower,
                upper=upper,
                coords=coords,
                name=vname,
                mask=_as_linopy_mask(mask),
                binary=vdef.domain == 'binary',
                integer=vdef.domain == 'integer',
                semi_continuous=vdef.domain == 'semi_continuous',
            )


def _check_bounds_are_defined(name: str, vdef: Any, dataset: xr.Dataset, mask: Any) -> None:
    """Refuse a bound with no value, at build, as the native lane does.

    Otherwise the NaN travels into linopy and surfaces two phases later from
    inside its IO layer — ``Continuous Variable x contains nan's in field(s)
    ['upper']``, raised at solve or write, naming neither the YAML nor the fix,
    from a ``build()`` that had already returned.

    Checked against the variable's own mask: a coordinate the variable does not
    occupy needs no bound, and supplying data only where it exists is the
    ordinary idiom.
    """
    missing = sum(
        gaps_under(dataset[bound], mask) for bound in (vdef.bounds.lower, vdef.bounds.upper) if isinstance(bound, str)
    )
    if missing:
        raise DataError(null_bounds_message(name, missing))


def _resolve_bound(
    value: float | str,
    dataset: xr.Dataset,
) -> Any:
    """Resolve a bound value — either a literal number or a parameter name."""
    if isinstance(value, str):
        if value not in dataset:
            msg = (
                f"Bound references parameter '{value}' which is not in the "
                f'loaded dataset. Available: {sorted(map(str, dataset.data_vars))}'
            )
            raise DataError(msg)
        return dataset[value]
    return value


def _as_linopy_mask(mask: xr.DataArray) -> xr.DataArray | None:
    """Convert an evaluated where mask to linopy's ``mask=`` argument.

    linopy expects ``None`` for "no mask"; a 0-d True mask means exactly
    that. Everything else (including 0-d False) passes through.
    """
    if mask.ndim == 0 and bool(mask):
        return None
    return mask


# ---------------------------------------------------------------------------
# Special-ordered sets
# ---------------------------------------------------------------------------


def _build_sos(ctx: EvaluationContext) -> None:
    """Attach every ``sos:`` block to the variable it names.

    linopy holds a set the same way the language declares one — a variable, a
    dimension of it, a type — so this is the block handed over, not a
    formulation rebuilt. Which is the point of copying its decomposition:
    the eager lane is the oracle, and a set it had to *reformulate* to accept
    would be an oracle for a different model.

    It runs before the constraints because a set is a property of the
    variable, so it belongs beside the declaration rather than after
    everything that uses it.
    """
    for name, sos in ctx.schema.sos.items():
        with note(f"while building sos '{name}'"):
            ctx.model.add_sos_constraints(
                ctx.model.variables[sos.variable],
                sos_type=1 if sos.type == 1 else 2,
                sos_dim=sos.over,
                big_m=sos.big_m,
            )


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------


def _build_constraints(ctx: EvaluationContext) -> None:
    for cname, cdef in ctx.schema.constraints.items():
        with note(f"while building constraint '{cname}'"):
            c_where = where_of(cdef.where, ctx.ns, f"constraint '{cname}'")
            mask = evaluate_where(c_where, ctx.dataset, ctx.master_coords, ctx.model)

            ast = expression_of(cdef.expression, ctx.schema, ctx.ns, f"constraint '{cname}'")
            if not isinstance(ast, ComparisonNode):
                msg = f'expression must contain exactly one comparison operator (<=, >=, ==).\nGot: {cdef.expression!r}'
                raise LanguageError(msg)

            check_divisors_cover(f"constraint '{cname}'", ast, ctx.schema, ctx.dataset, mask, ctx.model)
            check_constant_side_covers(f"constraint '{cname}'", ast, ctx.schema, ctx.dataset, mask)

            lhs = _eval_ast(ast.left, ctx)
            rhs = _eval_ast(ast.right, ctx)
            sign = _SIGN_MAP[ast.op]

            ctx.model.add_constraints(lhs, sign, rhs, name=cname, mask=_as_linopy_mask(mask))


# ---------------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------------


def _build_objectives(ctx: EvaluationContext) -> None:
    """Build every declared objective onto the model.

    An objective has no ``where``, so its divisor check runs with no row mask
    — the numerator's own presence is the only thing that can excuse a gap.
    """
    for oname, odef in ctx.schema.objectives.items():
        with note(f"while building objective '{oname}'"):
            ast = expression_of(odef.expression, ctx.schema, ctx.ns, f"objective '{oname}'")

            if isinstance(ast, ComparisonNode):
                msg = f'Expression must not contain a comparison operator. Got: {odef.expression!r}'
                raise LanguageError(msg)

            check_divisors_cover(f"objective '{oname}'", ast, ctx.schema, ctx.dataset, None, ctx.model)

            expr = _objective_expression(ast, ctx)

            sense = 'min' if odef.sense == 'minimize' else 'max'
            ctx.model.add_objective(expr, overwrite=True, sense=sense)


def _objective_expression(node: ArithmeticNode, ctx: EvaluationContext) -> Any:
    """*node* as a scalar: each additive term summed over the dims it carries.

    An objective has no ``foreach``, so every dim it names is summed (SPEC §2)
    — but *which* dims, per term rather than per objective. In
    ``x[i] * a[i] + y[j] * b[j]`` the first term has ``|i|`` summands and the
    second ``|j|``. Adding the operands first, as linopy's ``+`` does,
    broadcasts both to ``(i, j)`` and counts each term once per coordinate of
    the other, so an objective spanning a sparse and a dense variable comes out
    multiplied rather than summed.

    The relational lane never had the problem, an expression there being term
    fragments that each keep their own dims. Distributing the sum over addition
    reproduces that, which is what hard rule 3 requires (#197).
    """
    total: Any = None
    for term in _additive_terms(node, ctx):
        scalar = term.sum() if hasattr(term, 'sum') else term
        total = scalar if total is None else total + scalar
    return total


def _additive_terms(node: ArithmeticNode, ctx: EvaluationContext) -> list[Any]:
    """*node* as a list of terms to be summed, multiplication distributed.

    Only the operators that distribute are walked; everything else is one
    opaque term, a helper call having already reduced whatever it reduces.
    Distribution is what keeps ``(x[i] * a[i] + y[j] * b[j]) * c[k]`` two terms
    rather than one broadcast to ``(i, j, k)``.

    Degree 1 makes it safe: ``degree.check_binary`` has already refused the one
    product that would not survive, and a divisor carries no variables, so it
    is a single value applied to each term.
    """
    if isinstance(node, UnaryOperatorNode) and node.op in {'+', '-'}:
        terms = _additive_terms(node.operand, ctx)
        return [-t for t in terms] if node.op == '-' else terms

    if isinstance(node, BinaryOperatorNode):
        degree.check_binary(node)
        if node.op == '+':
            return _additive_terms(node.left, ctx) + _additive_terms(node.right, ctx)
        if node.op == '-':
            return _additive_terms(node.left, ctx) + [-t for t in _additive_terms(node.right, ctx)]
        if node.op == '*':
            return [
                left * right for left in _additive_terms(node.left, ctx) for right in _additive_terms(node.right, ctx)
            ]
        if node.op == '/':
            divisor = _eval_ast(node.right, ctx)
            return [term / divisor for term in _additive_terms(node.left, ctx)]

    return [_eval_ast(node, ctx)]


# ---------------------------------------------------------------------------
# AST evaluation
# ---------------------------------------------------------------------------


def _eval_ast(
    node: ArithmeticNode,
    ctx: EvaluationContext,
) -> Any:
    """Evaluate an expression AST node against the model namespace.

    Binary nodes go through ``degree.check_binary`` first: ``**``, a quadratic
    product and a variable divisor are all refused by ``language/degree.py``,
    the same verdict the relational lane asks for and in the same sentence.

    Unknown helper names were rejected by ``validation.py`` at load time; the
    guard on ``_HELPERS`` covers hand-built calls that skipped it.
    """
    if isinstance(node, NumberNode):
        return node.value

    if isinstance(node, VariableNode):
        return ctx.model.variables[node.name]

    if isinstance(node, ParameterNode):
        return _coefficient(ctx.dataset[node.name])

    if isinstance(node, EdgeNode):
        msg = f'EdgeNode({node.policy!r}) reached the evaluator: an edge policy is a shift() kwarg, not a value.'
        raise AssertionError(msg)

    if isinstance(node, KeywordNode):
        msg = (
            f'KeywordNode({node.value!r}) reached the evaluator. A quoted keyword is '
            f'consumed by its kwarg during resolution — reaching here means it was written '
            f'where no kwarg expects one.'
        )
        raise AssertionError(msg)
    if isinstance(node, (NameNode, DimensionNode, CoordinateNode)):
        msg = (
            f'{type(node).__name__}({node.name!r}) reached the evaluator. '
            f'Expressions must go through resolution.expression_of() first '
            f'(docs/ARCHITECTURE.md hard rule 1).'
        )
        raise AssertionError(msg)

    if isinstance(node, UnaryOperatorNode):
        operand = _eval_ast(node.operand, ctx)
        if node.op == '-':
            return -operand
        return operand

    if isinstance(node, BinaryOperatorNode):
        degree.check_binary(node)
        left = _eval_ast(node.left, ctx)
        right = _eval_ast(node.right, ctx)
        return _ARITHMETIC_OPS[node.op](left, right)

    if isinstance(node, FunctionCallNode):
        if node.name not in _HELPERS:
            raise NameError(unknown_helper_message(node.name))
        helper = _HELPERS[node.name]
        args = [_eval_ast(a, ctx) for a in node.args]
        if node.name == 'at':
            by = node.kwargs['by']
            assert isinstance(by, CoordinateNode)
            return _helper_at(args[0], _coordinate_array(by, ctx), into=by.into)
        if (by := node.kwargs.get('group_by')) is not None:
            assert isinstance(by, CoordinateNode)
            return _helper_grouped_sum(args[0], _coordinate_array(by, ctx), into=by.into)
        kwargs: dict[str, Any] = {}
        for k, v in node.kwargs.items():
            if isinstance(v, DimensionNode):
                kwargs[k] = v.name
            elif isinstance(v, EdgeNode):
                kwargs[k] = v.policy
            else:
                kwargs[k] = _eval_ast(v, ctx)
        return helper(*args, **kwargs)

    assert_never(node)


def _coordinate_array(by: CoordinateNode, ctx: EvaluationContext) -> Any:
    """The declared coordinate ``by`` as an array over the dimension carrying it.

    Looked up rather than evaluated as an operand: the coordinate lives on the
    dimension, not in the parameter dataset.
    """
    try:
        return ctx.dim_coords[by.dimension][by.name]
    except KeyError:
        msg = (
            f"coordinate '{by.name}' on dimension '{by.dimension}' has no bound values. "
            f"Pass coords={{'{by.dimension}': <DataFrame with '{by.dimension}' and "
            f"'{by.name}' columns>}}."
        )
        raise DataError(msg) from None


# ---------------------------------------------------------------------------
# Built-in helpers, eager evaluation — each operand is an xr.DataArray (a
# parameter) or a linopy Variable / LinearExpression
# ---------------------------------------------------------------------------


def _helper_sum(array: Any, *, over: str) -> Any:
    """Sum *array* over dimension *over*.

    A DataArray and a linopy expression both carry ``dims`` and both take the
    dim positionally, so there is one branch: if the array does not have the
    named dimension, it is returned unchanged.
    """
    if over in getattr(array, 'dims', ()):
        return array.sum(over)
    return array


def _helper_grouped_sum(array: Any, mapping: Any, *, into: str) -> Any:
    """Sum *array* through a declared coordinate, producing dimension *into*.

    YAML: ``sum(p, over=generator, group_by=bus)``. *mapping* is the
    coordinate's values as a one-dimensional array over the dim being grouped,
    from ``EvaluationContext.dim_coords``; that dim is summed out and *into*
    holds the group labels.

    A null coordinate says the label belongs to no group, so its terms
    contribute nowhere. linopy refuses to group by NaN at all, so those members
    are dropped before grouping rather than after.
    """
    if not isinstance(mapping, xr.DataArray):
        msg = (
            f'sum(group_by=) coordinate must be an array (got '
            f'{type(mapping).__name__}). Usage: sum(expr, over=dim, group_by=coord)'
        )
        raise TypeError(msg)
    if mapping.ndim != 1:
        msg = f'sum(group_by=) mapping must have exactly one dimension, got {list(mapping.dims)}'
        raise LanguageError(msg)

    group = mapping.rename(into)
    present = group.notnull()
    if not bool(present.all()):
        dim = str(group.dims[0])
        group = group.isel({dim: present.to_numpy()})
        array = array.isel({dim: present.to_numpy()})
    if isinstance(array, xr.DataArray) or hasattr(array, 'groupby'):
        return array.groupby(group).sum()
    raise _unsupported('sum(group_by=)', array)


def _helper_at(array: Any, mapping: Any, *, into: str) -> Any:
    """Read *array* through a declared coordinate — the adjoint of a group.

    YAML: ``at(on, onto=flow, by=component)``. *mapping* is the same
    one-dimensional array ``sum`` takes; grouping sums *along* it, this indexes
    *through* it, so the operand must carry ``into`` and the result carries the
    mapping's own dim.

    xarray's vectorised selection is the pullback exactly — one ``into`` label
    read once per fine label pointing at it — so the fan-out is the indexer's
    doing rather than a broadcast arranged here.

    A null coordinate reads nothing and its row is absent, the same reading
    ``sum`` gives a null group.
    """
    if not isinstance(mapping, xr.DataArray):
        msg = f'at() coordinate must be an array (got {type(mapping).__name__}). Usage: at(expr, onto=dim, by=coord)'
        raise TypeError(msg)
    if mapping.ndim != 1:
        msg = f'at() mapping must have exactly one dimension, got {list(mapping.dims)}'
        raise LanguageError(msg)

    present = mapping.notnull()
    if not bool(present.all()):
        dim = str(mapping.dims[0])
        mapping = mapping.isel({dim: present.to_numpy()})
    if isinstance(array, xr.DataArray) or hasattr(array, 'sel'):
        return array.sel({into: mapping.rename(into)})
    raise _unsupported('at()', array)


def _unsupported(call: str, array: Any) -> TypeError:
    """One wording for an operand shape a helper cannot take.

    Reached only from a hand-built call: every helper's operands come from
    ``_eval_ast``, so a lane running the language proper never sees this.
    """
    return TypeError(f"{call} does not support type '{type(array).__name__}'.")


def _translation(over: str, by: float) -> Mapping[Hashable, int]:
    """The ``{dim: n}`` mapping xarray and linopy both take."""
    if int(by) != by:
        msg = f'shift() by must be an integer, got {by!r}'
        raise TypeError(msg)
    return {over: int(by)}


def _helper_shift(array: Any, *, over: str, by: float, edge: str | float | None = None) -> Any:
    """Translate *array* along one dimension — the value at *t - by*.

    YAML: ``shift(soc, over=snapshot, by=1)``. ``edge`` carries all three
    policies so no two keywords can disagree: ``edge='wrap'`` is cyclic and
    vacates nothing, a number is what the vacated positions contribute, and
    omitting it leaves them **absent**, which propagates and drops the row.
    Nothing is done to the result in that default case — linopy v1 already
    gives that answer (#289).

    A DataArray shift always fills, absence not being representable in data, so
    lowering refuses a bare shift over a variable-free operand and that branch
    is only reached under a numeric ``edge=``.
    """
    amount = _translation(over, by)
    if edge == EDGE_WRAP:
        if isinstance(array, xr.DataArray):
            return array.roll(amount, roll_coords=False)
        if hasattr(array, 'roll'):
            return array.roll(amount)
        raise _unsupported("shift(edge='wrap')", array)
    if isinstance(edge, str):
        msg = f'shift(edge={edge!r}) reached the evaluator: only {EDGE_WRAP!r} or a number resolve.'
        raise AssertionError(msg)
    fill = edge
    if isinstance(array, xr.DataArray):
        return array.shift(amount, fill_value=fill if fill is not None else np.nan)
    if hasattr(array, 'shift'):
        shifted = array.shift(amount)
        return shifted if fill is None else _vacated(shifted, fill)
    raise _unsupported('shift()', array)


#: Eager evaluation of every name in ``helpers.BUILTIN_NAMES``. The two must
#: agree exactly — enforced by ``tests/test_architecture.py``, because a name
#: one lane implements and the other does not is precisely the divergence
#: that would make the differential tests a comparison of dialects.
_HELPERS: dict[str, Callable[..., Any]] = {
    'sum': _helper_sum,
    'at': _helper_at,
    'shift': _helper_shift,
}


# ---------------------------------------------------------------------------
# Where-mask evaluation
# ---------------------------------------------------------------------------


def evaluate_where(
    node: WhereNode | None,
    dataset: xr.Dataset,
    master_coords: dict[str, pd.Index],
    model: linopy.Model | None = None,
) -> xr.DataArray:
    """Evaluate a **resolved** where AST against a parameter dataset.

    A node, not a string: resolution has already decided what every name refers
    to, so this performs no lookups and cannot disagree with the relational lane
    about scoping. It lives here rather than in ``where_parser.py`` because it
    is xarray-only.

    Always a boolean DataArray. The no-mask case comes back 0-dimensional, so
    callers combine with ``&``/``|`` without case analysis.
    """
    if node is None:
        return xr.DataArray(True)

    return _eval_node(node, dataset, master_coords, model)


def _eval_node(
    node: WhereNode,
    dataset: xr.Dataset,
    master_coords: dict[str, pd.Index],
    model: linopy.Model | None = None,
) -> xr.DataArray:
    """One resolved where node as a boolean DataArray.

    Two absences read as exclusion rather than as an answer: a variable's
    masked-out coordinate carries label ``-1`` — linopy's own marker for an
    absent slot, which is exactly the question ``defined(v)`` asks — and a
    comparison over NaN comes back false. Comparison right-hand sides are
    literals; resolution rejected a parameter or variable there.
    """

    def evaluate(child: WhereNode) -> xr.DataArray:
        """Recurse carrying this call's bindings — what the connectives need."""
        return _eval_node(child, dataset, master_coords, model)

    if isinstance(node, BooleanLiteralNode):
        return xr.DataArray(node.value)

    if isinstance(node, (UnresolvedNameNode, UnresolvedComparisonNode)):
        msg = (
            f'{type(node).__name__} reached the evaluator unresolved. '
            f'Where strings must go through resolution.resolve_where() first.'
        )
        raise AssertionError(msg)

    if isinstance(node, ParameterDefinedNode):
        arr = dataset[node.name]
        if arr.dtype == bool:
            return arr
        return arr.notnull() & np.isfinite(arr)

    if isinstance(node, VariableDefinedNode):
        if model is None:
            msg = (
                f"where references variable '{node.name}', but no model was passed to the "
                f'evaluator — a variable mask can only be read off the model that holds it.'
            )
            raise AssertionError(msg)
        return model.variables[node.name].labels != -1

    if isinstance(node, (ParameterComparisonNode, DimensionComparisonNode)):
        if isinstance(node, ParameterComparisonNode):
            arr = dataset[node.name]
        else:
            arr = xr.DataArray(
                master_coords[node.name],
                coords={node.name: master_coords[node.name]},
                dims=[node.name],
            )

        result = _PREDICATE_OPS[node.op](arr, node.value)
        return result.fillna(False).astype(bool)

    if isinstance(node, NotNode):
        return ~evaluate(node.operand)

    if isinstance(node, AndNode):
        return evaluate(node.left) & evaluate(node.right)

    if isinstance(node, OrNode):
        return evaluate(node.left) | evaluate(node.right)

    assert_never(node)


def _coefficient(parameter: Any) -> Any:
    """A parameter in a coefficient position, its uncovered slots at zero.

    Where this lane answers linopy's v1 absence convention (linopy's
    ``doc/design/convention.rst``): the answer is *positional* — one missing
    row means zero in a coefficient, an error in ``bounds:``, false in a
    ``where`` operand — so it lives at the read, not as one fill in
    ``load_parameters`` that would be wrong for the other two. A tidy
    parameter table is a compressed dense array, not a record of absence:
    rows only for the live coordinates says the coefficient is zero elsewhere
    (SPEC §8). ``load_parameters`` reindexes to the master coordinates, so an
    uncovered slot arrives as NaN — and v1 §5 refuses a NaN in a
    user-supplied constant. Correct under the legacy convention too, so not
    conditional on ``linopy.options['semantics']``.
    """
    return parameter.fillna(0.0)


def _vacated(expression: Any, fill: float) -> Any:
    """A shifted expression with its vacated edge positions filled.

    linopy v1 counts ``.shift()`` among the operations that *create* absence
    (§4), so the edge propagates and drops the row — the language's answer too
    (SPEC §7, #289). This is the opt-out, reached only from ``shift(...,
    fill=0)``, and is the escape v1 itself prescribes rather than a rule of
    ours on top.

    ``to_linexpr()`` first when the operand is still a bare ``Variable``:
    ``Variable.fillna`` means a label fill on the released line and an
    expression fill on the v1 branch, and only the expression method is stable.
    """
    if hasattr(expression, 'to_linexpr'):
        expression = expression.to_linexpr()
    return expression.fillna(fill)

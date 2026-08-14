"""Lower a parsed YAML schema (typed AST) to the relational logical plan.

The lowering seam (docs/ARCHITECTURE.md, "The relational lane"): it consumes
the same typed AST the eager builder evaluates and emits a
:class:`~lpspec.relational.plan.Program`. It lives on the language side, so the
engine subpackage stays free of YAML knowledge and this module never imports
the eager builder.

Constructs with no lowering raise :class:`~lpspec.errors.LanguageError` naming
the construct and its rewrite, never a pointer to another backend: the two
lanes accept the same language, so a rejection here is a language gap
(docs/ROADMAP.md) rather than a routing decision.

Semantics mirror the eager builder exactly:

- a reduction over a dim the operand does not carry is an error, not a silent
  identity — ``dimensions.py`` owns that rule and this module asks it;
- a constraint is **one rule** carrying its own name, so a row is read back by
  the name the file writes, with no positional suffix to guess (#298);
- a file declares one objective, likewise one expression;
- an objective sums each term over the dims that term carries, which term
  fragments do for free and the eager lane has to distribute for.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, assert_never, cast

from lpspec.errors import LanguageError
from lpspec.language import degree
from lpspec.language.dimensions import dims_of
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
from lpspec.language.helpers import call_shape_error, cumsum_over_variable_message, edge_error
from lpspec.language.piecewise import expand_piecewise
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
from lpspec.relational import plan

if TYPE_CHECKING:
    from lpspec.language.model import Model

_SENSES = {'==', '<=', '>='}


def lower_program(schema: Model) -> plan.Program:
    """Compile a validated :class:`Model` into a :class:`Program`.

    A ``binary:`` variable lowers with fixed 0/1 bounds, matching linopy's
    ``binary=True``.

    Raises:
        LanguageError: A construct outside the streaming language, named with
            its rewrite.
    """
    schema = expand_piecewise(schema)
    ns = Namespace.of(schema)
    parameters = tuple(plan.ParameterDeclaration(name, tuple(pdef.dims)) for name, pdef in schema.parameters.items())

    variables = []
    for vname, vdef in schema.variables.items():
        variable_type: plan.VariableType
        if vdef.binary:
            variable_type, lower, upper = 'binary', plan.Constant(0.0), plan.Constant(1.0)
        else:
            variable_type = 'integer' if vdef.integer else 'continuous'
            lower, upper = _bound_expression(vdef.bounds.lower), _bound_expression(vdef.bounds.upper)
        variables.append(
            plan.VariableDeclaration(
                vname,
                tuple(vdef.foreach),
                where=_lower_where(vdef.where, ns, f"variable '{vname}'", self_variable=vname),
                lower=lower,
                upper=upper,
                variable_type=variable_type,
            )
        )

    constraints = []
    for cname, cdef in schema.constraints.items():
        where = _lower_where(cdef.where, ns, f"constraint '{cname}'")
        ast = expression_of(cdef.expression, schema, ns, f"constraint '{cname}'")
        if not isinstance(ast, ComparisonNode):
            raise LanguageError(
                f"constraint '{cname}': expression must contain exactly one "
                f'comparison operator (<=, >=, ==). Got: {cdef.expression!r}'
            )
        if ast.op not in _SENSES:
            raise LanguageError(f"constraint '{cname}': unsupported sense '{ast.op}'")
        constraints.append(
            plan.ConstraintDeclaration(
                cname,
                tuple(cdef.foreach),
                lhs=_lower_expr(ast.left, schema, f"constraint '{cname}'"),
                sense=ast.op,
                rhs=_lower_expr(ast.right, schema, f"constraint '{cname}'"),
                where=where,
            )
        )

    if not schema.objectives:
        raise LanguageError('the relational backend requires an objective')
    oname, odef = next(iter(schema.objectives.items()))
    ast = expression_of(odef.expression, schema, ns, f"objective '{oname}'")
    if isinstance(ast, ComparisonNode):
        raise LanguageError(f"objective '{oname}': expression must not contain a comparison operator")
    objective = plan.ObjectiveDeclaration(
        'min' if odef.sense == 'minimize' else 'max',
        _lower_expr(ast, schema, f"objective '{oname}'"),
    )

    dimensions = tuple(
        plan.DimensionDeclaration(
            dname,
            tuple(plan.CoordinateTarget(cname, target) for cname, target in ddef.targeted.items()),
            tuple(ddef.labels),
        )
        for dname, ddef in schema.dimensions.items()
    )
    sos = tuple(
        plan.SosDeclaration(
            sname,
            sdef.variable,
            sdef.over,
            sos_type=cast('Literal[1, 2]', sdef.type),
            big_m=sdef.big_m,
        )
        for sname, sdef in schema.sos.items()
    )
    return plan.Program(parameters, tuple(variables), tuple(constraints), objective, dimensions, sos)


# ---------------------------------------------------------------------------
# expression lowering
# ---------------------------------------------------------------------------


def _lower_expr(node: ArithmeticNode, schema: Model, context: str) -> plan.Expression:
    """Rewrite one resolved core-AST expression as a plan expression.

    Three language rules are *asked* here and answered elsewhere: the call
    shape (``helpers.call_shape_error``, re-asked so an AST that skipped
    resolution gets the language's wording rather than an ``IndexError``), the
    dim rules (``dimensions.dims_of``) and degree (``language/degree.py``).

    What stays is about the plan: which node a call becomes, and the shapes a
    node cannot represent — a ``GroupSum`` groups by a declared coordinate, a
    ``Translate`` distance is an integer literal. ``Sum`` and ``GroupSum`` stay
    two nodes under one surface verb, reducing a dim away and reducing it into
    another being different relational shapes.
    """
    if isinstance(node, NumberNode):
        return plan.Constant(node.value)

    if isinstance(node, VariableNode):
        return plan.Variable(node.name)

    if isinstance(node, ParameterNode):
        return plan.Parameter(node.name)

    if isinstance(node, EdgeNode):
        msg = f'EdgeNode({node.policy!r}) reached lowering: an edge policy is a shift() kwarg, not a value.'
        raise AssertionError(msg)

    if isinstance(node, KeywordNode):
        msg = (
            f'KeywordNode({node.value!r}) reached lowering. A quoted keyword is consumed '
            f'by its kwarg during resolution (docs/ARCHITECTURE.md hard rule 1).'
        )
        raise AssertionError(msg)
    if isinstance(node, (NameNode, DimensionNode, CoordinateNode)):
        msg = (
            f'{type(node).__name__}({node.name!r}) reached lowering. Expressions '
            f'must go through resolution.expression_of() first '
            f'(docs/ARCHITECTURE.md hard rule 1).'
        )
        raise AssertionError(msg)

    if isinstance(node, UnaryOperatorNode):
        inner = _lower_expr(node.operand, schema, context)
        return plan.Negate(inner) if node.op == '-' else inner

    if isinstance(node, BinaryOperatorNode):
        left = _lower_expr(node.left, schema, context)
        right = _lower_expr(node.right, schema, context)
        degree.check_binary(node, context)
        match node.op:
            case '+':
                return plan.Add(left, right)
            case '-':
                return plan.Add(left, plan.Negate(right))
            case '*':
                return plan.Multiply(left, right)
            case '/':
                return plan.Divide(left, right)
            case _:  # pragma: no cover — check_binary refuses every other operator
                raise AssertionError(f'{context}: operator {node.op!r} passed the degree check')

    if isinstance(node, FunctionCallNode):
        shape_error = call_shape_error(node.name, len(node.args), node.kwargs)
        if shape_error is not None:
            raise LanguageError(f'{context}: {shape_error}')

        if node.name == 'sum':
            over_node = node.kwargs['over']
            if not isinstance(over_node, DimensionNode):
                raise LanguageError(f'{context}: sum(over=...) must name a dimension')
            by_node = node.kwargs.get('group_by')
            _check_dim_rules(node, schema, context)
            operand = _lower_expr(node.args[0], schema, context)
            if by_node is None:
                return plan.Sum(operand, (over_node.name,))
            if not isinstance(by_node, CoordinateNode):
                raise LanguageError(f'{context}: sum(group_by=...) must name a coordinate')
            return plan.GroupSum(
                operand,
                over=over_node.name,
                coordinate=by_node.name,
                into=by_node.into,
            )

        if node.name == 'at':
            over_node = node.kwargs['onto']
            by_node = node.kwargs['by']
            if not isinstance(over_node, DimensionNode):
                raise LanguageError(f'{context}: at(onto=...) must name a dimension')
            if not isinstance(by_node, CoordinateNode):
                raise LanguageError(f'{context}: at(by=...) must name a coordinate')
            _check_dim_rules(node, schema, context)
            return plan.At(
                _lower_expr(node.args[0], schema, context),
                over=over_node.name,
                coordinate=by_node.name,
                into=by_node.into,
            )

        if node.name == 'shift':
            over_node = node.kwargs['over']
            if not isinstance(over_node, DimensionNode):
                raise LanguageError(f'{context}: shift(over=...) must name a dimension')
            by_node = node.kwargs['by']
            sign = 1
            if isinstance(by_node, UnaryOperatorNode) and by_node.op == '-':
                sign, by_node = -1, by_node.operand
            if not isinstance(by_node, NumberNode) or int(by_node.value) != by_node.value:
                raise LanguageError(f'{context}: shift(by=...) must be an integer literal')
            _check_dim_rules(node, schema, context)
            operand = _lower_expr(node.args[0], schema, context)
            has_var = degree.carries_variable(node.args[0])
            edge = node.kwargs.get('edge')
            wrap = isinstance(edge, EdgeNode)
            fill = None if wrap else _translate_fill(edge, context, has_var=has_var)
            if not wrap and fill is None and not has_var:
                raise LanguageError(_shift_over_data_message(context))
            return plan.Translate(
                operand,
                over_node.name,
                by=sign * int(by_node.value),
                wrap=wrap,
                fill=fill,
            )

        if node.name == 'cumsum':
            over_node = node.kwargs['over']
            if not isinstance(over_node, DimensionNode):
                raise LanguageError(f'{context}: cumsum(over=...) must name a dimension')
            _check_dim_rules(node, schema, context)
            if degree.carries_variable(node.args[0]):
                raise LanguageError(f'{context}: {cumsum_over_variable_message()}')
            return plan.CumulativeSum(_lower_expr(node.args[0], schema, context), over_node.name)

        raise LanguageError(f"{context}: built-in '{node.name}' declares no lowering case")

    assert_never(node)


def _check_dim_rules(node: FunctionCallNode, schema: Model, context: str) -> None:
    """Apply the language's dim rules to a helper call, discarding the dim set.

    Lowering wants the *raise*, not the answer. Called after the plan-shape
    checks so those speak first, and only for one call's dims — the enclosing
    frame is ``dimensions.check_schema``'s business.
    """
    dims_of(node, schema, context)


def _translate_fill(node: ArithmeticNode | None, context: str, *, has_var: bool) -> float | None:
    """The number an ``edge=`` names, or ``None`` for the absence default.

    One kwarg, three policies. ``edge='wrap'`` is cyclic and never reaches here,
    which makes a cyclic call that also asks for a fill unrepresentable rather
    than refused; a number is what the vacated slots contribute; an absent
    ``edge=`` leaves them absent.

    **The right fill is positional**, linopy v1's own reason for refusing to
    pick one (``convention.rst`` §7): 0 is the identity of a sum and 1 of a
    product, so ``x * shift(eff, over=t, by=1, edge=1)`` wants a different
    number from ``lam <= seg + shift(seg, over=bp, by=1, edge=0)``. Over data
    any number is accepted, both lanes filling natively.

    Over an operand carrying a **variable** the only representable fill is 0,
    the vacated slot contributing no term at all. A nonzero one would be a
    constant standing where a term was — a different fragment kind — and is
    refused rather than implemented on one lane and not the other.
    """
    if node is None:
        return None
    sign = 1.0
    if isinstance(node, UnaryOperatorNode) and node.op in ('-', '+'):
        sign, node = (-1.0 if node.op == '-' else 1.0), node.operand
    if not isinstance(node, NumberNode):
        raise LanguageError(f'{context}: {edge_error("shift", "...")}')
    fill = sign * float(node.value)
    if has_var and fill != 0:
        raise LanguageError(
            f'{context}: shift(edge={fill:g}) over an expression containing a variable — only '
            f'fill=0 is representable there, since a vacated slot contributes no term. A nonzero '
            f'fill would be a constant standing where a term was; add that constant to the '
            f'expression instead.'
        )
    return fill


def _shift_over_data_message(context: str) -> str:
    """The three ways out, one of which is two things at once.

    A ``where`` is a *companion* to ``edge=``, not an alternative: the refusal
    is decided on the expression alone so a mask does not lift it, and
    ``edge=0`` alone leaves a row at the vacated coordinate whose bound is that
    zero — the silent pinning this refusal exists to prevent. Either one alone
    is wrong, so the message says so rather than listing them as alternatives.
    """
    return (
        f'{context}: shift() over a variable-free expression leaves vacated positions with no '
        f'value, and inventing one is what silently pinned a bound to zero. Say which you mean:\n'
        f"  shift(x, over=d, by=n, edge='wrap')   the dimension really is cyclic\n"
        f'  shift(x, over=d, by=n, edge=0)        the vacated positions contribute zero\n'
        f'  ...and a where: excluding them        the vacated rows should not exist at all\n'
        f'A where: alone does not lift this — it is decided on the expression, before any mask '
        f'is read — and edge=0 alone leaves a row whose bound is that zero.'
    )


def _bound_expression(value: float | str) -> plan.Expression:
    if isinstance(value, str):
        return plan.Parameter(value)
    return plan.Constant(value)


# ---------------------------------------------------------------------------
# where lowering
# ---------------------------------------------------------------------------


def _lower_where(
    text: str | None, ns: Namespace, context: str, self_variable: str | None = None
) -> plan.Predicate | None:
    """Lower a where string to a plan predicate, ``None`` when there is no mask.

    A predicate that resolves to the constant ``True`` is dropped too: it is
    equivalent to no mask.
    """
    node = where_of(text, ns, context, self_variable)
    if node is None:
        return None
    pred = _lower_where_node(node, context)
    if isinstance(pred, plan.BooleanConstant) and pred.value:
        return None
    return pred


def _lower_where_node(node: WhereNode, context: str) -> plan.Predicate:
    if isinstance(node, BooleanLiteralNode):
        return plan.BooleanConstant(node.value)

    if isinstance(node, ParameterDefinedNode):
        return plan.ParameterDefined(node.name)

    if isinstance(node, VariableDefinedNode):
        return plan.VariableDefined(node.name)

    if isinstance(node, ParameterComparisonNode):
        return plan.ParameterComparison(node.name, node.op, node.value)

    if isinstance(node, DimensionComparisonNode):
        return plan.DimensionComparison(node.name, node.op, node.value)

    if isinstance(node, (UnresolvedNameNode, UnresolvedComparisonNode)):
        msg = (
            f'{type(node).__name__} reached lowering unresolved. Where strings '
            f'must go through resolution.where_of() first.'
        )
        raise AssertionError(msg)

    if isinstance(node, NotNode):
        return plan.Not(_lower_where_node(node.operand, context))
    if isinstance(node, (AndNode, OrNode)):
        node_type = plan.And if isinstance(node, AndNode) else plan.Or
        return node_type(
            _lower_where_node(node.left, context),
            _lower_where_node(node.right, context),
        )

    assert_never(node)


# ---------------------------------------------------------------------------
# check-time advice
# ---------------------------------------------------------------------------


def advice(program: plan.Program) -> list[str]:
    """Modeling advice ``check`` surfaces as warnings — never errors.

    One rule today: **everything under ``dimensions:`` should be an axis** —
    indexed by something, or aggregated into. A declared dimension with neither
    is a label space wearing a dimension's clothes, or dead weight. Advice
    rather than an error because ``extend()`` legitimately splits one model
    across files, and a base file's label space may become a real axis in an
    extension this check never sees.
    """
    axes: set[str] = set()
    for declaration in (*program.parameters, *program.variables, *program.constraints):
        axes.update(declaration.dims)
    expressions = [program.objective.expression]
    expressions.extend(side for c in program.constraints for side in (c.lhs, c.rhs))
    for e in expressions:
        axes |= _produced_axes(e)

    targeted = {c.target: (d.name, c.name) for d in program.dimensions for c in d.coordinates}
    notes: list[str] = []
    for d in program.dimensions:
        if d.name in axes:
            continue
        if d.name in targeted:
            owner, cname = targeted[d.name]
            notes.append(
                f"dimension '{d.name}' is never an axis: nothing is indexed by it and nothing "
                f"aggregates into it — it only serves as the target of coordinate '{cname}' on "
                f"'{owner}'. That is a label space, not a dimension of this model; declare it "
                f'inline instead:\n'
                f'  dimensions:\n'
                f'    {owner}:\n'
                f'      coords:\n'
                f'        {cname}: {{dtype: str}}'
            )
        else:
            notes.append(
                f"dimension '{d.name}' is never used: nothing is indexed by it, nothing "
                f'aggregates into it, and no coordinate targets it. Remove it — or keep it '
                f'knowingly, if an extension file supplies the use.'
            )
    return notes


def _produced_axes(e: plan.Expression) -> set[str]:
    """The axes an expression *creates*, beyond what its declarations index.

    ``group_by=`` lands terms on its target and ``at()`` spreads onto its fine
    dimension, so both are axes even when no declaration is indexed by them —
    an objective may group into a dimension and then implicitly sum it away.
    """
    out: set[str] = set()
    if isinstance(e, plan.GroupSum):
        out.add(e.into)
    if isinstance(e, plan.At):
        out.add(e.over)
    for child in plan.children(e):
        out |= _produced_axes(child)
    return out

"""The logical plan: relational LP construction, one step above SQL.

An intermediate representation in the compiler sense — the module is named
for what it *is* to this engine (duckdb, Calcite and Spark all call this
shape a logical plan) rather than for the generic category.

The lane is described in docs/ARCHITECTURE.md, "The relational lane".

Frozen dataclasses only — no execution logic, no engine imports. A `Program`
is a complete declarative description of a linear program over named tidy
tables; actual data is bound at execution time via a source registry.

Expressions support operator sugar so plans read naturally in Python:

    balance = GroupSum(Variable("p"), over="generator", coordinate="bus", into="bus") - Parameter("load")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, TypeVar

from lpspec.errors import LanguageError, unknown_name_message

if TYPE_CHECKING:
    from collections.abc import Mapping

if TYPE_CHECKING:
    import datetime

ConstraintSense = Literal['==', '<=', '>=']
ObjectiveSense = Literal['min', 'max']
ComparisonOperator = Literal['==', '!=', '<=', '>=', '<', '>']
VariableType = Literal['continuous', 'binary', 'integer']


# --------------------------------------------------------------------------
# Affine expressions
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Expression:
    """Base class for affine expressions over variables and parameters."""

    def __add__(self, other: Expression | float | int) -> Expression:
        return Add(self, _coerce(other))

    def __radd__(self, other: Expression | float | int) -> Expression:
        return Add(_coerce(other), self)

    def __sub__(self, other: Expression | float | int) -> Expression:
        return Add(self, Negate(_coerce(other)))

    def __rsub__(self, other: Expression | float | int) -> Expression:
        return Add(_coerce(other), Negate(self))

    def __mul__(self, other: Expression | float | int) -> Expression:
        return Multiply(self, _coerce(other))

    def __rmul__(self, other: Expression | float | int) -> Expression:
        return Multiply(_coerce(other), self)

    def __truediv__(self, other: Expression | float | int) -> Expression:
        return Divide(self, _coerce(other))

    def __neg__(self) -> Expression:
        return Negate(self)


def _coerce(x: Expression | float | int) -> Expression:
    if isinstance(x, Expression):
        return x
    return Constant(float(x))


@dataclass(frozen=True)
class Constant(Expression):
    """A scalar constant."""

    value: float


@dataclass(frozen=True)
class Parameter(Expression):
    """A parameter reference — contributes to the constant part."""

    name: str


@dataclass(frozen=True)
class Variable(Expression):
    """A variable reference — one term per existing variable row."""

    name: str


@dataclass(frozen=True)
class Negate(Expression):
    operand: Expression


@dataclass(frozen=True)
class Add(Expression):
    left: Expression
    right: Expression


@dataclass(frozen=True)
class Multiply(Expression):
    """Product. At least one factor must be variable-free (affine algebra)."""

    left: Expression
    right: Expression


@dataclass(frozen=True)
class Divide(Expression):
    """Quotient ``numerator / divisor``. The divisor must be variable-free."""

    numerator: Expression
    divisor: Expression


@dataclass(frozen=True)
class Sum(Expression):
    """Sum ``operand`` over the named dims, removing them from the result."""

    operand: Expression
    over: tuple[str, ...]


@dataclass(frozen=True)
class GroupSum(Expression):
    """Sum ``operand`` through a coordinate declared on dim ``over``.

    ``coordinate`` names a coordinate carried by dim ``over`` whose values are
    labels of dim ``into``; the result replaces ``over`` with ``into``. All
    three are resolved before lowering, so the executor needs no schema lookup
    to place the terms.
    """

    operand: Expression
    over: str
    coordinate: str
    into: str


@dataclass(frozen=True)
class At(Expression):
    """Read ``operand`` through a coordinate — the adjoint of :class:`GroupSum`.

    Same mapping table, walked the other way: ``GroupSum`` consumes ``over``
    and produces ``into``, this consumes ``into`` and produces ``over``. The
    fields are named for the *table* rather than the direction, so the pair
    reads as one relation; the surface says which end you stand on
    (``sum(over=)`` consumes, ``at(onto=)`` produces, ``by=`` names the map).

    The join fans out, many ``over`` labels sharing one ``into`` — the fan-out
    ``GroupSum`` pays in reverse, so the locality class is unchanged.
    """

    operand: Expression
    over: str
    coordinate: str
    into: str


@dataclass(frozen=True)
class Translate(Expression):
    """Re-index along one dimension: the result at *t* is ``operand`` at *t - by*.

    One node for the whole of ``shift``, whose ``edge=`` decides ``wrap``:
    ``edge='wrap'`` is periodic (``xarray.roll``), absent or numeric is not.

    ``fill`` decides what an acyclic shift leaves behind. ``None``, what bare
    ``shift`` lowers to, leaves the vacated positions **absent** so they
    propagate and drop the row — linopy v1's ``.shift()``. A number makes them
    present and contribute it, the ``.fillna(0)`` escape hatch spelled in the
    language. Always ``None`` under ``wrap``, a cyclic map vacating nothing.
    """

    operand: Expression
    dimension: str
    by: int
    wrap: bool = True
    fill: float | None = None


def children(expression: Expression) -> tuple[Expression, ...]:
    """The sub-expressions of *expression* — the structural half of any walk.

    Every walk over a plan expression recurses through here and differs only in
    what it does at the leaves. Enumerating the children once is how a node
    added later reaches all of them rather than one.
    """
    if isinstance(expression, Negate):
        return (expression.operand,)
    if isinstance(expression, (Add, Multiply)):
        return (expression.left, expression.right)
    if isinstance(expression, Divide):
        return (expression.numerator, expression.divisor)
    if isinstance(expression, (Sum, GroupSum, At, Translate)):
        return (expression.operand,)
    return ()


# --------------------------------------------------------------------------
# Predicates (where masks — row absence)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Predicate:
    """Base class for where-predicates."""


@dataclass(frozen=True)
class ParameterComparison(Predicate):
    parameter: str
    op: ComparisonOperator
    value: float | str


@dataclass(frozen=True)
class DimensionComparison(Predicate):
    """Compare a *dimension coordinate* to a literal — ``where: "snapshot > 0"``.

    Unlike :class:`ParameterComparison`, no parameter is involved: the dim
    table is already in the frame, so this is a filter on its own column.
    """

    dimension: str
    op: ComparisonOperator
    #: ``datetime`` widens this: a datetime dimension's boundary is a date,
    #: and comparing one to a number reads it as an epoch offset (#460).
    value: float | str | datetime.date


@dataclass(frozen=True)
class ParameterDefined(Predicate):
    """True where the parameter has a non-null, finite value."""

    parameter: str


@dataclass(frozen=True)
class VariableDefined(Predicate):
    """True at the coordinates where the variable exists.

    A semi-join against the variable's own frame — pointwise, and the same
    shape as any mapping-table join.
    """

    variable: str


@dataclass(frozen=True)
class BooleanConstant(Predicate):
    """Constant predicate (``BooleanConstant(False)`` masks out every row)."""

    value: bool


@dataclass(frozen=True)
class And(Predicate):
    left: Predicate
    right: Predicate


@dataclass(frozen=True)
class Or(Predicate):
    left: Predicate
    right: Predicate


@dataclass(frozen=True)
class Not(Predicate):
    operand: Predicate


# --------------------------------------------------------------------------
# Declarations
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DimensionDeclaration:
    """A dimension and the coordinates its labels carry.

    ``coordinates`` maps a coordinate name to the dimension its values are
    labels of, checked for containment once the dim tables exist — which keeps
    a mistyped label from silently dropping its terms in the join that places
    them.

    ``labels`` are the inline label spaces: index columns the dimension owns
    outright, with no target and so nothing to check. They ride the dim table
    for selection and rendering, and resolution refuses to group into one, so
    no expression node reaches them.
    """

    name: str
    coordinates: tuple[tuple[str, str], ...] = ()
    labels: tuple[str, ...] = ()

    @property
    def carried(self) -> list[str]:
        """Every coordinate column the dimension's index source must supply."""
        return sorted([*(c for c, _ in self.coordinates), *self.labels])


@dataclass(frozen=True)
class ParameterDeclaration:
    """Shape declaration; data is bound at execution time by name."""

    name: str
    dims: tuple[str, ...]


@dataclass(frozen=True)
class VariableDeclaration:
    name: str
    dims: tuple[str, ...]
    where: Predicate | None = None
    lower: Expression = field(default_factory=lambda: Constant(float('-inf')))
    upper: Expression = field(default_factory=lambda: Constant(float('inf')))
    variable_type: VariableType = 'continuous'


@dataclass(frozen=True)
class ConstraintDeclaration:
    """``lhs sense rhs`` for each coord combination of ``dims``.

    Both sides are affine; the executor normalises constants to the RHS.
    ``where`` masks out coord combinations (row absence, like variables).
    """

    name: str
    dims: tuple[str, ...]
    lhs: Expression
    sense: ConstraintSense
    rhs: Expression
    where: Predicate | None = None


@dataclass(frozen=True)
class ObjectiveDeclaration:
    """Objective; dims remaining after explicit Sums are implicitly summed."""

    sense: ObjectiveSense
    expression: Expression


_Declaration = TypeVar('_Declaration', ParameterDeclaration, VariableDeclaration, ConstraintDeclaration)


def _declared(items: tuple[_Declaration, ...], name: str, kind: str) -> _Declaration:
    """The declaration called *name*, or a ``KeyError`` naming the near miss."""
    for item in items:
        if item.name == name:
            return item
    raise KeyError(unknown_name_message(kind, name, (i.name for i in items)))


@dataclass(frozen=True)
class Program:
    """A complete linear program over named tidy tables."""

    parameters: tuple[ParameterDeclaration, ...]
    variables: tuple[VariableDeclaration, ...]
    constraints: tuple[ConstraintDeclaration, ...]
    objective: ObjectiveDeclaration
    dimensions: tuple[DimensionDeclaration, ...] = ()

    def dimension(self, name: str) -> DimensionDeclaration:
        """The dimension called *name*.

        Undeclared is not an error here: a dimension with no coordinates has
        nothing to declare.
        """
        for d in self.dimensions:
            if d.name == name:
                return d
        return DimensionDeclaration(name)

    def parameter(self, name: str) -> ParameterDeclaration:
        return _declared(self.parameters, name, 'parameter')

    def variable(self, name: str) -> VariableDeclaration:
        return _declared(self.variables, name, 'variable')

    def constraint(self, name: str) -> ConstraintDeclaration:
        return _declared(self.constraints, name, 'constraint')


def name_dims(program: Program) -> dict[str, tuple[str, ...]]:
    """The dims each declared name is read through.

    Parameters by their ``dims`` and variables by their ``foreach``. A bare
    name in a ``where`` may be either, and the questions below only ask which
    dims it touches. One flat mapping, because the language has one flat
    namespace and the two cannot collide.
    """
    dims: dict[str, tuple[str, ...]] = {p.name: p.dims for p in program.parameters}
    dims.update({v.name: v.dims for v in program.variables})
    return dims


def predicate_dims(where: Predicate, dims: Mapping[str, tuple[str, ...]]) -> frozenset[str]:
    """Which dims *where* reads, against the mapping :func:`name_dims` builds.

    A parameter is read through its own dims, a variable through its foreach,
    a dimension comparison through the dim it names, and a constant reads
    nothing.

    Raises:
        LanguageError: A predicate this function does not know. One that forgot
            to answer here would silently mis-restrict or mislabel a model —
            the polars compiler's semi-join, its label planner and the duckdb
            executor's all read this.
    """
    if isinstance(where, BooleanConstant):
        return frozenset()
    if isinstance(where, DimensionComparison):
        return frozenset({where.dimension})
    if isinstance(where, (ParameterComparison, ParameterDefined)):
        touched = frozenset(dims.get(where.parameter, ()))
        # a parameter compared against another parameter reads both
        value = getattr(where, 'value', None)
        if isinstance(value, str) and value in dims:
            touched |= frozenset(dims[value])
        return touched
    if isinstance(where, VariableDefined):
        # Read through the variable's own foreach, exactly as a parameter is
        # read through its dims. `free_prefix` then keeps the arithmetic path
        # for the leading dims this mask cannot see, as for any other predicate.
        return frozenset(dims.get(where.variable, ()))
    if isinstance(where, (And, Or)):
        return predicate_dims(where.left, dims) | predicate_dims(where.right, dims)
    if isinstance(where, Not):
        return predicate_dims(where.operand, dims)
    raise LanguageError(
        f'{type(where).__name__} is a predicate the mask planner does not know how to read; '
        'add it to predicate_dims before using it in a where'
    )


def free_prefix(dims: tuple[str, ...], touched: frozenset[str]) -> int:
    """How many leading dims a mask does not read.

    Leading, not merely absent: a label follows declaration order, so only a
    prefix leaves the surviving set contiguous under each of its coordinates.
    Returns 0 when the mask reads the first dim — the case that has to count
    its survivors the slow way — and 0 again when *no* dim is read, where the
    split would gain nothing over the one-path arithmetic.

    Static like the rest of this module, and shared for the reason a label is
    shared: the engines must not be able to disagree about which coordinate
    gets which solver index, and the cheapest way to guarantee that is for
    them to ask one function which route to take.
    """
    free = 0
    while free < len(dims) and dims[free] not in touched:
        free += 1
    # every remaining dim is masked-or-not; the split only helps if something is left
    return free if free < len(dims) else 0


def parameters_of(*expressions: Expression) -> frozenset[str]:
    """Every parameter named anywhere under *expressions*."""
    found: set[str] = set()

    def walk(e: Expression) -> None:
        if isinstance(e, Parameter):
            found.add(e.name)
        for child in children(e):
            walk(child)

    for e in expressions:
        walk(e)
    return frozenset(found)


def divisor_parameters(*expressions: Expression) -> frozenset[str]:
    """Parameters appearing anywhere in a divisor position.

    Static, like :func:`parameters_of`: which names *can* reach a divisor is
    the plan's to answer, and *where* they must have values is decided by the
    rows a declaration builds.
    """
    found: set[str] = set()

    def walk(e: Expression) -> None:
        if isinstance(e, Divide):
            found.update(parameters_of(e.divisor))
        for child in children(e):
            walk(child)

    for e in expressions:
        walk(e)
    return frozenset(found)

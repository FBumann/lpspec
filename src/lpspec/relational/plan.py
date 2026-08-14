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
from typing import TYPE_CHECKING, Literal, NamedTuple, TypeVar

from lpspec.errors import unknown_name_message

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
    """Base class for affine expressions over variables and parameters.

    The four operators exist for the tests that compose plans by hand;
    constructing Programs in Python is not supported API, so there is no
    scalar coercion and no reflected form.
    """

    def __add__(self, other: Expression) -> Expression:
        return Add(self, other)

    def __sub__(self, other: Expression) -> Expression:
        return Add(self, Negate(other))

    def __mul__(self, other: Expression) -> Expression:
        return Multiply(self, other)

    def __neg__(self) -> Expression:
        return Negate(self)


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
    three are resolved before lowering, so the engine needs no schema lookup
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


@dataclass(frozen=True)
class CumulativeSum(Expression):
    """Running sum along one dimension, in its declared coordinate order.

    The result at *t* sums ``operand`` over every coordinate up to and
    including *t* — the same order ``shift`` reads positionally.

    Variable-free by construction — lowering refuses an operand carrying a
    variable (``helpers.cumsum_over_variable_message``), so this is always a
    constant part: one ordered scan over an already-materialised table, and no
    coefficient stream is touched. A missing operand row contributes zero, the
    identity of the sum it feeds (SPEC §8), so the result is dense along
    ``dimension``.
    """

    operand: Expression
    dimension: str


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
    if isinstance(expression, (Sum, GroupSum, At, Translate, CumulativeSum)):
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


class CoordinateTarget(NamedTuple):
    """One declared coordinate and the dimension its values are labels of."""

    name: str
    target: str


@dataclass(frozen=True)
class DimensionDeclaration:
    """A dimension and the coordinates its labels carry.

    ``coordinates`` names each coordinate and the dimension its values are
    labels of, checked for containment once the dim tables exist — which keeps
    a mistyped label from silently dropping its terms in the join that places
    them.

    ``labels`` are the inline label spaces: index columns the dimension owns
    outright, with no target and so nothing to check. They ride the dim table
    for selection and rendering, and resolution refuses to group into one, so
    no expression node reaches them.
    """

    name: str
    coordinates: tuple[CoordinateTarget, ...] = ()
    labels: tuple[str, ...] = ()

    @property
    def carried(self) -> list[str]:
        """Every coordinate column the dimension's index source must supply."""
        return sorted([*(c.name for c in self.coordinates), *self.labels])


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

    Both sides are affine; the engine normalises constants to the RHS.
    ``where`` masks out coord combinations (row absence, like variables).
    """

    name: str
    dims: tuple[str, ...]
    lhs: Expression
    sense: ConstraintSense
    rhs: Expression
    where: Predicate | None = None


@dataclass(frozen=True)
class SosDeclaration:
    """One special-ordered set per coordinate of the variable's ``foreach`` minus ``over``.

    The only declaration that adds neither a column nor a row: it names
    columns a sink already has and says what may be nonzero among them. Which
    dims those are is the variable's own ``foreach`` and is read from it: a
    copy here would be a second home for a fact
    (:meth:`Program.variable`).

    ``big_m`` caps the linking coefficient a sink without the concept
    reformulates with, and is ``None`` where the variable's own upper bound is
    the only cap.
    """

    name: str
    variable: str
    over: str
    sos_type: Literal[1, 2]
    big_m: float | None = None


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
    sos: tuple[SosDeclaration, ...] = ()

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

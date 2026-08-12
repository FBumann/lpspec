"""Logical plan → duckdb relations. The duckdb twin of `relational/compiler.py`.

Written to be read **beside** its polars original, not instead of it: the
method names, the argument order and the column conventions are the same, so a
reviewer can put the two files side by side and see what the engine choice
costs. That is the whole point of this module existing — `bench/duckdb-spike.md`
priced the port by counting lines, and a count cannot say whether the result is
readable.

Column conventions, identical to the polars compiler:

===================  ==========================================
relation             columns
===================  ==========================================
dimension table      ``val``, ``ord``, plus declared coordinates
parameter table      ``dims…``, ``value``
variable relation    ``dims…``, ``var_label``
term fragment        ``dims…``, ``var_label``, ``coeff``
const fragment       ``dims…``, ``cval``
===================  ==========================================

**A fragment is a `DuckDBPyRelation` and a value is an `Expression`**, which is
what makes this a twin of the polars compiler rather than a string builder
wearing its method names: `LazyFrame` and `pl.Expr` are the same two objects.
An arithmetic node is composed, so precedence is structural rather than
parenthesised by hand, a literal carries the type the plan holds rather than
the one SQL would read out of its spelling, and a name or an arity that does
not exist raises where it is written instead of when duckdb parses the query
against data.

**An identifier is a value here, never syntax** — the same rule the polars
compiler states, and the one that cost the original engine a `sql.py` module
and an identifier restriction (#189 deleted both). Every name that reaches a
relation goes through :func:`col`. Four constructs have no expression form and
are written out where they are needed, each in a helper that says so:
null-matching joins (:func:`matching`), window functions and ordered
aggregates (both in the executor), and group-by lists.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import duckdb
from duckdb import CoalesceOperator, ConstantExpression, Expression, FunctionExpression, SQLExpression

from lpspec.errors import LanguageError
from lpspec.relational import plan

if TYPE_CHECKING:
    import datetime
    from collections.abc import Callable, Iterable, Mapping, Sequence

#: Scratch column. The spaces make it unrepresentable as a declared name, so it
#: cannot collide with a dimension or coordinate the model already has.
UNIT = '__unit__'

#: What a fragment is. Aliased because `duckdb.DuckDBPyRelation` appears in
#: every signature here and reads as machinery rather than as the thing.
Relation = duckdb.DuckDBPyRelation


def q(name: str) -> str:
    """A name as a SQL identifier. The only way one may reach a query."""
    return '"' + name.replace('"', '""') + '"'


def col(name: str, *, of: str = '') -> Expression:
    """Column *name*, of the relation aliased *of* where one is given.

    `SQLExpression` over a quoted name rather than `ColumnExpression`, which
    parses its argument as a *qualified* name: a dimension called ``a.b`` binds
    as column ``b`` of a table ``a`` that does not exist, and one holding a
    quote raises a parser error. Both are legal model names — a name comes from
    the caller's YAML and no language rule constrains it — so an identifier
    becomes syntax in :func:`q` and nowhere else.
    """
    return SQLExpression(f'{of}.{q(name)}' if of else q(name))


def matching(dims: Sequence[str], left: str = 'l', right: str = 'r') -> Expression:
    """Join condition over *dims*, **matching null to null**.

    ``IS NOT DISTINCT FROM``, which has no expression-API form and so is
    written out. A coordinate may legitimately be null — it says the label
    belongs to no group — and an equi-join would drop the row rather than
    match it.
    """
    return SQLExpression(' AND '.join(f'{left}.{q(d)} IS NOT DISTINCT FROM {right}.{q(d)}' for d in dims))


def falsy_if_null(condition: Expression) -> Expression:
    """A row the mask cannot judge is absent rather than kept.

    SQL three-valued logic makes a null mask neither true nor false. The polars
    side spells this `_falsy_if_null`, and both engines apply it at the same
    two points — the top-level filter and inside a negation — because collapsing
    before the negation and collapsing after it give opposite answers.
    """
    return CoalesceOperator(condition, ConstantExpression(False))


def cross_all(first: Relation, rest: Iterable[Relation]) -> Relation:
    """*first* crossed with each of *rest*, left to right."""
    for other in rest:
        first = first.cross(other)
    return first


def union_all(first: Relation, rest: Iterable[Relation]) -> Relation:
    """*first* stacked with each of *rest*, keeping duplicates.

    `DuckDBPyRelation.union` is ``UNION ALL``; the deduplicating one is
    `union` followed by `distinct`, which is what the presence sets ask for
    and the value fragments must not.
    """
    for other in rest:
        first = first.union(other)
    return first


def _ordinal(dim: str) -> str:
    return f'__ord {dim}__'


@dataclass(frozen=True)
class TermFragment:
    """One additive piece of a compiled affine expression.

    The same shape as the polars compiler's, field for field, because the
    executor above it must not be able to tell which engine filled it.
    """

    dims: tuple[str, ...]
    rel: Relation
    is_term: bool
    keyed: bool = True
    label_dims: frozenset[str] = frozenset()
    presence: Relation | None = None
    presence_dims: tuple[str, ...] | None = None
    variable: str | None = None
    """The variable whose labels this fragment carries; ``None`` for a constant part."""
    mapping: tuple[tuple[str, ...], ...] = ()
    """What has moved a label since it was read, oldest first.

    A ``GroupSum`` records ``('group', over, coordinate, into)`` and a
    ``Translate`` ``('shift', dimension, by, wrap)``. Read only to compare two
    fragments of one variable — :meth:`DuckCompiler.may_share_a_column` — and
    never to build a relation.
    """

    @property
    def value_column(self) -> str:
        return 'coeff' if self.is_term else 'cval'

    @property
    def carried(self) -> list[str]:
        return ['var_label', self.value_column] if self.is_term else [self.value_column]

    def survives_dropping(self, dropped: set[str]) -> bool:
        return self.keyed and dropped <= self.label_dims


@dataclass(frozen=True)
class CompiledExpression:
    """Terms and constant parts, kept apart until assembly."""

    terms: tuple[TermFragment, ...] = ()
    consts: tuple[TermFragment, ...] = ()


@dataclass
class DuckCompiler:
    """Plan → relations, against tables already registered on *con*."""

    con: duckdb.DuckDBPyConnection
    program: plan.Program
    dimensions: Mapping[str, str]
    parameters: Mapping[str, str]
    cardinality: Mapping[str, int]
    boolean_parameters: frozenset[str]
    variables: dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # frames — the masked coordinate product a declaration is instantiated over
    # ------------------------------------------------------------------

    def frame(self, dims: tuple[str, ...], where: plan.Predicate | None) -> Relation:
        """The masked coordinate product over *dims*.

        Carries the labels and the ordinals a caller sorts by, so labels
        follow declaration order.
        """
        out = self._coordinate_product(dims)
        if where is None:
            return out
        out, condition = self._predicate(out, where, dims)
        kept = [c for c in out.columns if not c.startswith('__where')]
        return out.filter(falsy_if_null(condition)).select(*(col(c) for c in kept))

    def unit(self) -> Relation:
        """The one row an empty coordinate product is.

        Not nothing: a `where` on a scalar declaration filters this relation,
        and a filter needs a row to reject.
        """
        return self.con.sql(f'SELECT 0 AS {q(UNIT)}')

    def _coordinate_product(self, dims: tuple[str, ...]) -> Relation:
        """Cross join of the dim tables: labels and ordinals, nothing else."""
        if not dims:
            return self.unit()
        tables = [self.con.table(self.dimensions[d]).set_alias(f'd{i}') for i, d in enumerate(dims)]
        picked = [
            e
            for i, d in enumerate(dims)
            for e in (col('val', of=f'd{i}').alias(d), col('ord', of=f'd{i}').alias(_ordinal(d)))
        ]
        return cross_all(tables[0], tables[1:]).select(*picked)

    def parameter_join(
        self, rel: Relation, param: str, frame_dims: tuple[str, ...], alias: str, subject: str
    ) -> Relation:
        """Left-join *param* onto *rel*, its value column renamed to *alias*."""
        declaration = self.program.parameter(param)
        extra = set(declaration.dims) - set(frame_dims)
        if extra:
            raise LanguageError(f'{subject} has dims {sorted(extra)} outside the foreach dims {list(frame_dims)}')
        left = rel.set_alias('l')
        right = self.con.table(self.parameters[param]).set_alias('r')
        joined = left.cross(right) if not declaration.dims else left.join(right, matching(declaration.dims), how='left')
        return joined.select(*(col(c, of='l') for c in rel.columns), col('value', of='r').alias(alias))

    # ------------------------------------------------------------------
    # predicates (where masks — row absence)
    # ------------------------------------------------------------------

    def _predicate(self, rel: Relation, pred: plan.Predicate, dims: tuple[str, ...]) -> tuple[Relation, Expression]:
        """``(rel with the mask's parameters joined, boolean expression)``."""
        joined: set[str] = set()
        carrier = rel

        def join_param(param: str) -> str:
            nonlocal carrier
            alias = f'__where {param}__'
            if alias not in joined:
                carrier = self.parameter_join(carrier, param, dims, alias, f"where-parameter '{param}'")
                joined.add(alias)
            return alias

        def walk(p: plan.Predicate) -> Expression:
            if isinstance(p, plan.ParameterComparison):
                return _compare(col(join_param(p.parameter)), p.op, p.value)
            if isinstance(p, plan.DimensionComparison):
                if p.dimension not in dims:
                    raise LanguageError(
                        f"where-comparison on dimension '{p.dimension}' is outside the foreach dims "
                        f'{list(dims)} — reducing a mask over an unlisted dim is not supported'
                    )
                return _compare(col(p.dimension), p.op, p.value)
            if isinstance(p, plan.ParameterDefined):
                value = col(join_param(p.parameter))
                if p.parameter in self.boolean_parameters:
                    return value.isnotnull() & value.cast('BOOLEAN')
                return value.isnotnull() & FunctionExpression('isfinite', value)
            if isinstance(p, plan.VariableDefined):
                nonlocal carrier
                flag = f'__where defined {p.variable}__'
                if flag not in joined:
                    carrier = self._mark_defined(carrier, p.variable, flag)
                    joined.add(flag)
                return falsy_if_null(col(flag))
            if isinstance(p, plan.BooleanConstant):
                return ConstantExpression(p.value)
            if isinstance(p, plan.And):
                return walk(p.left) & walk(p.right)
            if isinstance(p, plan.Or):
                return walk(p.left) | walk(p.right)
            if isinstance(p, plan.Not):
                return ~falsy_if_null(walk(p.operand))
            raise LanguageError(f'unsupported predicate node {type(p).__name__}')

        condition = walk(pred)
        return carrier, condition

    def _mark_defined(self, carrier: Relation, variable: str, flag: str) -> Relation:
        """*carrier* with *flag* true where *variable* has a row.

        A left join rather than a semi-join, because the predicate this feeds
        may be negated: a semi-join answers "keep the ones that exist", and
        `not defined(x)` needs the ones that do not to still be here.
        """
        on = list(self.program.variable(variable).dims)
        marked = (
            self.con.table(self.variables[variable])
            .select(*(col(d) for d in on))
            .distinct()
            .select(*(col(d) for d in on), ConstantExpression(True).alias(flag))
            .set_alias('r')
        )
        return (
            carrier.set_alias('l')
            .join(marked, matching(on), how='left')
            .select(*(col(c, of='l') for c in carrier.columns), col(flag, of='r'))
        )

    # ------------------------------------------------------------------
    # bounds
    # ------------------------------------------------------------------

    def bounds(self, rel: Relation, v: plan.VariableDeclaration) -> Relation:
        """*rel* with ``lb``/``ub`` columns for variable *v*.

        Joins and arithmetic are one object, so a bound cannot be evaluated
        against a relation missing what it reads.
        """
        carrier = rel
        joined: set[str] = set()

        def walk(e: plan.Expression) -> Expression:
            nonlocal carrier
            if isinstance(e, plan.Constant):
                return ConstantExpression(float(e.value))
            if isinstance(e, plan.Parameter):
                alias = f'__bound {e.name}__'
                if alias not in joined:
                    carrier = self.parameter_join(
                        carrier, e.name, v.dims, alias, f"bound parameter '{e.name}' of variable '{v.name}'"
                    )
                    joined.add(alias)
                return col(alias).cast('DOUBLE')
            if isinstance(e, plan.Negate):
                return -walk(e.operand)
            if isinstance(e, plan.Add):
                return walk(e.left) + walk(e.right)
            if isinstance(e, plan.Multiply):
                return walk(e.left) * walk(e.right)
            raise LanguageError(
                f"unsupported node {type(e).__name__} in bounds of variable '{v.name}' "
                f'(bounds must be variable-free arithmetic over Constant/Parameter)'
            )

        lower, upper = walk(v.lower), walk(v.upper)
        return carrier.select(*(col(c) for c in rel.columns), lower.alias('lb'), upper.alias('ub'))

    # ------------------------------------------------------------------
    # expressions
    # ------------------------------------------------------------------

    def expression(self, expr: plan.Expression, context: str) -> CompiledExpression:
        if isinstance(expr, plan.Constant):
            return CompiledExpression(consts=(self._constant_fragment(expr.value),))
        if isinstance(expr, plan.Parameter):
            return CompiledExpression(consts=(self._parameter_fragment(expr.name),))
        if isinstance(expr, plan.Variable):
            return CompiledExpression(terms=(self._variable_fragment(expr.name),))
        if isinstance(expr, plan.Negate):
            inner = self.expression(expr.operand, context)
            return CompiledExpression(tuple(_negate(p) for p in inner.terms), tuple(_negate(p) for p in inner.consts))
        if isinstance(expr, plan.Add):
            left, right = self.expression(expr.left, context), self.expression(expr.right, context)
            return CompiledExpression(left.terms + right.terms, left.consts + right.consts)
        if isinstance(expr, plan.Multiply):
            return self._product(self.expression(expr.left, context), self.expression(expr.right, context), context)
        if isinstance(expr, plan.Divide):
            return self._quotient(
                self.expression(expr.numerator, context), self.expression(expr.divisor, context), context
            )
        if isinstance(expr, plan.Sum):
            inner = self.expression(expr.operand, context)
            return _propagate_absence(
                CompiledExpression(
                    tuple(self._sum_fragment(p, expr.over, context) for p in inner.terms),
                    tuple(self._sum_fragment(p, expr.over, context) for p in inner.consts),
                )
            )
        if isinstance(expr, plan.GroupSum):
            inner = self.expression(expr.operand, context)
            return _propagate_absence(
                CompiledExpression(
                    tuple(self._group_fragment(p, expr, context) for p in inner.terms),
                    tuple(self._group_fragment(p, expr, context) for p in inner.consts),
                )
            )
        if isinstance(expr, plan.At):
            inner = self.expression(expr.operand, context)
            return CompiledExpression(
                tuple(self._at_fragment(p, expr, context) for p in inner.terms),
                tuple(self._at_fragment(p, expr, context) for p in inner.consts),
            )
        if isinstance(expr, plan.Translate):
            inner = self.expression(expr.operand, context)
            return CompiledExpression(
                tuple(self._translate_fragment(p, expr, context) for p in inner.terms),
                tuple(self._translate_fragment(p, expr, context) for p in inner.consts),
            )
        raise LanguageError(f'unsupported expression node {type(expr).__name__} in {context}')

    def _constant_fragment(self, value: float) -> TermFragment:
        return TermFragment((), self.unit().select(ConstantExpression(float(value)).alias('cval')), False)

    def _parameter_fragment(self, name: str) -> TermFragment:
        dims = self.program.parameter(name).dims
        table = self.con.table(self.parameters[name])
        return TermFragment(dims, table.select(*(col(d) for d in dims), col('value').alias('cval')), False)

    def _variable_fragment(self, name: str) -> TermFragment:
        dims = self.program.variable(name).dims
        table = self.con.table(self.variables[name])
        rel = table.select(*(col(d) for d in dims), col('var_label'), ConstantExpression(1.0).alias('coeff'))
        masked = self.program.variable(name).where is not None
        presence = table.select(*(col(d) for d in dims or (UNIT,))) if masked else None
        return TermFragment(
            dims,
            rel,
            True,
            label_dims=frozenset(dims),
            presence=presence,
            presence_dims=dims if masked else None,
            variable=name,
        )

    def _product(self, a: CompiledExpression, b: CompiledExpression, context: str) -> CompiledExpression:
        if a.terms and b.terms:
            raise LanguageError(f'nonlinear product in {context}: both factors contain variables')
        if b.terms:
            a, b = b, a
        return CompiledExpression(
            tuple(_join_mul(p, c, is_term=True) for p in a.terms for c in b.consts),
            tuple(_join_mul(p, c, is_term=False) for p in a.consts for c in b.consts),
        )

    def _quotient(self, a: CompiledExpression, b: CompiledExpression, context: str) -> CompiledExpression:
        if b.terms:
            raise LanguageError(f'division by a variable in {context}')
        return CompiledExpression(
            tuple(_join_mul(p, c, is_term=True, divide=True) for p in a.terms for c in b.consts),
            tuple(_join_mul(p, c, is_term=False, divide=True) for p in a.consts for c in b.consts),
        )

    def _sum_fragment(self, p: TermFragment, over: tuple[str, ...], context: str) -> TermFragment:
        """Drop the summed dims — **not** an aggregate."""
        missing = [d for d in over if d not in p.dims]
        if missing and not p.is_term:
            raise LanguageError(
                f'in {context}: Sum over {list(over)} of a constant part lacking dims '
                f'{missing} is ambiguous under masks — multiply explicitly instead'
            )
        keep = tuple(d for d in p.dims if d not in over)
        dropped = {d for d in p.dims if d not in keep}
        scale = math.prod(self.cardinality[d] for d in missing)
        value = col(p.value_column)
        if scale != 1:
            value = value * ConstantExpression(float(scale))
        carried = [col(c) for c in p.carried[:-1]] + [value.alias(p.value_column)]
        return TermFragment(
            keep,
            p.rel.select(*(col(d) for d in keep), *carried),
            p.is_term,
            p.survives_dropping(dropped),
            p.label_dims - dropped,
            variable=p.variable,
            mapping=p.mapping,
        )

    def _group_fragment(self, p: TermFragment, g: plan.GroupSum, context: str) -> TermFragment:
        """Relabel dim ``over`` to ``into`` through a declared coordinate."""
        if g.over not in p.dims:
            raise LanguageError(f"in {context}: GroupSum over '{g.over}' but the expression has dims {list(p.dims)}")
        keep = tuple(x for x in p.dims if x != g.over)
        table = self.con.table(self.dimensions[g.over]).set_alias('r')
        rel = (
            p.rel.set_alias('l')
            .join(table, col(g.over, of='l') == col('val', of='r'))
            .select(
                *(col(d, of='l') for d in keep),
                col(g.coordinate, of='r').alias(g.into),
                *(col(c, of='l') for c in p.carried),
            )
        )
        keyed = p.keyed and g.over in p.label_dims
        return TermFragment(
            (*keep, g.into),
            rel,
            p.is_term,
            keyed,
            _relabel(p.label_dims, g.over, g.into),
            variable=p.variable,
            mapping=(*p.mapping, ('group', g.over, g.coordinate, g.into)),
        )

    def _at_fragment(self, p: TermFragment, a: plan.At, context: str) -> TermFragment:
        """Spread ``into`` back out over ``over`` — the adjoint of a group.

        The same mapping table as `_group_fragment`, joined on the other
        column: grouping reads one row per ``over`` label and lands it on one
        ``into``, and this reads one row per ``into`` and lands it on *every*
        ``over`` sharing it. The join fans out, which is the fan-out a group
        pays in reverse and still one equi-join against a dim table.

        **The key claim has to weaken, and that is the whole difference.** A
        pullback duplicates a ``var_label`` across every fine coordinate of its
        component, so the label no longer spans a dim the frame carries and a
        later reduction can bring two copies into one row. ``keyed=False`` is
        what makes the terminal aggregate run and add them, rather than the
        frame silently holding a cell twice.
        """
        if a.into not in p.dims:
            raise LanguageError(f"in {context}: At through '{a.into}' but the expression has dims {list(p.dims)}")
        keep = tuple(x for x in p.dims if x != a.into)
        table = self.con.table(self.dimensions[a.over]).set_alias('r')
        rel = (
            p.rel.set_alias('l')
            .join(table, col(a.into, of='l') == col(a.coordinate, of='r'))
            .select(
                *(col(d, of='l') for d in keep),
                col('val', of='r').alias(a.over),
                *(col(c, of='l') for c in p.carried),
            )
        )
        return TermFragment(
            (*keep, a.over),
            rel,
            p.is_term,
            keyed=False,
            label_dims=p.label_dims - {a.into},
            variable=p.variable,
        )

    def _translate_fragment(self, p: TermFragment, s: plan.Translate, context: str) -> TermFragment:
        """A pointwise remap of the dim through its ord.

        Two joins on the dim table, never a window: that is what keeps this
        bounded-halo rather than global, and it is the property `#189`'s test
        suite asserts on the polars side (``OVER`` absent from a translation).
        """
        if s.dimension not in p.dims:
            raise LanguageError(
                f"in {context}: translation along '{s.dimension}' but the expression has dims {list(p.dims)}"
            )
        card = self.cardinality[s.dimension]
        others = [d for d in p.dims if d != s.dimension]
        moved = self._moved(s, card)

        def remap(rel: Relation, carried: Sequence[str]) -> Relation:
            table = self.dimensions[s.dimension]
            return (
                rel.set_alias('v')
                .join(self.con.table(table).set_alias('i'), col(s.dimension, of='v') == col('val', of='i'))
                .join(self.con.table(table).set_alias('o'), col('ord', of='o') == moved)
                .select(
                    *(col(d, of='v') for d in others),
                    col('val', of='o').alias(s.dimension),
                    *(col(c, of='v') for c in carried),
                )
            )

        rel = remap(p.rel, p.carried)
        if not s.wrap and s.fill:
            rel = rel.union(self._filled_edge(s, card, others, s.fill))
        presence, presence_dims = None, None
        if p.presence is not None:
            presence = remap(p.presence, ())
            if not s.wrap and s.fill is not None:
                vacated = self._vacated(p, s, card, others).select(*(col(c) for c in presence.columns))
                presence = presence.union(vacated).distinct()
        elif not s.wrap and s.fill is None:
            presence, presence_dims = self._edge(s, card, vacated=False), (s.dimension,)
        return replace(
            p,
            rel=rel,
            presence=presence,
            presence_dims=presence_dims,
            mapping=(*p.mapping, ('shift', s.dimension, str(s.by), str(s.wrap))),
        )

    @staticmethod
    def _moved(s: plan.Translate, card: int) -> Expression:
        """The ordinal a shifted row reads from, wrapped where the shift is cyclic.

        SQL's ``%`` keeps the sign of its left operand, so a negative ``by``
        would land outside the dim table and simply fail to join — dropping the
        row instead of wrapping it. The doubled modulo is not redundant here,
        and is inert on the polars side, where ``%`` already floors.
        """
        shifted = col('ord', of='i') + ConstantExpression(s.by)
        if not s.wrap:
            return shifted
        modulus = ConstantExpression(card)
        return (shifted % modulus + modulus) % modulus

    def _filled_edge(self, s: plan.Translate, card: int, others: Sequence[str], fill: float) -> Relation:
        """``(dims…, cval=fill)`` at every coordinate the shift vacated."""
        edge = self._edge(s, card, vacated=True).set_alias('e')
        tables = [self.con.table(self.dimensions[d]).set_alias(f'o{i}') for i, d in enumerate(others)]
        return cross_all(edge, tables).select(
            *(col('val', of=f'o{i}').alias(d) for i, d in enumerate(others)),
            col(s.dimension, of='e'),
            ConstantExpression(float(fill)).alias('cval'),
        )

    def _edge(self, s: plan.Translate, card: int, *, vacated: bool) -> Relation:
        """The labels of ``s.dimension`` an acyclic shift vacates, or keeps.

        One predicate negated rather than two kept in step — a fill and the
        presence set it implies must not be able to disagree.
        """
        origin = col('ord') - ConstantExpression(s.by)
        outside = (origin < ConstantExpression(0)) | (origin >= ConstantExpression(card))
        table = self.con.table(self.dimensions[s.dimension])
        return table.filter(outside if vacated else ~outside).select(col('val').alias(s.dimension))

    def _vacated(self, p: TermFragment, s: plan.Translate, card: int, others: Sequence[str]) -> Relation:
        """The edge positions ``shift`` leaves with nothing to move in."""
        edge = self._edge(s, card, vacated=True)
        if not others or p.presence is None:
            return edge
        coordinates = p.presence.select(*(col(d) for d in others)).distinct().set_alias('o')
        return coordinates.cross(edge.set_alias('e')).select(
            *(col(d, of='o') for d in others), col(s.dimension, of='e')
        )

    # ------------------------------------------------------------------
    # assembly helpers used by the executor
    # ------------------------------------------------------------------

    def may_share_a_column(self, a: TermFragment, b: TermFragment) -> bool:
        """Whether two fragments of one variable can put a row on one column.

        **Distinct variables never do.** Labels are dense and assigned one
        declaration at a time, so two fragments naming different variables draw
        from disjoint ranges however either was reshaped (#408). What is left is
        whether two fragments of *one* variable send some label to the same
        **row**.

        A label's row is decided by what moved it, so equal
        :attr:`~TermFragment.mapping` means the same row and a certain
        collision. Mappings that differ **only in a coordinate** — the network
        shape, ``sum(f, over=line, group_by=to) - sum(f, over=line, group_by=from)``
        — send it to the same row exactly where those coordinates agree, which
        is a question about a *dimension table*: is there a line whose ends are
        one bus? The `line` table is forty rows where the matrix is 12.6M.

        Anything else is answered **yes**. A shift on one side and not the
        other, a reduction that left them over different dims, a product that
        broadcast one wider — each changes where a label lands in a way this
        does not model, and the cost of being wrong is a silently wrong model
        against the cost of a sort.

        The polars twin, reasoning included: two engines answering this
        differently is two engines aggregating differently, and the frame a
        sink reads has to be the same either way.
        """
        if a.variable is None or b.variable is None:
            return True
        if a.variable != b.variable:
            return False
        if a.dims != b.dims or a.label_dims != b.label_dims:
            return True
        if len(a.mapping) != len(b.mapping):
            return True
        differing = []
        for one, other in zip(a.mapping, b.mapping, strict=True):
            if one == other:
                continue
            kind, *rest = one
            if kind != 'group' or other[0] != 'group' or (rest[0], rest[2]) != (other[1], other[3]):
                return True
            differing.append((rest[0], rest[1], other[2]))
        return all(self._coordinates_meet(over, one, other) for over, one, other in differing)

    def _coordinates_meet(self, dimension: str, one: str, other: str) -> bool:
        """Whether any label of *dimension* carries the same value in both.

        One row is the whole answer, so the scan stops at the first — where the
        polars twin reduces the column. The table is the model's shape rather
        than its size: forty lines against 12.6M nonzeros.
        """
        table = self.con.table(self.dimensions[dimension])
        return table.filter(col(one) == col(other)).limit(1).fetchone() is not None

    @staticmethod
    def constant_scalar(p: TermFragment) -> Relation:
        """The const fragment summed per coordinate: ``(dims…, cval)``.

        The group-by list is spelled rather than composed: `aggregate` takes its
        grouping as SQL, so the names go through :func:`q` on the way in.
        """
        total = FunctionExpression('sum', col('cval')).alias('cval')
        if not p.dims:
            return p.rel.aggregate([total])
        return p.rel.aggregate([*(col(d) for d in p.dims), total], ', '.join(q(d) for d in p.dims))


# --------------------------------------------------------------------------
# free functions — the polars module's, with duckdb underneath
# --------------------------------------------------------------------------


def _relabel(label_dims: frozenset[str], over: str, into: str) -> frozenset[str]:
    return (label_dims - {over}) | {into} if over in label_dims else label_dims


#: The five comparisons whose SQL answer is already the one the mask wants.
#: `!=` is not among them — see :func:`_compare`.
_COMPARISONS: Mapping[str, Callable[[Expression, Expression], Expression]] = {
    '==': lambda a, b: a == b,
    '<=': lambda a, b: a <= b,
    '>=': lambda a, b: a >= b,
    '<': lambda a, b: a < b,
    '>': lambda a, b: a > b,
}


def _compare(column: Expression, op: plan.ComparisonOperator, value: float | str | datetime.date) -> Expression:
    """*column* against *value*, where an absent value is not the one asked for.

    `!=` is the asymmetric one. SQL's is null where the column is, and
    :func:`falsy_if_null` then drops the row — so a coordinate the parameter
    says nothing about would fail a test it plainly passes. ``IS DISTINCT
    FROM`` is the SQL for that, and coalescing to true is the same answer
    composed, since *value* is never null.
    """
    literal = ConstantExpression(value)
    if op == '!=':
        return CoalesceOperator(column != literal, ConstantExpression(True))
    return _COMPARISONS[op](column, literal)


def _negate(p: TermFragment) -> TermFragment:
    others = [col(c) for c in p.rel.columns if c != p.value_column]
    return replace(p, rel=p.rel.select(*others, (-col(p.value_column)).alias(p.value_column)))


def _join_mul(a: TermFragment, c: TermFragment, *, is_term: bool, divide: bool = False) -> TermFragment:
    """Broadcast-join two fragments and multiply (or divide) their values."""
    shared = [d for d in a.dims if d in c.dims]
    dims = (*a.dims, *(d for d in c.dims if d not in a.dims))
    value = 'coeff' if is_term else 'cval'
    left, right = a.rel.set_alias('l'), c.rel.set_alias('r')
    # Left for a divide, so a coordinate the divisor has no value for yields a
    # *null* coefficient instead of silently dropping the term: the question is
    # not "is this divisor dense" but "is it defined where the model divides".
    joined = left.join(right, matching(shared), how='left' if divide else 'inner') if shared else left.cross(right)
    numerator, divisor = col(a.value_column, of='l'), col('cval', of='r')
    rel = joined.select(
        *(col(d, of='l') for d in a.dims),
        *(col(d, of='r') for d in c.dims if d not in a.dims),
        *((col('var_label', of='l'),) if is_term else ()),
        (numerator / divisor if divide else numerator * divisor).alias(value),
    )
    return TermFragment(dims, rel, is_term, a.keyed, a.label_dims, a.presence, a.presence_dims, a.variable, a.mapping)


def restrict_to(rel: Relation, on: Sequence[str], presence: Relation) -> Relation:
    """*rel*, keeping only the rows *presence* has a coordinate for.

    A semi-join is the whole of it: `how='semi'` filters without widening,
    which is what a correlated ``EXISTS`` was standing in for. Empty *on* is a
    scalar's presence — it holds a row or it does not, and the restriction then
    removes every row of *rel* or none, which a semi-join on a constant already
    says without a branch below it.
    """
    keys = presence.select(*(col(d) for d in on)).distinct() if on else presence
    condition = matching(on) if on else SQLExpression('TRUE')
    return rel.set_alias('l').join(keys.set_alias('r'), condition, how='semi')


def _propagate_absence(compiled: CompiledExpression) -> CompiledExpression:
    """Restrict every fragment to where the *whole* expression exists.

    A reduction consumes the expression before any row exists, so without this
    each additive stream would be summed over its own coordinates —
    ``sum(x + size, over=f)`` silently becoming ``sum(x) + sum(size)``, which
    reads an absent ``size`` as zero (SPEC §6, §7).
    """
    restrictions = [
        (p.presence_dims or p.dims, p.presence) for p in (*compiled.terms, *compiled.consts) if p.presence is not None
    ]
    if not restrictions:
        return _map_fragments(compiled, lambda p: replace(p, presence=None, presence_dims=None))

    def restrict(p: TermFragment) -> TermFragment:
        rel = p.rel
        for on, presence in restrictions:
            if all(d in p.dims for d in on):
                rel = restrict_to(rel, on, presence)
        return replace(p, rel=rel, presence=None, presence_dims=None)

    return _map_fragments(compiled, restrict)


def _map_fragments(compiled: CompiledExpression, rewrite: Callable[[TermFragment], TermFragment]) -> CompiledExpression:
    """Apply *rewrite* to every fragment, keeping the term/const split."""
    return CompiledExpression(
        tuple(rewrite(p) for p in compiled.terms),
        tuple(rewrite(p) for p in compiled.consts),
    )

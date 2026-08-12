"""Logical plan → polars. Lazy: nothing is read, nothing is executed.

`lowering.py` compiles the AST to a plan; this compiles the plan to a query, so
docs/ARCHITECTURE.md's admissibility test is a ``.explain()`` away. An identifier is
a value here, never syntax.

Column conventions, relied on by the executor:

===================  ==========================================
frame                columns
===================  ==========================================
dimension table      ``val``, ``ord``, plus declared coordinates
parameter table      ``dims…``, ``value``
variable frame       ``dims…``, ``var_label``
term fragment        ``dims…``, ``var_label``, ``coeff``
const fragment       ``dims…``, ``cval``
===================  ==========================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import polars as pl

from lpspec.errors import LanguageError, LpspecError
from lpspec.relational import plan

if TYPE_CHECKING:
    import datetime
    from collections.abc import Callable, Mapping, Sequence

    from polars._typing import JoinStrategy, MaintainOrderJoin

    from lpspec.relational.binding import BoundSources


#: Scratch columns. The spaces make them unrepresentable as declared names, so
#: they cannot collide with a dimension or coordinate the model already has.
_RHS = '__rhs value__'
_ORD_IN = '__ord in__'
_ORD_OUT = '__ord out__'

#: Carries the single row of the empty coordinate product. Polars cannot hold a
#: frame with one row and no columns — collecting one reports ``(0, 0)`` — so the
#: unit needs a column to exist in, and every path drops it by selecting the
#: dims and the label instead.
UNIT = '__unit__'

#: The same trick for a *scalar* declaration's presence frame. It is distinct
#: from :data:`UNIT` rather than reusing it because the two meet: the empty
#: coordinate product already carries ``UNIT``, and the restriction below cross
#: joins a presence into it.
PRESENT = '__present__'


@dataclass(frozen=True)
class TermFragment:
    """One additive piece of a compiled affine expression.

    Terms yield ``(dims…, var_label, coeff)``, const parts ``(dims…, cval)``.
    An LP row *is* a sum of pieces, so every shape operator rewrites one.
    """

    dims: tuple[str, ...]
    frame: pl.LazyFrame
    is_term: bool

    presence: pl.LazyFrame | None = None
    """Where the *variable* under this fragment exists, keyed by :attr:`dims`.

    Not which rows :attr:`frame` has. A fragment loses rows for two unrelated
    reasons and a constraint row reacts to only one: a **masked variable** is
    genuinely absent, a **sparse parameter** is a compressed dense array whose
    missing rows mean a zero coefficient (SPEC §8). Multiplied together the
    frame cannot tell them apart, so the variable's coordinates ride alongside.

    ``None`` is nothing to report — a constant fragment has no variable, and a
    reduction clears it, ``sum`` skipping absent slots rather than propagating
    them (v1 ``convention.rst`` §13).
    """
    presence_dims: tuple[str, ...] | None = None
    """The columns :attr:`presence` is keyed by; ``None`` means :attr:`dims`.

    Only an acyclic ``shift`` sets it: the coordinates it vacates are an edge
    along *one* dimension, so the restriction is one column wide however many
    dims the fragment carries — where keying it by :attr:`dims` would
    materialise the whole coordinate product to name an edge.
    """

    @property
    def value_column(self) -> str:
        """``coeff`` for a term, ``cval`` for a constant part."""
        return value_column(is_term=self.is_term)

    @property
    def carried(self) -> list[str]:
        """The non-dim columns a projection has to keep."""
        return carried_columns(is_term=self.is_term)


def value_column(*, is_term: bool) -> str:
    """The value column a fragment of this kind carries.

    A free function as well as a :class:`TermFragment` property because
    :func:`_join_mul` names the columns of the fragment it is *building*, whose
    kind need not be either operand's.
    """
    return 'coeff' if is_term else 'cval'


def carried_columns(*, is_term: bool) -> list[str]:
    """The non-dim columns a projection of this fragment kind has to keep."""
    return ['var_label', value_column(is_term=is_term)] if is_term else [value_column(is_term=is_term)]


class _Carrier:
    """A frame a walk joins onto, each attachment made at most once.

    Both walks that read parameters — the mask (:meth:`PolarsCompiler._predicate`)
    and the bounds (:meth:`PolarsCompiler.bounds`) — build an expression over
    columns they are joining on as they go, so the frame and the set of aliases
    already attached travel together rather than as two ``nonlocal``s and an
    ``if alias not in joined`` at every site.
    """

    def __init__(self, frame: pl.LazyFrame) -> None:
        self.frame = frame
        self._attached: set[str] = set()

    def once(self, alias: str, attach: Callable[[pl.LazyFrame, str], pl.LazyFrame]) -> str:
        """Join *attach* onto the frame under *alias*, unless it already is.

        Returns:
            *alias*, so a caller reads the column it just made sure of.
        """
        if alias not in self._attached:
            self.frame = attach(self.frame, alias)
            self._attached.add(alias)
        return alias


@dataclass(frozen=True)
class CompiledExpression:
    """An affine expression as fragments: variable terms and a constant part."""

    terms: tuple[TermFragment, ...]
    consts: tuple[TermFragment, ...]


@dataclass(frozen=True)
class PolarsCompiler:
    """Turn plan nodes into polars queries over the model's tidy frames.

    ``data`` is everything binding produced, frozen. ``variables`` is
    deliberately outside it — the executor's own dict, not a copy, because a
    variable frame appears while its declaration is built and a constraint
    compiled afterwards has to see it.
    """

    program: plan.Program
    data: BoundSources
    variables: Mapping[str, pl.LazyFrame]

    @property
    def dimensions(self) -> Mapping[str, pl.LazyFrame]:
        return self.data.dimensions

    @property
    def parameters(self) -> Mapping[str, pl.LazyFrame]:
        return self.data.parameters

    @property
    def dimension_cardinality(self) -> Mapping[str, int]:
        return self.data.cardinality

    @property
    def boolean_parameters(self) -> frozenset[str]:
        return self.data.boolean_parameters

    # ------------------------------------------------------------------
    # frames — the masked coordinate product a declaration is instantiated over
    # ------------------------------------------------------------------

    @property
    def name_dims(self) -> dict[str, tuple[str, ...]]:
        """The dims each name in a where is read through.

        Parameters by their ``dims`` and variables by their ``foreach``. One
        flat mapping, because the language has one flat namespace and the two
        cannot collide.
        """
        return plan.name_dims(self.program)

    def frame(self, dims: tuple[str, ...], where: plan.Predicate | None) -> pl.LazyFrame:
        """The masked coordinate product over *dims*.

        Labels, plus the ordinals a caller sorts by so labels follow
        declaration order.

        **A mask that has to join restricts by semi-join, not by value join.**
        The predicate reads only its own dims, so it is evaluated over *their*
        product and the full product is semi-joined against the truth set: the
        mask's parameter columns never touch the full product, and a semi-join
        leaves the left side's row order alone where a value join + filter does
        not — which keeps labelling's verify-then-sort a verify.

        Four shapes stay on the direct filter path, which is pointwise and
        keeps order too: a predicate that joins nothing (``_predicate`` hands
        the carrier back unchanged), one reading no frame dim, one reading dims
        outside the frame (so errors name the full frame), and one reading
        **every** frame dim — where the truth set is as wide as the product and
        the semi-join would build it twice to save no width. `sector`'s balance
        mask is that shape, and the semi-join there was a measurable share of
        the whole pipeline (#520).
        """
        out = self._coordinate_product(dims)
        if where is None:
            return out
        carrier, condition = self._predicate(out, where, dims)
        if carrier is out:
            return out.filter(_falsy_if_null(condition))
        touched = plan.predicate_dims(where, self.name_dims)
        on = tuple(d for d in dims if d in touched)
        if on and len(on) < len(dims) and touched <= set(dims):
            keyed, keyed_condition = self._predicate(self._coordinate_product(on), where, on)
            surviving = keyed.filter(_falsy_if_null(keyed_condition)).select(*on)
            return out.join(surviving, on=list(on), how='semi')
        return carrier.filter(_falsy_if_null(condition))

    def _coordinate_product(self, dims: tuple[str, ...]) -> pl.LazyFrame:
        """Cross join of the dim tables: labels and ordinals, nothing else.

        **Folded in reverse, then projected back.** polars' streaming engine
        walks a cross join right-major, so folding backwards makes the product
        arrive in declaration row-major order — label order, which is what lets
        labelling and ``cols`` be read positionally instead of sorted (#433).
        :func:`labels.frame` verifies that rather than trusting it, so the fold
        decides speed, never correctness; the projection restores the column
        order the fold reversed.

        The empty product is one *real* row carrying only :data:`UNIT`: a
        ``where`` on a scalar declaration filters this frame, and nothing
        survives a filter.
        """
        out: pl.LazyFrame | None = None
        for d in reversed(dims):
            table = self.dimensions[d].select(pl.col('val').alias(d), pl.col('ord').alias(ordinal(d)))
            out = table if out is None else out.join(table, how='cross')
        if out is None:
            return pl.LazyFrame({UNIT: [0]})
        return out.select(*(c for d in dims for c in (d, ordinal(d))))

    def parameter_join(
        self,
        frame: pl.LazyFrame,
        param: str,
        frame_dims: tuple[str, ...],
        alias: str,
        subject: str,
        how: JoinStrategy = 'left',
        maintain_order: MaintainOrderJoin | None = None,
    ) -> pl.LazyFrame:
        """Join *param* onto *frame*, its value column renamed to *alias*.

        A parameter carrying a dim the frame lacks would be reduced over it,
        widening a mask or picking an arbitrary bound, so that is refused;
        *subject* is the caller's word for the declaration to name.

        *how* is ``left`` for a bound, where a missing value is a fact to
        report rather than a row to drop. ``inner`` is :meth:`_predicate`'s
        story.

        *maintain_order* is asked for only by the bounds, which become ``cols``
        and are read in order. Asking the join for that order costs an order of
        magnitude less than sorting the same frame afterwards (#433), so it is
        passed deliberately rather than defaulted on; every other consumer
        verifies order where it reads.
        """
        declaration = self.program.parameter(param)
        extra = set(declaration.dims) - set(frame_dims)
        if extra:
            raise LanguageError(f'{subject} has dims {sorted(extra)} outside the foreach dims {list(frame_dims)}')
        table = self.parameters[param].rename({'value': alias})
        if not declaration.dims:
            return frame.join(table, how='cross', maintain_order=maintain_order)
        return frame.join(table, on=list(declaration.dims), how=how, maintain_order=maintain_order)

    # ------------------------------------------------------------------
    # predicates (where masks — row absence)
    # ------------------------------------------------------------------

    def _predicate(
        self, frame: pl.LazyFrame, pred: plan.Predicate, dims: tuple[str, ...]
    ) -> tuple[pl.LazyFrame, pl.Expr]:
        """``(frame with the mask's parameters joined, boolean expression)``.

        Walking joins the parameters, so the condition is built first and the
        frame read after — one expression would return the pre-walk frame.

        **A name the mask is certain of is joined rather than left-joined**,
        and a certain variable is semi-joined and never read
        (:func:`_certain_parameters`). An atom over a missing value reads as
        false either way, so the strategies differ only in *where* the row is
        dropped, and the inner join saves the width of the product it is
        dropped from — a few percent of a pipeline on the direct path, and
        nothing measurable behind a semi-join's key product (#520).

        ``VariableDefined`` is the one atom answered by a join rather than a
        column test — existence lives in the variable's own frame — keyed by
        dims the dim rule has already checked are inside this frame.

        No join here maintains order: consumers verify where they read
        (:func:`labels.in_position_order`), so a shuffle costs a sort
        downstream at worst, never a wrong label.
        """
        certain = _certain_parameters(pred)
        carrier = _Carrier(frame)

        def join_param(param: str) -> str:
            how: JoinStrategy = 'inner' if param in certain else 'left'
            return carrier.once(
                f'__where {param}__',
                lambda f, alias: self.parameter_join(f, param, dims, alias, f"where-parameter '{param}'", how),
            )

        def walk(p: plan.Predicate) -> pl.Expr:
            if isinstance(p, plan.ParameterComparison):
                return _compare(pl.col(join_param(p.parameter)), p.op, p.value)
            if isinstance(p, plan.DimensionComparison):
                if p.dimension not in dims:
                    raise LanguageError(
                        f"where-comparison on dimension '{p.dimension}' is outside the foreach dims "
                        f'{list(dims)} — reducing a mask over an unlisted dim is not supported'
                    )
                return _compare(_dimension_column(p.dimension, p.value), p.op, p.value)
            if isinstance(p, plan.ParameterDefined):
                col = pl.col(join_param(p.parameter))
                if p.parameter in self.boolean_parameters:
                    return col.is_not_null() & col.cast(pl.Boolean)
                return col.is_not_null() & col.is_finite()
            if isinstance(p, plan.VariableDefined):
                on = list(self.program.variable(p.variable).dims)
                coordinates = self.variables[p.variable].select(*on)
                if p.variable in certain:
                    carrier.once(f'__where defined {p.variable}__', lambda f, _: f.join(coordinates, on=on, how='semi'))
                    return pl.lit(value=True)
                flag = carrier.once(
                    f'__where defined {p.variable}__',
                    lambda f, alias: f.join(
                        coordinates.unique().with_columns(pl.lit(value=True).alias(alias)), on=on, how='left'
                    ),
                )
                return _falsy_if_null(pl.col(flag))
            if isinstance(p, plan.BooleanConstant):
                return pl.lit(value=p.value)
            if isinstance(p, plan.And):
                return walk(p.left) & walk(p.right)
            if isinstance(p, plan.Or):
                return walk(p.left) | walk(p.right)
            if isinstance(p, plan.Not):
                return ~_falsy_if_null(walk(p.operand))
            raise LanguageError(f'unsupported predicate node {type(p).__name__}')

        condition = walk(pred)
        return carrier.frame, condition

    # ------------------------------------------------------------------
    # bounds
    # ------------------------------------------------------------------

    def bounds(self, frame: pl.LazyFrame, v: plan.VariableDeclaration) -> pl.LazyFrame:
        """*frame* with ``lb``/``ub`` columns for variable *v*.

        Joins and arithmetic are one object, so a bound cannot be evaluated
        against a frame missing what it reads.
        """
        carrier = _Carrier(frame)

        def attach_bound(f: pl.LazyFrame, alias: str, name: str) -> pl.LazyFrame:
            aligned = self._aligned_bound(f, name, v, alias)
            if aligned is not None:
                return aligned
            subject = f"bound parameter '{name}' of variable '{v.name}'"
            return self.parameter_join(f, name, v.dims, alias, subject, maintain_order='left')

        def walk(e: plan.Expression) -> pl.Expr:
            if isinstance(e, plan.Constant):
                return pl.lit(float(e.value), dtype=pl.Float64)
            if isinstance(e, plan.Parameter):
                alias = carrier.once(f'__bound {e.name}__', lambda f, a: attach_bound(f, a, e.name))
                return pl.col(alias).cast(pl.Float64)
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
        return carrier.frame.with_columns(lower.alias('lb'), upper.alias('ub'))

    def _aligned_bound(
        self, frame: pl.LazyFrame, param: str, v: plan.VariableDeclaration, alias: str
    ) -> pl.LazyFrame | None:
        """*frame* with *param* attached **by position**, or ``None`` to join.

        A bound dense over the whole variable product — a profile per node per
        technology per hour, the ordinary shape in energy modelling — costs a
        full-size join against a full-size coordinate product here, where the
        eager lane gets it free from array position — on `profiled` that join
        was most of the build (#511).

        Position means the coordinate here too. The label frame is row-major
        over the product in each dimension's own index order (*not*
        lexicographic dim order, which is why sorting by the dim columns does
        not match), so sorting the parameter by those ordinals reproduces it
        exactly and the value column is attached rather than joined.

        **Wrong bounds are a wrong model with no error**, so this is refused
        unless all three hold, each a fact already computed:

        * the parameter's dims are exactly the variable's, in the same order —
          fewer broadcast, more is already refused, a different order is a
          different row-major walk
        * the variable declares no ``where`` — a mask makes the label frame a
          subset of the product and position stops lining up
        * the parameter is dense over that product, its height equal to the
          product of the cardinalities binding cached

        Duplicate coordinates would break density without changing the height,
        and are refused before this by ``check_one_row_per_coordinate``.
        """
        declaration = self.program.parameter(param)
        if v.where is not None or tuple(declaration.dims) != tuple(v.dims) or not v.dims:
            return None

        cards = [self.dimension_cardinality[d] for d in v.dims]
        expected = math.prod(cards)
        table = self.parameters[param]
        if table.select(pl.len()).collect().item() != expected:
            return None

        stride = 1
        strides: list[int] = []
        for card in reversed(cards):
            strides.insert(0, stride)
            stride *= card
        position = sum(
            (self._ordinal_of(d) * step for d, step in zip(v.dims, strides, strict=True)),
            start=pl.lit(0, dtype=pl.Int64),
        )
        pairs = table.select(position.alias('__at__'), pl.col('value')).collect(engine='streaming')
        return frame.with_columns(pl.Series(alias, _scattered(pairs['__at__'], pairs['value'], expected)))

    def _ordinal_of(self, dim: str) -> pl.Expr:
        """A parameter's *dim* column as that dimension's ordinal.

        **Free for a string dimension**, which is most of them: binding encodes
        those as an ``Enum`` over the labels in ordinal order
        (``_Binder.encode_dimensions``), so the physical code already *is* the
        ordinal. Every other dtype pays a dictionary built from the dimension
        table — one entry per label, not per row.
        """
        column = pl.col(dim)
        if self.data.is_enum_encoded(dim):
            return column.to_physical().cast(pl.Int64)
        labels = self.dimensions[dim].select('val').collect()['val']
        return column.replace_strict({value: at for at, value in enumerate(labels)}, return_dtype=pl.Int64)

    # ------------------------------------------------------------------
    # expressions → fragments
    # ------------------------------------------------------------------

    def expression(self, expr: plan.Expression, context: str) -> CompiledExpression:
        """Compile an affine expression into term and const fragments.

        No join in the walk maintains order, on evidence rather than by
        omission: the mul join's ``maintain_order`` holds the label order on
        some shapes and loses it on others differing only in data, so the tax
        lands unpredictably and `profiled/l`'s objective phase triples paying
        it for nothing. Every consumer re-derives or verifies order where it
        reads (:meth:`PolarsExecutor._build_objective`'s docstring keeps the
        numbers).
        """

        def product(a: CompiledExpression, b: CompiledExpression) -> CompiledExpression:
            """``a * b``, with the variable-carrying side normalised to the left."""
            if a.terms and b.terms:
                raise LanguageError(f'nonlinear product in {context}: both factors contain variables')
            if b.terms:
                a, b = b, a
            terms = tuple(_join_mul(t, c, is_term=True) for t in a.terms for c in b.consts)
            consts = tuple(_join_mul(x, c, is_term=False) for x in a.consts for c in b.consts)
            return CompiledExpression(terms, consts)

        def quotient(a: CompiledExpression, b: CompiledExpression) -> CompiledExpression:
            """``a / b``, where *b* must be a single variable-free factor."""
            if b.terms:
                raise LanguageError(f'nonlinear quotient in {context}: the divisor contains variables')
            if len(b.consts) != 1:
                raise LanguageError(
                    f'in {context}: a divisor must be a single Constant/Parameter factor, '
                    f'not a sum — rewrite as multiplication by a precomputed parameter'
                )
            inv = b.consts[0]
            terms = tuple(_join_mul(t, inv, is_term=True, divide=True) for t in a.terms)
            consts = tuple(_join_mul(x, inv, is_term=False, divide=True) for x in a.consts)
            return CompiledExpression(terms, consts)

        def ev(e: plan.Expression) -> CompiledExpression:
            if isinstance(e, plan.Constant):
                frame = pl.LazyFrame({'cval': [float(e.value)]}, schema={'cval': pl.Float64})
                return CompiledExpression((), (TermFragment((), frame, False),))
            if isinstance(e, plan.Parameter):
                return CompiledExpression((), (self._parameter_fragment(e.name),))
            if isinstance(e, plan.Variable):
                return CompiledExpression((self._variable_fragment(e.name),), ())
            if isinstance(e, plan.Negate):
                return _map_fragments(ev(e.operand), _negate)
            if isinstance(e, plan.Add):
                a, b = ev(e.left), ev(e.right)
                return CompiledExpression(a.terms + b.terms, a.consts + b.consts)
            if isinstance(e, plan.Multiply):
                return product(ev(e.left), ev(e.right))
            if isinstance(e, plan.Divide):
                return quotient(ev(e.numerator), ev(e.divisor))
            if isinstance(e, plan.Sum):
                inner = _propagate_absence(ev(e.operand))
                return _map_fragments(inner, lambda p: self._sum_fragment(p, e.over, context))
            if isinstance(e, plan.GroupSum):
                inner = _propagate_absence(ev(e.operand))
                return _map_fragments(inner, lambda p: self._group_fragment(p, e, context))
            if isinstance(e, plan.At):
                return _map_fragments(ev(e.operand), lambda p: self._at_fragment(p, e, context))
            if isinstance(e, plan.Translate):
                return _map_fragments(ev(e.operand), lambda p: self._translate_fragment(p, e, context))
            raise LanguageError(f'unsupported expression node {type(e).__name__} in {context}')

        return ev(expr)

    def _parameter_fragment(self, name: str) -> TermFragment:
        """A parameter as a constant part, keyed by its declared dims.

        One row per coordinate, which the executor enforces by refusing a
        duplicated one.
        """
        dims = self.program.parameter(name).dims
        frame = self.parameters[name].select(*dims, pl.col('value').cast(pl.Float64).alias('cval'))
        return TermFragment(dims, frame, False)

    def _variable_fragment(self, name: str) -> TermFragment:
        """A variable as a term with unit coefficients.

        Presence is attached only for a masked variable, decided off the
        declaration before any data is read: an unmasked one exists at every
        coordinate of its foreach and could restrict nothing.

        ``presence_dims`` is stated rather than left to its ``None`` default,
        because dims are rewritten downstream (a product broadcasts, a sum
        drops) while the presence frame is not — an implied key silently
        becomes a claim about columns it never had.
        """
        dims = self.program.variable(name).dims
        frame = self.variables[name].select(*dims, 'var_label', pl.lit(1.0, dtype=pl.Float64).alias('coeff'))
        masked = self.program.variable(name).where is not None
        presence = self._presence(name, dims) if masked else None
        return TermFragment(dims, frame, True, presence=presence, presence_dims=dims if masked else None)

    def _presence(self, name: str, dims: tuple[str, ...]) -> pl.LazyFrame:
        """The coordinates a masked variable exists at.

        A **scalar** declaration has none, and ``select()`` over no dims is the
        empty frame polars cannot represent, so the marker column carries the
        one bit left: whether the row is there at all. It is renamed from a
        real column, never a ``pl.lit()`` — a select of literals alone is
        length 1 whatever it selects from, so an absent scalar would come back
        present.
        """
        frame = self.variables[name]
        if dims:
            return frame.select(*dims)
        return frame.select(pl.col('var_label').alias(PRESENT))

    # ------------------------------------------------------------------
    # shape operators — one dim rewritten per fragment
    # ------------------------------------------------------------------

    def _sum_fragment(self, p: TermFragment, over: tuple[str, ...], context: str) -> TermFragment:
        """Drop the summed dims — **not an aggregate**.

        The rows that carried them stay and collapse in the terminal
        ``sum(coeff)`` at assembly. Constructed rather than ``replace``d so
        ``presence`` is *dropped*: §13 reads a reduction as skipping absent
        slots, so summing over a partly-masked dim reports nothing.
        """
        missing = [d for d in over if d not in p.dims]
        if missing and not p.is_term:
            raise LanguageError(
                f'in {context}: Sum over {list(over)} of a constant part lacking dims '
                f'{missing} is ambiguous under masks — multiply explicitly instead'
            )
        keep = tuple(d for d in p.dims if d not in over)
        scale = math.prod(self.dimension_cardinality[d] for d in missing)
        frame = p.frame.select(*keep, *p.carried)
        if scale != 1:
            frame = frame.with_columns(pl.col(p.value_column) * scale)
        return TermFragment(keep, frame, p.is_term)

    def _group_fragment(self, p: TermFragment, g: plan.GroupSum, context: str) -> TermFragment:
        """Relabel dim ``over`` to ``into`` through a declared coordinate.

        No aggregate either: the dim table holds one row per label and its
        coordinate was checked for containment at build time, so the join
        neither duplicates nor drops a term, and rows landing on one ``into``
        are added by the terminal aggregate as ``Sum``'s are. A group is a sum,
        so §13 applies and this constructs rather than ``replace``s — see
        :meth:`_sum_fragment`.
        """
        if g.over not in p.dims:
            raise LanguageError(f"in {context}: GroupSum over '{g.over}' but the expression has dims {list(p.dims)}")
        return self._remap_fragment(p, g, consumed=g.over, produced=g.into)

    def _at_fragment(self, p: TermFragment, a: plan.At, context: str) -> TermFragment:
        """Spread ``into`` back out over ``over`` — the adjoint of a group.

        The same mapping table as :meth:`_group_fragment`, joined on the other
        column, so the join **fans out**: one row per ``into`` lands on every
        ``over`` sharing it. Still one equi-join against a table the frame
        holds, so the locality class does not move.

        A pullback duplicates a label — the same ``var_label`` at every fine
        coordinate of its component — so a later reduction can bring two copies
        into one row, where the terminal aggregate adds them.
        """
        if a.into not in p.dims:
            raise LanguageError(f"in {context}: At through '{a.into}' but the expression has dims {list(p.dims)}")
        return self._remap_fragment(p, a, consumed=a.into, produced=a.over)

    def _remap_fragment(
        self, p: TermFragment, node: plan.GroupSum | plan.At, *, consumed: str, produced: str
    ) -> TermFragment:
        """Trade dim *consumed* for *produced* through *node*'s coordinate.

        The mapping table is the declared coordinate read as two columns —
        ``val`` as ``over``, the coordinate as ``into`` — and the rewrite is a
        single inner equi-join on *consumed*. A group consumes ``over``
        (:meth:`_group_fragment`); an ``At`` reads the same table backwards
        (:meth:`_at_fragment`). Written once so the adjoints cannot drift: a
        change to how the mapping joins is a change to both.
        """
        keep = tuple(x for x in p.dims if x != consumed)
        mapping = self.dimensions[node.over].select(
            pl.col('val').alias(node.over), pl.col(node.coordinate).alias(node.into)
        )
        frame = p.frame.join(mapping, on=consumed, how='inner').select(*keep, produced, *p.carried)
        return TermFragment((*keep, produced), frame, p.is_term)

    def _translate_fragment(self, p: TermFragment, s: plan.Translate, context: str) -> TermFragment:
        """A pointwise remap of the dim through its ord.

        A row at *o* contributes at ``(o + by) % card``.

        Both joins are on a dim-table key, so the row count is unchanged and an
        out-of-range ordinal does not join. No window function; bounded-halo
        locality. The operand's *presence* is :func:`travelled_presence` below.

        Every fill over a *constant* is written, ``0`` included (#551): the
        arithmetic is unchanged, but the slot now has a value, so asking for
        zero stops being indistinguishable from having nothing. Over a *term*
        there is nothing to write — ``edge=0`` on a variable means the vacated
        slot contributes no term at all (SPEC §7), where a zero-coefficient
        entry would be a matrix nonzero standing for a term that is not there.
        Lowering refuses every other numeric edge over a variable.
        """
        if s.dimension not in p.dims:
            raise LanguageError(
                f"in {context}: translation along '{s.dimension}' but the expression has dims {list(p.dims)}"
            )
        card = self.dimension_cardinality[s.dimension]
        others = [d for d in p.dims if d != s.dimension]
        table = self.dimensions[s.dimension]
        incoming = table.select(pl.col('val').alias(s.dimension), pl.col('ord').alias(_ORD_IN))
        outgoing = table.select(pl.col('val').alias(s.dimension), pl.col('ord').alias(_ORD_OUT))

        moved = pl.col(_ORD_IN) + s.by
        if s.wrap:
            moved = (moved % card + card) % card

        def remap(source: pl.LazyFrame, carried: list[str], source_dims: tuple[str, ...] | None = None) -> pl.LazyFrame:
            """*source*, with ``s.dimension`` moved by ``s.by``.

            ``source_dims`` is the caller's, because a presence frame need not
            carry the fragment's: an acyclic shift's presence speaks only about
            the dim it vacated, so projecting the fragment's dims onto it asks
            for columns it never had.
            """
            kept = [d for d in (source_dims if source_dims is not None else p.dims) if d != s.dimension]
            return (
                source.join(incoming, on=s.dimension, how='inner')
                .drop(s.dimension)
                .with_columns(moved.alias(_ORD_OUT))
                .join(outgoing, on=_ORD_OUT, how='inner')
                .select(*kept, s.dimension, *carried)
            )

        def travelled_presence() -> tuple[pl.LazyFrame | None, tuple[str, ...] | None]:
            """Where the variable exists after the shift, and what keys it.

            An existing presence **travels**: the coordinate set goes through
            the same map the rows did, and the inner join drops whatever the
            edge vacated. Under a fill the vacated positions go back in
            (:meth:`_vacated`) — a filled slot counts as present. A narrow
            presence is widened first when the shift moves a dim it is silent
            about, since there is no column to remap otherwise and joining the
            row set on a column it never had is #546; :meth:`_widen` changes no
            answer, and what comes back is full-width.

            An operand with **no** presence gets one: nothing was absent before
            and the acyclic edge now is, where without this the vacated slot
            would merely fail to join and the row would survive with its term
            quietly gone (#239, #289). It is keyed by the one dimension it
            speaks about, which is what :attr:`TermFragment.presence_dims` is
            for — keying it by the fragment's dims would materialise the whole
            coordinate product to name an edge, which costs a fifth again of
            build on a wide ramp (#520), a shape no case in `bench/` covers.
            """
            if p.presence is None:
                if s.wrap or s.fill is not None:
                    return None, None
                return self._edge(s, card, vacated=False), (s.dimension,)

            source, keyed_by = p.presence, p.presence_dims
            if keyed_by is not None and s.dimension not in keyed_by:
                source, keyed_by = self._widen(source, keyed_by, p.dims), None
            moved_presence = remap(source, [], keyed_by)
            if s.wrap or s.fill is None:
                return moved_presence, keyed_by
            vacated = self._vacated(p, s, card, others)
            return pl.concat([moved_presence, vacated], how='vertical_relaxed').unique(), None

        frame = remap(p.frame, p.carried)
        if not s.wrap and s.fill is not None and not p.is_term:
            frame = pl.concat([frame, self._filled_edge(s, card, others, s.fill)], how='vertical_relaxed')
        presence, presence_dims = travelled_presence()
        return replace(p, frame=frame, presence=presence, presence_dims=presence_dims)

    def _widen(self, presence: pl.LazyFrame, have: tuple[str, ...], want: tuple[str, ...]) -> pl.LazyFrame:
        """*presence* over every dim in *want*, saying the same thing.

        A presence frame is silent about the dims it omits, which reads as
        "present at all of them" — so the widening is a cross join with those
        dimensions' own tables, and it changes no answer.
        """
        for d in want:
            if d not in have:
                presence = presence.join(self.dimensions[d].select(pl.col('val').alias(d)), how='cross')
        return presence.select(*want)

    def _filled_edge(self, s: plan.Translate, card: int, others: list[str], fill: float) -> pl.LazyFrame:
        """``(dims…, cval=fill)`` at every coordinate the shift vacated.

        Dense over *others*, not over the rows the operand happened to carry:
        the eager lane shifts an array already reindexed to the master
        coordinates, so a fill appearing only where the parameter was
        non-sparse would be a second answer to the same question.

        Only a *truthy* fill gets here, ``fill=0`` needing no rows at all. A
        nonzero fill reaches a translation only over a variable-free operand,
        so this is always the const branch and never invents a ``var_label``.
        """
        edge = self._edge(s, card, vacated=True)
        for d in others:
            edge = edge.join(self.dimensions[d].select(pl.col('val').alias(d)), how='cross')
        return edge.with_columns(pl.lit(fill, dtype=pl.Float64).alias('cval')).select(*others, s.dimension, 'cval')

    def _edge(self, s: plan.Translate, card: int, *, vacated: bool) -> pl.LazyFrame:
        """The labels of ``s.dimension`` an acyclic shift vacates, or keeps.

        Exact complements, so one filter negated rather than two conditions to
        keep in step: a fill and the presence set it implies must not disagree
        about which coordinates the edge is. One column wide either way, an
        edge being vacated for *every* combination of the other dims.
        """
        source = pl.col('ord') - s.by
        outside = (source < 0) | (source >= card)
        return (
            self.dimensions[s.dimension]
            .filter(outside if vacated else ~outside)
            .select(pl.col('val').alias(s.dimension))
        )

    def _vacated(self, p: TermFragment, s: plan.Translate, card: int, others: list[str]) -> pl.LazyFrame:
        """The edge positions ``shift`` leaves with nothing to move in.

        Reached only under ``fill=0``, which is the whole of what ``fill`` does
        here: back in the presence set they are present-with-no-term, a zero
        contribution and a surviving row. Left out, absence propagates and the
        row drops — linopy v1's reading of ``.shift()``.

        Only the ``shift`` edge qualifies; a coordinate the variable's own mask
        removed is genuinely absent and remapping already dropped it. So the
        edge is crossed with the other-dim combinations the variable actually
        has, one vacated row each.
        """
        edge = self._edge(s, card, vacated=True)
        if not others:
            return edge
        return p.presence.select(*others).unique().join(edge, how='cross') if p.presence is not None else edge

    # ------------------------------------------------------------------
    # assembly helpers used by the executor
    # ------------------------------------------------------------------

    @staticmethod
    def constant_scalar(p: TermFragment) -> pl.LazyFrame:
        """The const fragment summed per coordinate: ``(dims…, cval)``."""
        if not p.dims:
            return p.frame.select(pl.col('cval').sum())
        return p.frame.group_by(p.dims).agg(pl.col('cval').sum())


def ordinal(dim: str) -> str:
    """The frame column carrying *dim*'s position in its declared order."""
    return f'__ord {dim}__'


def _certain_parameters(pred: plan.Predicate) -> frozenset[str]:
    """Names whose absence alone makes the whole mask false.

    A row those names have no value for is one the filter would drop anyway, so
    the join may drop it first. Only ``And`` descends — under ``Or`` or ``Not``
    an absent value can still leave the mask true, and dropping the row there
    is a wrong model rather than a slow one, which is why the fallthrough
    answers nothing rather than raising on a node it does not know.
    """
    if isinstance(pred, plan.And):
        return _certain_parameters(pred.left) | _certain_parameters(pred.right)
    if isinstance(pred, (plan.ParameterComparison, plan.ParameterDefined)):
        return frozenset({pred.parameter})
    if isinstance(pred, plan.VariableDefined):
        return frozenset({pred.variable})
    return frozenset()


def _falsy_if_null(condition: pl.Expr) -> pl.Expr:
    """*condition* with null read as false.

    A missing parameter row must exclude the coordinate rather than
    propagate. Masks are row absence.
    """
    return condition.fill_null(value=False)


def _dimension_column(dimension: str, value: float | str | datetime.date) -> pl.Expr:
    """The column a where-comparison on *dimension* reads.

    A string label is compared in ``String`` space, undoing binding's ``Enum``:
    §6.1 orders labels bytewise and reads an unknown label as matching nothing,
    where an ``Enum`` orders by declaration and refuses strangers.
    """
    column = pl.col(dimension)
    return column.cast(pl.String) if isinstance(value, str) else column


def _compare(column: pl.Expr, op: plan.ComparisonOperator, value: float | str | datetime.date) -> pl.Expr:
    """One where-comparison. A string, a float and a date are all literals here."""
    literal = pl.lit(value)
    match op:
        case '==':
            return column == literal
        case '!=':
            return column != literal
        case '<':
            return column < literal
        case '<=':
            return column <= literal
        case '>':
            return column > literal
        case '>=':
            return column >= literal


def restrict_by_presence(frame: pl.LazyFrame, presence: pl.LazyFrame, on: Sequence[str]) -> pl.LazyFrame:
    """Keep only the rows of *frame* that *presence* admits.

    Keyed by *on* this is a semi-join. A **scalar** declaration has no key, its
    presence being at most one row saying only whether it exists, so the
    question becomes a cross join: every row survives a present scalar and none
    survives an absent one — absence spreading through arithmetic (SPEC §7) at
    no dimension.
    """
    if on:
        return frame.join(presence.select(list(on)), on=list(on), how='semi')
    return frame.join(presence.select(PRESENT), how='cross').drop(PRESENT)


def _propagate_absence(compiled: CompiledExpression) -> CompiledExpression:
    """Restrict every fragment to where the *whole* expression exists.

    Addition is fragment concatenation, so ``x + size`` is two independent
    streams — right at row level, where the executor intersects the presences,
    but a **reduction** consumes the expression before any row exists. That is
    the difference between ``sum(x + size, over=f)``, which sums where the
    summand exists, and ``sum(x, over=f) + sum(size, over=f)``, which sums each
    operand over its own domain and reads the absent ``size`` as a zero (SPEC
    §6, §7).

    Applied only where the key columns are dims the fragment carries: a
    restriction naming a dim a fragment lacks cannot speak about it.

    **A fragment is never restricted by its own presence.** Its rows and its
    presence are built from one frame and rewritten in step — a product joins
    the rows and leaves the coordinates, a translation remaps both, a fill adds
    to both — so the rows are inside the coordinates by construction and the
    join could only return them all. Under a mask over a single term, the
    ordinary case, that made the pass a semi-join of a frame against itself,
    which is measurable on any model large enough to care (#413).

    The presence frame is not deduplicated first: a semi-join asks whether a
    key occurs, and occurring twice is still occurring, so the distinct changes
    no row and costs a hash pass over every coordinate the variable has.
    """
    absent = [p for p in (*compiled.terms, *compiled.consts) if p.presence is not None]
    if not absent:
        return compiled

    def restrict(p: TermFragment) -> TermFragment:
        frame = p.frame
        for source in absent:
            if source is p or source.presence is None:
                continue
            on = list(source.presence_dims or source.dims)
            if all(d in p.dims for d in on):
                frame = restrict_by_presence(frame, source.presence, on)
        return p if frame is p.frame else replace(p, frame=frame)

    return _map_fragments(compiled, restrict)


def _map_fragments(
    compiled: CompiledExpression,
    rewrite: Callable[[TermFragment], TermFragment],
) -> CompiledExpression:
    """Apply *rewrite* to every fragment, keeping the term/const split.

    Rewriting one fragment at a time is what pointwise and bounded-halo
    locality mean; a node needing them together is global, and rejected at
    lowering.
    """
    return CompiledExpression(
        tuple(rewrite(p) for p in compiled.terms),
        tuple(rewrite(p) for p in compiled.consts),
    )


def _negate(p: TermFragment) -> TermFragment:
    return replace(p, frame=p.frame.with_columns(-pl.col(p.value_column)))


def _join_mul(a: TermFragment, c: TermFragment, is_term: bool, divide: bool = False) -> TermFragment:
    """``a * c`` (or ``a / c``) where *c* is a const fragment.

    Joins on shared dims, broadcasts the rest. The right-hand value is renamed
    first: both sides may carry ``cval``, and a suffix collision would multiply
    a column by itself. The dims *c* contributes are broadcast, so the label
    says nothing about them.

    A divide joins **left**, so a coordinate the divisor has no value for
    yields a *null* coefficient instead of silently dropping the term. The row
    may still be masked out downstream, taking the null with it: the question
    is not whether the divisor is dense but whether it is defined where the
    model divides by it.

    *c* is variable-free, so it contributes no absence: a sparse coefficient
    zeroes a term, it does not unmake the variable underneath it. The output
    dims may be wider than ``a.dims``, which is why the presence key travels
    with the fragment rather than being re-derived from dims here (#345).
    """
    shared = [d for d in a.dims if d in c.dims]
    out_dims = a.dims + tuple(d for d in c.dims if d not in a.dims)
    right = c.frame.rename({'cval': _RHS})
    how = 'left' if divide else 'inner'
    joined = a.frame.join(right, on=shared, how=how) if shared else a.frame.join(right, how='cross')

    value, rhs = pl.col(a.value_column), pl.col(_RHS)
    combined = value / rhs if divide else value * rhs
    out = value_column(is_term=is_term)
    frame = joined.with_columns(combined.alias(out)).select(*out_dims, *carried_columns(is_term=is_term))
    return replace(a, dims=out_dims, frame=frame, is_term=is_term)


def _scattered(at: pl.Series, values: pl.Series, size: int) -> Any:
    """*values* moved to the positions *at* names, one pass, order checked."""
    import numpy as np

    indices = at.to_numpy()
    written = np.zeros(size, dtype=bool)
    written[indices] = True
    if not written.all():
        msg = 'a parameter passed the density gate but does not cover the coordinate product'
        raise LpspecError(msg)

    out = np.empty(size, dtype=np.float64)
    out[indices] = values.to_numpy()
    return out

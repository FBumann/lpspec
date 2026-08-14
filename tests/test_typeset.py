"""The typesetter (spike).

Three kinds of test, and the split is the point:

* **Shared** — run against every entry in ``FORMATS``. These are properties of
  the *walk*, so a new format inherits them and cannot quietly drop one.
* **Per format** — the spelling. Fragments, not golden documents: a golden
  file for a generator this young is rewritten by every cosmetic change and
  stops being read.
* **Compiled** — the only check that the output is real. Typst is a pip wheel
  so it runs here; LaTeX needs a toolchain and is compiled in CI instead.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import lpspec as lps
from lpspec.typeset import FORMATS, SymbolTable, to_latex, to_markdown, to_typst, typeset
from lpspec.typeset.format import OPERATOR_NAMES
from lpspec.typeset.symbols import _derive_name_symbol
from tests import golden
from tests.conftest import MODEL_PATHS, override
from tools import gallery_math

if TYPE_CHECKING:
    from lpspec.typeset.format import Format

LATEX, TYPST = FORMATS['latex'], FORMATS['typst']
EVERY_FORMAT = pytest.mark.parametrize('fmt', list(FORMATS.values()), ids=list(FORMATS))

DISPATCH = {
    'dimensions': {'snapshot': {'dtype': 'int'}, 'generator': {'dtype': 'str'}},
    'parameters': {
        'p_max': {'dims': ['generator']},
        'cost': {'dims': ['generator']},
        'load': {'dims': ['snapshot']},
    },
    'variables': {'p': {'foreach': ['snapshot', 'generator'], 'bounds': {'lower': 0, 'upper': 'p_max'}}},
    'constraints': {
        'power_balance': {
            'foreach': ['snapshot'],
            'expression': 'sum(p, over=generator) == load',
        }
    },
    'objectives': {'total_cost': {'sense': 'minimize', 'expression': 'p * cost'}},
}


# ---------------------------------------------------------------------------
# shared: properties of the walk, asserted for every format
# ---------------------------------------------------------------------------


@EVERY_FORMAT
def test_a_format_spells_every_operator_the_walk_can_emit(fmt: Format):
    """A missing spelling is a KeyError deep in a walk, on whichever model
    first happens to use that operator. Checking the table instead makes it a
    failure the format's own author sees."""
    assert set(fmt.operators) == OPERATOR_NAMES


@EVERY_FORMAT
def test_every_example_renders(fmt: Format):
    """The walk consumes the same AST as lowering, so anything ``check``
    accepts it must print — a node it forgot is an exception, not a blank."""
    for path in MODEL_PATHS:
        assert typeset(path, fmt).strip()


@EVERY_FORMAT
def test_a_dimension_index_never_steals_a_letter_a_variable_owns(fmt: Format):
    """With `plant` -> `p` and a variable `p`, the output was `p_{t,p}` and no
    reader could tell which `p` was which."""
    model = {
        'dimensions': {'plant': {'dtype': 'str'}, 'snapshot': {'dtype': 'int'}},
        'parameters': {'cost': {'dims': ['plant']}},
        'variables': {'p': {'foreach': ['snapshot', 'plant'], 'bounds': {'lower': 0}}},
        'objectives': {'o': {'expression': 'p * cost'}},
    }
    text = typeset(model, fmt)
    assert fmt.subscript('p', ['t', 'p']) not in text
    assert fmt.subscript('p', ['t', 'l']) in text


@EVERY_FORMAT
def test_a_where_lands_on_the_quantifier_not_in_the_equation(fmt: Format):
    """A mask is row absence, so it belongs to the ∀ that names the rows."""
    model = override(DISPATCH, **{'variables.p.where': 'p_max > 0'})
    text = typeset(model, fmt, legend=False)
    assert fmt.operators['forall'] in text
    assert fmt.operators['such_that'] in text


@EVERY_FORMAT
def test_translation_distinguishes_a_wrapping_edge_from_a_dropping_one(fmt: Format):
    """``edge='wrap'`` wraps and a bare shift does not — one symbol each, since a
    reader who cannot tell them apart cannot tell the two models apart either."""

    def storage(edge: str) -> dict[str, object]:
        return {
            'dimensions': {'snapshot': {'dtype': 'int'}},
            'parameters': {'load': {'dims': ['snapshot']}},
            'variables': {'soc': {'foreach': ['snapshot'], 'bounds': {'lower': 0, 'upper': 100}}},
            'constraints': {
                'balance': {
                    'foreach': ['snapshot'],
                    'expression': f'soc == shift(soc, over=snapshot, by=1{edge}) + load',
                }
            },
        }

    cyclic = fmt.operators['cyclic_minus']
    assert cyclic in typeset(storage(", edge='wrap'"), fmt, legend=False)
    assert cyclic not in typeset(storage(''), fmt, legend=False)


@EVERY_FORMAT
def test_the_legend_explains_wraparound_only_when_it_is_used(fmt: Format):
    rolled = {
        'dimensions': {'snapshot': {'dtype': 'int'}},
        'variables': {'soc': {'foreach': ['snapshot'], 'bounds': {'lower': 0}}},
        'constraints': {
            'b': {'foreach': ['snapshot'], 'expression': "soc == shift(soc, over=snapshot, by=1, edge='wrap')"}
        },
    }
    assert 'cyclic translation' in typeset(rolled, fmt)
    assert 'cyclic translation' not in typeset(DISPATCH, fmt)


@EVERY_FORMAT
def test_macros_and_named_expressions_are_expanded_away(fmt: Format):
    """What prints is the math a backend builds, not the sugar it was spelled with."""
    model = override(
        DISPATCH,
        **{'expressions.supply': 'sum(p, over=generator)', 'constraints.power_balance.expression': 'supply == load'},
    )
    assert 'supply' not in typeset(model, fmt, legend=False)


@EVERY_FORMAT
def test_an_invalid_model_fails_the_same_way_check_does(fmt: Format):
    broken = override(DISPATCH, **{'objectives.total_cost.expression': 'p * nonexistent'})
    with pytest.raises(lps.LpspecError):
        typeset(broken, fmt)


#: Syntax that could only have come from one family of formats. Markdown is
#: absent on purpose: it *is* LaTeX math in a Markdown wrapper, and inherits
#: every math method — so sharing LaTeX's spelling is the design, not a leak.
_FINGERPRINTS = {
    'latex': (r'\mathcal{', r'\mathit{', r'\sum_{', r'\begin{align'),
    'typst': ('cal(', 'italic("', 'sum_(', '#set '),
}


@pytest.mark.parametrize(
    ('name', 'foreign'),
    [('latex', 'typst'), ('typst', 'latex'), ('markdown', 'typst')],
)
def test_no_format_leaks_another_formats_syntax(name: str, foreign: str):
    """The seam's whole job. Typst syntax in the LaTeX output means the walk is
    spelling something itself instead of asking the format to.

    Checking *syntax families* rather than one rendered symbol matters: an
    earlier version of this test looked for the literal ``\\mathcal{X}``, which
    no model declares, so it passed without ever reading the output.
    """
    text = '\n'.join(typeset(p, FORMATS[name], standalone=True) for p in MODEL_PATHS)
    assert any(mark in text for mark in _FINGERPRINTS[name if name != 'markdown' else 'latex']), (
        f'{name} output contains none of its own syntax — is this test still reading anything?'
    )
    for mark in _FINGERPRINTS[foreign]:
        assert mark not in text, f'{foreign} syntax {mark!r} leaked into the {name} output'


# ---------------------------------------------------------------------------
# derivation: unambiguous by default
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('name', 'expected'),
    [
        pytest.param('p_max', r'p^{\mathrm{max}}', id='single-letter-head-so-the-tail-is-a-qualifier'),
        pytest.param('soc_max', r'\mathit{soc}^{\mathrm{max}}', id='declared-head-so-the-tail-is-a-qualifier'),
        pytest.param('marginal_cost', r'\mathit{marginal\_cost}', id='neither-so-it-stays-one-word'),
        pytest.param('shut_down', r'\mathit{shut\_down}', id='neither-even-when-the-tail-reads-like-a-qualifier'),
    ],
)
def test_an_underscore_is_only_a_qualifier_when_its_head_is_a_symbol(name: str, expected: str):
    """`marginal_cost` is not *marginal* raised to *cost*. Splitting every
    underscore turned about a third of real names into nonsense."""
    assert _derive_name_symbol(name, frozenset({'p', 'soc'}), LATEX) == expected


# ---------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'fragment',
    [
        pytest.param('p_{t,g}', id='symbols-follow-the-names-variable'),
        pytest.param(r'\mathit{load}_{t}', id='symbols-follow-the-names-parameter'),
        pytest.param(r'p^{\mathrm{max}}_{g}', id='symbols-follow-the-names-qualifier'),
        pytest.param(
            r'\sum_{g \in \mathcal{G}} p_{t,g} & = \mathit{load}_{t}',
            id='sum-binds-the-dimension-it-reduces',
        ),
        pytest.param(
            r'\sum_{t \in \mathcal{T},\ g \in \mathcal{G}} p_{t,g} \cdot \mathit{cost}_{g}',
            id='objective-sums-over-every-dim-its-term-carries',
        ),
        pytest.param(r'0 \le p_{t,g} & \le p^{\mathrm{max}}_{g}', id='bounds-become-a-domain-line'),
        pytest.param(r'\text{power\_balance}', id='names-are-escaped-in-text-mode'),
    ],
)
def test_latex_spells_the_dispatch_model(fragment: str):
    assert fragment in to_latex(DISPATCH)


@pytest.mark.parametrize(
    ('bounds', 'expected'),
    [
        ({}, r'p_{t,g} & \in \mathbb{R}'),
        ({'lower': 0}, r'p_{t,g} & \ge 0'),
        ({'upper': 10}, r'p_{t,g} & \le 10'),
    ],
)
def test_latex_a_missing_bound_is_not_silently_zero(bounds: dict[str, object], expected: str):
    model = override(DISPATCH, **{'variables.p.bounds': bounds})
    assert expected in to_latex(model)


def test_latex_binary_and_integer_variables_state_their_domain():
    model = override(
        DISPATCH,
        **{
            'variables.on': {'foreach': ['snapshot', 'generator'], 'domain': 'binary'},
            'variables.n': {'foreach': ['generator'], 'domain': 'integer', 'bounds': {'lower': 0, 'upper': 5}},
        },
    )
    tex = to_latex(model)
    assert r'\{0, 1\}' in tex
    assert r'\in \mathbb{Z}' in tex


def test_latex_a_semi_continuous_variable_states_its_zero_or_banded_domain():
    model = override(
        DISPATCH,
        **{'variables.p.domain': 'semi_continuous', 'variables.p.bounds': {'lower': 10, 'upper': 100}},
    )
    assert r'p_{t,g} & \in \{0\} \cup [10, 100]' in to_latex(model)


def test_latex_sum_renders_the_coordinate_map_as_a_set_condition():
    tex = to_latex('examples/transport.yaml', legend=False)
    assert r'\sum_{g \in \mathcal{G} \,:\, \mathrm{bus}(g) = b} p_{t,g}' in tex
    assert r'\sum_{l \in \mathcal{L} \,:\, \mathrm{to}(l) = b} f_{t,l}' in tex


def test_latex_a_sum_used_as_a_factor_is_bracketed():
    """Unbracketed, `\\sum_g x_g \\cdot 2` reads as the sum capturing the 2."""
    model = override(DISPATCH, **{'constraints.power_balance.expression': 'sum(p, over=generator) * 2 == load'})
    assert r'\left( \sum_{g \in \mathcal{G}} p_{t,g} \right) \cdot 2' in to_latex(model, legend=False)


def test_latex_standalone_is_a_whole_document():
    tex = to_latex(DISPATCH, standalone=True)
    assert tex.startswith(r'\documentclass')
    assert r'\usepackage{amsmath}' in tex
    assert tex.rstrip().endswith(r'\end{document}')


def test_latex_numbering_can_be_turned_off():
    assert r'\begin{align*}' in to_latex(DISPATCH, numbered=False)
    assert r'\begin{align}' in to_latex(DISPATCH, numbered=True)


# ---------------------------------------------------------------------------
# Typst
# ---------------------------------------------------------------------------


def test_typst_uses_its_own_grouping_and_set_notation():
    typ = to_typst(DISPATCH, legend=False)
    assert 'p_(t,g)' in typ
    assert 'sum_(g in cal(G))' in typ
    assert 'italic("load")_(t)' in typ


def test_typst_sum_renders_the_coordinate_map():
    typ = to_typst('examples/transport.yaml', legend=False)
    assert 'sum_(g in cal(G) colon upright("bus")(g) = b) p_(t,g)' in typ


# ---------------------------------------------------------------------------
# Markdown — the one that renders where the docs already live
# ---------------------------------------------------------------------------


def _generated(stem: str, legend: bool = False) -> str:
    """A shipped example rendered to Markdown with its committed symbol table."""
    return to_markdown(f'examples/{stem}.yaml', symbols=f'examples/symbols/{stem}.yaml', legend=legend)


def test_markdown_is_latex_math_in_a_markdown_wrapper():
    """The math is byte-identical to the LaTeX lane's; only the wrapper differs.
    That is the claim the module makes, so it is the one asserted."""
    md = to_markdown(DISPATCH, legend=False)
    assert r'\sum_{g \in \mathcal{G}} p_{t,g}' in md, 'the math is spelled exactly as LaTeX spells it'
    assert '#### Subject to' in md, 'the document layer is the whole difference'
    assert r'\begin{align}' not in md
    assert r'\paragraph' not in md


def test_markdown_keeps_names_out_of_the_math():
    """`\\text{total\\_cost}` is correct in a LaTeX document and wrong in a
    browser: MathJax renders the `\\_` escape literally, backslash and all. A
    name is not math, so it goes outside the `$$` as a code span."""
    md = to_markdown(DISPATCH, legend=False)
    assert '**`total_cost`**' in md
    assert '**`power_balance`**' in md
    for block in md.split('$$')[1::2]:
        assert '\\_' not in block, f'escaped underscore reached the math: {block!r}'


def test_markdown_avoids_escapes_github_eats_inside_math():
    r"""GitHub runs Markdown's backslash-escape processing *inside* `$$`.

    `\,` arrives as a literal comma and `\;` as a semicolon, so `\forall\, s`
    renders as "\u2200, s" and `\,:\,` as ",:,". Letter-named macros are
    untouched and MathJax treats them identically, so the Markdown format uses
    those. LaTeX and Typst are unaffected — no Markdown processor sees them.
    """
    md = _generated('dispatch', legend=True)
    for block in md.split('$$')[1::2]:
        for eaten in (r'\,', r'\;', r'\!', r'\:'):
            assert eaten not in block, f'{eaten!r} does not survive GitHub inside math: {block!r}'


def test_markdown_gives_each_equation_its_own_block():
    """`aligned` columns line up *across rows*. A page shows one equation at a
    time under its own heading, so the separators aligned against nothing and
    rendered as stretches of empty space."""
    md = to_markdown(DISPATCH, legend=False)
    assert md.count('$$') % 2 == 0
    assert 'aligned' not in md
    assert '&' not in md.replace('&&', ''), 'no alignment separators at all'


def test_markdown_renders_the_legend_as_a_table():
    md = to_markdown(DISPATCH)
    assert '| Symbol | Meaning |' in md
    assert '| `p_max` over' in md.replace('$p^{\\mathrm{max}}$ ', '')


GALLERY = Path(__file__).resolve().parent.parent / 'docs' / 'models'

#: Pages whose hand-written summary states the **model's** math. The notation a
#: gallery reader expects is the spec and `typeset/` is what is under test — so
#: every symbol the summary uses, the generator has to be able to reach.
REPRODUCIBLE = ('dispatch', 'monthly_budget', 'transport')

#: Pages whose summary deliberately says something *else*, each with its reason.
#: Declared rather than assumed: `test_every_summary_declares_itself` fails on a
#: page in neither list, so a new summary cannot quietly opt out of the check.
DIVERGENT = {
    'multi_period': (
        "renders the objective as one sum over the union of both terms' dims. The "
        'capex term carries (period, generator) and the operating term carries '
        '(snapshot, generator), and the engine sums each over its own dims — so the '
        'rendered sum over t as well would multiply capex by the snapshots per '
        'period. Reproducing it would need the generator to split an additive '
        'objective into one sum per term.'
    ),
    'storage': (
        'writes soc_{s-1}, ordinary index arithmetic. The model rolls, and a roll '
        'wraps — which the generator writes as the cyclic ⊖. Matching would mean '
        'either dropping the wrap or opening with a symbol nobody has met yet.'
    ),
    'piecewise': (
        "shows one generator's curve. The model carries the snapshot dim through λ "
        'as well, so the generated subscripts are (t, g, k) where the summary has (g, k).'
    ),
    'sos': (
        'states one curve, as the textbook writes it: λ_k against the breakpoints k. '
        'The model carries snapshot and generator through λ as well, so the generated '
        'subscripts are (t, g, b) — piecewise diverges from its own summary for the '
        'same reason, these being the two spellings of one formulation.'
    ),
    'transport_dantzig': (
        'is the textbook statement of the transportation problem, with an abstract '
        'c_{ij}. The model is the GAMS instance, whose cost is distance times freight over 1000.'
    ),
    'tsp_mtz': (
        'is DFJ subtour elimination — the formulation the language refuses, which is '
        'the point of the section it sits in. The model is MTZ.'
    ),
}

#: Reduction operators carry a subscript without being a symbol.
_OPERATORS = frozenset({r'\sum', r'\min', r'\max', r'\prod', r'\int'})
_SUBSCRIPTED = re.compile(r'(\\[a-zA-Z]+|[A-Za-z])\s*_\s*(?:\{([^{}]*)\}|(\S))')


def _summary(stem: str) -> str:
    """The hand-written math on a gallery page — the whole page *minus* the
    generated block, which is the definition of hand-written here.

    Not the first `$$` in the file: positional indexing survives only until
    someone adds math above it, and then it silently checks a different
    equation. Not a heading name either — `tsp_mtz` states its math under
    "What genuinely is refused", because for that page the summary is the
    formulation the language *cannot* use. Keying on the machine-maintained
    markers is the one anchor that holds for both.

    The closing marker is searched for *from* the opening one, so a marker that
    is missing and one that sits above its partner are the same failure — and
    the assertion names the file, where ``index`` would raise a bare
    ``ValueError``. Only ``$$`` blocks are returned: the prose and the YAML
    fence around them are full of identifiers like ``p_max`` and ``sum``, which
    read as subscripts.
    """
    path = GALLERY / f'{stem}.md'
    page = path.read_text()
    if gallery_math.BEGIN in page:
        begin = page.index(gallery_math.BEGIN)
        end = page.find(gallery_math.END, begin)
        assert end != -1, (
            f'{path}: has {gallery_math.BEGIN} with no {gallery_math.END} after it, '
            f'so the generated block cannot be separated from the hand-written math'
        )
        page = page[:begin] + page[end:]
    return '\n'.join(page.split('$$')[1::2])


def _symbols(latex: str) -> set[str]:
    """Every subscripted quantity, as `head_subscript` with braces dropped.

    Brace-insensitive because the two sides spell single-character subscripts
    differently by convention — a summary writes `c_g`, the generator `c_{g}` —
    and that is a spelling difference, not a disagreement about the math.
    """
    found = set()
    for head, braced, bare in _SUBSCRIPTED.findall(latex):
        if head in _OPERATORS:
            continue
        found.add(f'{head}_{f"{braced}{bare}".strip()}')
    return found


def test_every_summary_declares_itself():
    """A page with hand-written math is checked against the generator, or says
    why not. Being in neither list is the failure this guards."""
    with_math = {p.stem for p in GALLERY.glob('*.md') if p.stem != 'index' and _summary(p.stem).strip()}
    undeclared = with_math - set(REPRODUCIBLE) - set(DIVERGENT)
    assert not undeclared, (
        f'gallery summaries that neither claim reproducibility nor explain a divergence: '
        f'{sorted(undeclared)} — add each to REPRODUCIBLE or to DIVERGENT with its reason'
    )
    stale = (set(REPRODUCIBLE) | set(DIVERGENT)) - with_math
    assert not stale, f'declared pages that no longer carry hand-written math: {sorted(stale)}'


@pytest.mark.parametrize('stem', REPRODUCIBLE)
def test_a_reproducible_summary_uses_only_symbols_the_generator_emits(stem: str):
    """The oracle direction: the hand-written notation is the expectation, and
    the renderer is what has to meet it.

    This began as the opposite assertion, on `dispatch`. Its summary showed a
    bound for every `(s, g)` while the prose beneath called `where: "p_max > 0"`
    the one line worth pausing on — found by generating the same equation, and
    fixed in the same change. A summary is prose, so nothing else would notice
    it drifting again.
    """
    generated = _generated(stem)
    missing = sorted(_symbols(_summary(stem)) - _symbols(generated))
    assert not missing, (
        f'docs/models/{stem}.md writes {missing}, which the generated math does not — '
        f'either the summary drifted from the model, or the renderer cannot say what '
        f'the gallery promises it can'
    )


def test_the_dispatch_summary_still_carries_the_mask():
    """The specific regression above, pinned by value rather than by symbol set:
    `> 0` is a condition, not a subscripted quantity, so the check below would
    not see it disappear."""
    assert r'\bar p_g > 0' in _summary('dispatch')
    assert r'\bar p_{g} > 0' in _generated('dispatch')


def test_typst_standalone_adds_page_setup():
    assert to_typst(DISPATCH, standalone=True).startswith('#set page')
    assert not to_typst(DISPATCH).startswith('#set page')


@pytest.fixture(scope='module')
def typst():
    return pytest.importorskip('typst', reason='typst is a dev dependency; the bare install skips it')


def test_typst_output_compiles(typst, tmp_path: Path):
    """The only check that the Typst is real, and it has already earned its
    place: the first run rejected `minus.circle`, which is not a Typst symbol."""
    for path in MODEL_PATHS:
        source = tmp_path / f'{path.stem}.typ'
        source.write_text(to_typst(path, standalone=True))
        typst.compile(str(source), output=str(tmp_path / f'{path.stem}.pdf'))


def test_every_typst_operator_compiles(typst, tmp_path: Path):
    """Only a handful of operators appear in `examples/`; the rest would
    otherwise first fail on somebody's own model."""
    probe = tmp_path / 'operators.typ'
    probe.write_text('\n'.join(f'$ a {TYPST.operators[name]} b $' for name in sorted(OPERATOR_NAMES)))
    typst.compile(str(probe), output=str(tmp_path / 'operators.pdf'))


# ---------------------------------------------------------------------------
# golden output — the only test that notices a change nobody pinned
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('name', list(FORMATS), ids=list(FORMATS))
def test_the_output_matches_the_committed_golden_file(name: str):
    """One model, every format, byte for byte.

    Fragment assertions pin the constructs someone thought to pin, and survive
    anything leaving those substrings intact — a stray prefix, a lost space, a
    changed separator. Perturbing `TypstFormat.summation` to emit `~sum_(...)`
    failed *no test* before this existed, because every Typst assertion was a
    substring check and a `~` compiles fine.

    The same trade `examples/walkthrough.out` makes: the committed file is the
    output, so a format that starts saying something different shows up as a
    diff instead of as nothing at all.
    """
    expected = golden.path_for(name)
    actual = typeset(golden.MODEL, FORMATS[name], standalone=True)
    assert actual == expected.read_text(), (
        f'{expected.relative_to(Path.cwd())} is stale.\n'
        f'If the change was intended: `uv run python -m tests.golden`, then read the diff.'
    )


# ---------------------------------------------------------------------------
# structural well-formedness (no toolchain needed)
# ---------------------------------------------------------------------------


def _structural_errors(tex: str) -> list[str]:
    """The three ways generated LaTeX usually fails to compile.

    Not a substitute for running TeX — it cannot know whether ``\\mathcal``
    takes an argument — but brace balance, environment nesting and
    ``\\left``/``\\right`` pairing are exactly what a *generator* gets wrong,
    and they are checkable without a toolchain.
    """
    errors = []
    depth = 0
    for i, c in enumerate(tex):
        escaped = i > 0 and tex[i - 1] == '\\'
        if c == '{' and not escaped:
            depth += 1
        elif c == '}' and not escaped:
            depth -= 1
            if depth < 0:
                errors.append(f'unbalanced closing brace at offset {i}')
                break
    if depth > 0:
        errors.append(f'{depth} unclosed brace(s)')

    stack: list[str] = []
    for verb, environment in re.findall(r'\\(begin|end)\{(\w+\*?)\}', tex):
        if verb == 'begin':
            stack.append(environment)
        elif not stack:
            errors.append(rf'\end{{{environment}}} with nothing open')
        elif stack.pop() != environment:
            errors.append(rf'\end{{{environment}}} does not close the open environment')
    if stack:
        errors.append(f'environments left open: {stack}')

    left, right = tex.count(r'\left'), tex.count(r'\right')
    if left != right:
        errors.append(rf'\left/\right mismatch: {left} vs {right}')
    return errors


@pytest.mark.parametrize('path', MODEL_PATHS, ids=lambda p: p.stem)
def test_the_latex_is_structurally_well_formed(path: Path):
    assert _structural_errors(to_latex(path, standalone=True)) == []


# ---------------------------------------------------------------------------
# the symbol table
# ---------------------------------------------------------------------------

SYMBOLS = {
    'dimensions': {'generator': {'index': 'u', 'set': r'\mathcal{U}'}},
    'names': {'p': r'\pi', 'marginal_cost': r'c^{\mathrm{marg}}'},
    'descriptions': {'generator': 'dispatchable units'},
}


WITH_MARGINAL_COST = override(
    DISPATCH,
    **{'parameters.marginal_cost': {'dims': ['generator']}, 'objectives.total_cost.expression': 'p * marginal_cost'},
)


def test_the_table_overrides_and_the_rest_is_still_derived():
    tex = to_latex(WITH_MARGINAL_COST, symbols=SYMBOLS, legend=False)
    assert r'\pi_{t,u}' in tex, 'both the symbol and its subscripts were overridden'
    assert r'c^{\mathrm{marg}}_{u}' in tex
    assert r'\mathit{load}_{t}' in tex, 'untouched, so still derived'
    assert r'u \in \mathcal{U}' in tex


def test_a_description_reaches_the_legend_without_hiding_the_name():
    tex = to_latex(WITH_MARGINAL_COST, symbols=SYMBOLS)
    assert r'\texttt{generator}' in tex
    assert 'dispatchable units' in tex


@pytest.mark.parametrize(
    ('symbols', 'match'),
    [
        pytest.param({'names': {'p_maxx': 'x'}}, "Did you mean 'p_max'", id='a-misspelled-name'),
        pytest.param(
            {'dimensions': {'generatr': {'index': 'g'}}},
            "Did you mean 'generator'",
            id='a-misspelled-dimension',
        ),
        pytest.param({'symbols': {'p': 'x'}}, 'unknown section', id='an-unknown-section'),
        pytest.param({'dimensions': {'generator': {'letter': 'g'}}}, 'unknown key', id='an-unknown-key'),
    ],
)
def test_an_entry_naming_nothing_is_an_error_with_the_near_miss(symbols, match):
    """A silent typo means a symbol that never applies and a reader who never
    finds out — so it fails, and says what it probably meant."""
    with pytest.raises(lps.SchemaError, match=match):
        to_latex(DISPATCH, symbols=symbols)


def test_the_table_loads_from_a_file_and_the_committed_one_applies():
    tex = to_latex('examples/piecewise.yaml', symbols='examples/symbols/piecewise.yaml')
    assert r'\lambda_{' in tex
    assert r'k \in \mathcal{K}' in tex
    assert 'breakpoints of the cost curve' in tex


@pytest.mark.parametrize('table', sorted(Path('examples/symbols').glob('*.yaml')), ids=lambda p: p.stem)
@EVERY_FORMAT
def test_every_committed_symbol_table_still_fits_its_model(table: Path, fmt: Format):
    """A sidecar is matched to its model by filename, and nothing else ties
    them together — so renaming a parameter would leave the table naming
    something that no longer exists. `checked_against` makes that an error, and
    this is what runs it for every committed pair."""
    candidates = [Path('examples') / f'{table.stem}.yaml', Path('examples/ports') / f'{table.stem}.yaml']
    model = next((c for c in candidates if c.exists()), None)
    assert model is not None, f'{table} names no model: looked in {[str(c) for c in candidates]}'
    assert typeset(model, fmt, symbols=table).strip()


def test_a_model_renders_identically_with_an_empty_table():
    assert to_latex(DISPATCH) == to_latex(DISPATCH, symbols=SymbolTable())


def test_exported_from_the_package():
    assert lps.to_latex is to_latex
    assert lps.to_typst is to_typst

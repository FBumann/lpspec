"""The examples in the docs, checked against the code.

Every example here was wrong at some point: ``lps.write_lp`` never existed, a
dimension index was passed as a bare ``RangeIndex`` where the streaming lane
wants a ``coords=`` entry, the ``piecewise:`` block carried a sign on three
links while the prose two lines below said a sign needs exactly two, and four
module docstrings leaked the executor by never closing the ``Result``. Three
separate hand sweeps found three separate batches, which is the argument for
this file: an example nobody runs is a claim nobody checked.

Coverage cannot silently drop, and that claim needs two guards, not one.
:func:`test_every_block_is_covered` polices blocks that were *matched*; it is
blind to a fence the regex failed to recognise, which is the same silent loss
by another route. :func:`test_every_fence_is_seen` closes that by scanning for
fences language-agnostically and asserting every ``python``/``yaml`` one was
matched — so an unclosed fence, or a style the regex has not learned, fails
loudly instead of quietly shrinking the sweep.

A block may therefore be indented inside a list item, carry an info string
after the language (``python title="a.py"``), or use tilde fences — all are
matched, and the code is dedented before parsing.

Annotations go in an HTML comment on the line before the fence, so they are
invisible in rendered markdown:

    <!-- doctest: wrap=constraints -->   nest the block under that schema key
    <!-- doctest: extends=p(t,g) -->     a variable, with its dims, from the model it extends
    <!-- doctest: skip -->               excluded, and the reason belongs in a comment

A YAML block with no annotation is validated whole, which means it must
resolve its own cross-references: a lone ``parameters:`` section naming a
dimension it does not declare is a failure, and the fix is usually ``wrap=``
or ``skip`` rather than a bigger example.

In module docstrings, an example is an indented run introduced by ``::`` —
the reST literal-block marker. Guessing instead ("a run that mentions ``lps.``")
reads an indented English sentence as code and fails it as a syntax error,
which would stop prose from naming the API it documents.
"""

from __future__ import annotations

import ast
import re
import textwrap
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, NamedTuple, get_args

import pytest
import yaml

import lpspec as lps
from lpspec.language.model import Model
from lpspec.language.validation import load_model
from lpspec.relational.engines.polars.executor import PolarsExecutor
from lpspec.relational.result import Result

try:
    from lpspec import linopy as linopy_lane
except ModuleNotFoundError:
    linopy_lane = None  # bare install, no [linopy] extra

REPO = Path(__file__).resolve().parent.parent
TRACKED = [
    'README.md',
    'docs/SPEC.md',
    'docs/api.md',
    'docs/design/linopy.md',
    'docs/ARCHITECTURE.md',
    'docs/design/ceiling.md',
    'docs/ROADMAP.md',
    'docs/design/decomposition.md',
]

#: Names an example may dot into, and the object that decides what is valid.
#: Anything else (pd, np, network, ...) is external and not our contract.
ROOTS: dict[str, Any] = {
    'lps': lps,
    'lpspec_linopy': linopy_lane,
    'result': Result,
    'ex': PolarsExecutor,
    'schema': Model,
}

#: Every root an example may name, whether or not this install can resolve it.
#: Recognising an example must not depend on the extras: a linopy-lane example
#: is still one on a bare install, it just cannot be name-checked.
ROOT_NAMES = frozenset(ROOTS)
ROOTS = {name: obj for name, obj in ROOTS.items() if obj is not None}


def _unresolvable(code: str) -> set[str]:
    """Roots this example names that the install cannot supply."""
    return {root for root in ROOT_NAMES - set(ROOTS) if f'{root}.' in code}


_EXTRA = 'needs the [linopy] extra to check {}'

#: A fence may be ``` or ~~~, three or more, indented (inside a list item), and
#: may carry an info string after the language (```python title="a.py").
#: Matching only the bare form is how a block goes unchecked *without* tripping
#: the coverage guard, which only ever inspects blocks it already matched —
#: `test_every_fence_is_seen` is what actually closes that.
_FENCE = re.compile(
    r'(?:^[ \t]*<!--\s*doctest:\s*(?P<note>[^>]*?)\s*-->[ \t]*\n)?'
    r'^[ \t]*(?P<fence>`{3,}|~{3,})[ \t]*(?P<lang>python|yaml)\b[^\n]*\n'
    r'(?P<code>.*?)'
    r'^[ \t]*(?P=fence)[ \t]*$',
    re.DOTALL | re.MULTILINE,
)

#: Any fenced block, whatever its language — used only to prove _FENCE saw
#: every block it should have.
_ANY_FENCE = re.compile(r'^[ \t]*(?P<fence>`{3,}|~{3,})[ \t]*(?P<info>[^\n]*)$', re.MULTILINE)


def _fence_openings(text: str) -> list[tuple[int, str]]:
    """(line, language) for every opening fence, by walking open/close pairs."""
    out: list[tuple[int, str]] = []
    open_delim: str | None = None
    for m in _ANY_FENCE.finditer(text):
        delim, info = m.group('fence'), m.group('info').strip()
        line = text.count('\n', 0, m.start()) + 1
        if open_delim is None:
            open_delim = delim
            out.append((line, info.split()[0] if info else ''))
        elif delim == open_delim:
            open_delim = None  # closing fence
    return out


class Block(NamedTuple):
    doc: str
    lang: str
    index: int
    code: str
    note: str
    line: int

    @property
    def where(self) -> str:
        return f'{self.doc}:{self.line} ({self.lang} block #{self.index})'


def _blocks(lang: str | None = None) -> list[Block]:
    """Every tracked fenced block, optionally narrowed to one language.

    A block nested in a list item is indented, so it is dedented here: without
    that every such example fails on `unexpected indent` rather than on
    anything a reader would call a mistake. The recorded line is the fence
    itself, not the doctest comment above it.
    """
    out: list[Block] = []
    for doc in TRACKED:
        text = (REPO / doc).read_text()
        counters: dict[str, int] = {}
        for m in _FENCE.finditer(text):
            got = m.group('lang')
            i = counters.get(got, 0)
            counters[got] = i + 1
            out.append(
                Block(
                    doc=doc,
                    lang=got,
                    index=i,
                    code=textwrap.dedent(m.group('code')),
                    note=(m.group('note') or '').strip(),
                    line=text.count('\n', 0, m.start('fence')) + 1,
                )
            )
    return [b for b in out if lang is None or b.lang == lang]


def _public(obj: Any) -> set[str]:
    """Attribute names an example may use — dataclass fields included, since a
    field with no default is not a class attribute and ``dir`` misses it."""
    names = {n for n in dir(obj) if not n.startswith('_')}
    if is_dataclass(obj):
        names |= {f.name for f in fields(obj) if not f.name.startswith('_')}
    return names


# --------------------------------------------------------------------------
# python blocks
# --------------------------------------------------------------------------


@pytest.mark.parametrize('block', _blocks('python'), ids=lambda b: b.where)
def test_python_block_parses(block: Block) -> None:
    if block.note == 'skip':
        pytest.skip('explicitly skipped')
    try:
        ast.parse(block.code)
    except SyntaxError as exc:  # pragma: no cover - only on a broken doc
        pytest.fail(f'{block.where} is not valid Python: {exc}')


@pytest.mark.parametrize('block', _blocks('python'), ids=lambda b: b.where)
def test_python_block_uses_real_api(block: Block) -> None:
    """Every ``lps.x`` / ``result.x`` an example shows must exist.

    This is the check that would have caught ``lps.write_lp``, which was
    documented for months and never existed.
    """
    if block.note == 'skip':
        pytest.skip('explicitly skipped')
    if missing := _unresolvable(block.code):
        pytest.skip(_EXTRA.format(sorted(missing)))
    tree = ast.parse(block.code)
    bad: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute) or not isinstance(node.value, ast.Name):
            continue
        root = node.value.id
        if root not in ROOTS:
            continue
        if node.attr not in _public(ROOTS[root]):
            bad.append(f'{root}.{node.attr}')
    assert not bad, (
        f'{block.where} uses names that do not exist: {sorted(set(bad))}. Fix the example, or the API it documents.'
    )


def test_readme_example_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The front-door example must actually solve, and produce the number the
    README claims it produces.

    The README states its objective in a trailing comment; this is what keeps
    the two in sync.
    """
    yaml_blocks = [b for b in _blocks('yaml') if b.doc == 'README.md']
    py_blocks = [b for b in _blocks('python') if b.doc == 'README.md']
    model = next(b for b in yaml_blocks if '# dispatch.yaml' in b.code)
    script = next(b for b in py_blocks if 'lps.solve' in b.code)

    (tmp_path / 'dispatch.yaml').write_text(model.code)
    monkeypatch.chdir(tmp_path)

    ns: dict[str, Any] = {}
    exec(compile(script.code, 'README.md', 'exec'), ns)

    result = ns['result']
    assert result.is_ok

    claimed = re.search(r'#\s*([0-9]+\.?[0-9]*)\s*$', script.code, re.MULTILINE)
    assert claimed, 'README example no longer states its objective in a comment'
    assert result.objective == pytest.approx(float(claimed.group(1))), (
        f'README claims objective {claimed.group(1)}, run produced {result.objective}'
    )


# --------------------------------------------------------------------------
# yaml blocks
# --------------------------------------------------------------------------


def _entry_model(section: str) -> Any:
    """The per-entry model behind a schema section, e.g. constraints -> ConstraintDef."""
    args = get_args(Model.model_fields[section].annotation)
    return args[1] if len(args) == 2 else None


_BORROWED = re.compile(r'(?P<name>\w+)\s*(?:\((?P<dims>[^)]*)\))?')


def _borrowed(note: str) -> dict[str, list[str]]:
    """Variables an ``extends=`` block takes from the model it extends.

    Written ``extends=p(snapshot,generator)`` — the dims are part of it because
    dim checking is a language rule, so a borrowed variable with no dims would
    have ``shift(over=snapshot)`` fail on the very block that demonstrates it.
    """
    if not note.startswith('extends='):
        return {}
    return {
        m['name']: [d.strip() for d in (m['dims'] or '').split(',') if d.strip()]
        for m in _BORROWED.finditer(note.removeprefix('extends='))
    }


@pytest.mark.parametrize('block', _blocks('yaml'), ids=lambda b: b.where)
def test_yaml_block_validates(block: Block) -> None:
    """A YAML example must be a thing the schema accepts.

    Whole-section blocks go through ``Model`` — including ``piecewise:``,
    which is why this catches a sign on three links. A ``wrap=`` block shows a
    single entry of a section and deliberately omits the declarations around
    it, so it is checked against that section's own model: its *shape* is our
    claim, its cross-references are not.

    An ``extends=`` block is validated *whole* — keys, shapes and expressions —
    against a namespace widened by the names it borrows, which is what an
    extension gets from ``linopy.extend()``. It is the narrow form of ``skip``:
    a typo'd key in one of these is still a failure.
    """
    if block.note == 'skip':
        pytest.skip('explicitly skipped')

    doc = yaml.safe_load(block.code)
    assert isinstance(doc, dict), f'{block.where} is not a YAML mapping'

    if block.note.startswith('wrap='):
        section = block.note.removeprefix('wrap=')
        assert section in Model.model_fields, f'{block.where}: wrap={section!r} is not a schema section'
        model = _entry_model(section)
        for name, entry in doc.items():
            try:
                model.model_validate(entry)
            except Exception as exc:
                pytest.fail(f'{block.where}: entry {name!r} does not validate:\n{exc}')
        return

    try:
        load_model(doc, known_variables=_borrowed(block.note))
    except Exception as exc:
        pytest.fail(
            f'{block.where} does not validate:\n{exc}\n\n'
            'If the block is not meant to be a complete model — a section shown '
            'on its own still has to resolve its cross-references, so a lone '
            '`parameters:` must declare the dims it names — annotate the fence '
            'instead:\n'
            '  <!-- doctest: wrap=<section> -->  a single entry of that section\n'
            '  <!-- doctest: extends=v(dims) --> a variable borrowed from another model\n'
            '  <!-- doctest: skip -->            not a model, or wrong on purpose'
        )


# --------------------------------------------------------------------------
# the anti-rot guard
# --------------------------------------------------------------------------


def test_every_fence_is_seen() -> None:
    """`_FENCE` must match every python/yaml block that exists.

    The other guard below can only inspect blocks that were matched, so a fence
    it fails to recognise is *invisible* rather than reported — coverage drops
    with nothing to show for it. That is the failure this file exists to
    prevent, so it is checked directly against a language-agnostic scan.
    """
    missed = []
    for doc in TRACKED:
        text = (REPO / doc).read_text()
        seen = {b.line for b in _blocks() if b.doc == doc}
        for line, lang in _fence_openings(text):
            if lang in ('python', 'yaml') and line not in seen:
                missed.append(f'{doc}:{line} (```{lang})')
    assert not missed, (
        'these blocks exist but _FENCE did not match them, so nothing checks '
        'them and no other test would notice:\n  ' + '\n  '.join(missed)
    )


def test_every_block_is_covered() -> None:
    """A new example must be checkable or explicitly skipped — never ignored."""
    unhandled = []
    for block in _blocks():
        if block.note == 'skip' or block.note.startswith('wrap='):
            continue
        if block.lang == 'python':
            continue  # parsed and name-checked by test_docstring_example_uses_real_api
        keys = yaml.safe_load(block.code)
        if not isinstance(keys, dict) or not set(keys) <= set(Model.model_fields):
            unhandled.append(block.where)
    assert not unhandled, (
        'these YAML blocks are neither whole schema sections nor annotated, so '
        'nothing checks them:\n  ' + '\n  '.join(unhandled) + '\n'
        'Add <!-- doctest: wrap=<section> --> or <!-- doctest: skip --> above the fence.'
    )


# --------------------------------------------------------------------------
# module docstrings — where the executor leak actually lived
# --------------------------------------------------------------------------

DOCSTRING_MODULES = ['src/lpspec/__init__.py', 'src/lpspec/api.py', 'src/lpspec/linopy/__init__.py']


def _docstring_examples(path: Path) -> list[str]:
    """Indented runs introduced by ``::`` — reST literal blocks.

    The marker is the author's own statement that a run is code, which is why
    it is used instead of guessing. Guessing by "mentions a known root" reads
    an indented English sentence containing ``lps.solve`` as an example and
    fails it as a syntax error, so prose could not mention the API it
    documents.
    """
    tree = ast.parse(path.read_text())
    doc = ast.get_docstring(tree) or ''
    runs: list[tuple[str, list[str]]] = []
    current: list[str] = []
    lead = ''  # the last non-blank unindented line before the current run
    prev = ''
    for line in doc.splitlines():
        if not line.strip() or line.startswith('    '):
            if not current:
                lead = prev
            current.append(line)
            continue
        if current:
            runs.append((lead, current))
            current = []
        prev = line
    if current:
        runs.append((lead, current))

    out = []
    for lead, run in runs:
        text = '\n'.join(run).strip('\n')
        if not text.strip() or not lead.rstrip().endswith('::'):
            continue
        out.append(textwrap.dedent(text))
    return out


class Example(NamedTuple):
    module: str
    index: int
    code: str

    @property
    def where(self) -> str:
        return f'{self.module} (docstring example #{self.index})'


def _docstring_cases() -> list[Example]:
    """One case per example, so an install that cannot check one of them says
    so about that example rather than about the whole module."""
    return [
        Example(module, i, code)
        for module in DOCSTRING_MODULES
        for i, code in enumerate(_docstring_examples(REPO / module))
    ]


@pytest.mark.parametrize('module', DOCSTRING_MODULES)
def test_module_documents_its_api(module: str) -> None:
    """A module docstring that stops showing its API is a doc regression the
    per-example tests below cannot see — they would just collect nothing."""
    assert _docstring_examples(REPO / module), f'{module}: no API example found in the module docstring'


@pytest.mark.parametrize('example', _docstring_cases(), ids=lambda e: e.where)
def test_docstring_example_uses_real_api(example: Example) -> None:
    """Every name a module docstring's example dots into has to exist.

    Syntax is checked on every install; only the name check needs the object
    behind the root, so only that part stands down on a bare one.
    """
    try:
        tree = ast.parse(example.code)
    except SyntaxError as exc:
        pytest.fail(f'{example.where} is not valid Python: {exc}\n{example.code}')
    if missing := _unresolvable(example.code):
        pytest.skip(_EXTRA.format(sorted(missing)))
    bad = [
        f'{n.value.id}.{n.attr}'
        for n in ast.walk(tree)
        if isinstance(n, ast.Attribute)
        and isinstance(n.value, ast.Name)
        and n.value.id in ROOTS
        and n.attr not in _public(ROOTS[n.value.id])
    ]
    assert not bad, f'{example.where} uses names that do not exist: {sorted(set(bad))}'


def test_tracked_docs_exist() -> None:
    """Renaming a doc must not silently drop its examples from the sweep."""
    missing = [d for d in TRACKED if not (REPO / d).is_file()]
    assert not missing, f'tracked docs missing (update TRACKED): {missing}'
    assert _blocks('python'), 'no python examples found — the regex has drifted'
    assert _blocks('yaml'), 'no yaml examples found — the regex has drifted'

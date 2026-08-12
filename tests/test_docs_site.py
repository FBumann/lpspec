"""The docs are read in two places; these are the checks that keep them honest in both.

``docs/`` is browsed on GitHub and served as a site, from one set of files. A
link *inside* ``docs/`` is relative and mkdocs validates it — ``build --strict``
in CI fails on a dead one. A link *outside* ``docs/`` cannot be relative,
because the site has no `../CONTRIBUTING.md` to resolve to, so it is written as
a full GitHub URL.

That convention is the whole mechanism, and it is unenforceable by mkdocs in
both directions: a relative link escaping ``docs/`` builds a silent 404, and a
blob URL is opaque to every checker there is — the file it names can be deleted
and nothing anywhere fails. Hence this module.

``docs/README.md`` is the one exception and is exempted throughout: it is
excluded from the site (``exclude_docs``), exists only as the folder view
GitHub renders, and its relative links out of the tree are correct there.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / 'docs'
REPO_URL = 'https://github.com/fluxopt/lpspec'
BLOB = f'{REPO_URL}/blob/main'

#: `](target)` and `[label]: target`, the two ways markdown names a destination.
_TARGETS = re.compile(r'\]\(\s*([^)\s]+)|^\[[^\]]+\]:\s+(\S+)', re.MULTILINE)

#: Already absolute, a bare fragment, or a protocol that names no path.
_ABSOLUTE = re.compile(r'^([a-z][a-z0-9+.-]*:|//|#|/)', re.IGNORECASE)


def _pages() -> list[Path]:
    """Every page mkdocs builds — so, not `docs/README.md`."""
    return [p for p in sorted(DOCS.rglob('*.md')) if p.relative_to(DOCS).as_posix() != 'README.md']


def _targets(page: Path) -> list[str]:
    return [inline or reference for inline, reference in _TARGETS.findall(page.read_text())]


def test_no_relative_link_escapes_the_docs_tree():
    """The failure mkdocs cannot see.

    `[x](../CONTRIBUTING.md)` is correct in the repo and a 404 on the site.
    mkdocs resolves it against `docs/`, finds nothing above the root, and —
    because the target is outside the tree it knows about — does not treat it
    as a broken internal link. It just ships. Write the full GitHub URL.
    """
    escaping = []
    for page in _pages():
        for target in _targets(page):
            if _ABSOLUTE.match(target):
                continue
            path = target.partition('#')[0]
            if not path:
                continue
            resolved = (page.parent / path).resolve()
            if resolved != DOCS and DOCS not in resolved.parents:
                escaping.append(f'{page.relative_to(REPO)} -> {target}')
    assert not escaping, (
        f'relative links pointing outside docs/, which 404 on the site: {escaping}\nwrite them as {BLOB}/<path> instead'
    )


def test_every_blob_url_names_a_file_that_exists():
    """The other half: a blob URL is checked by nothing at all.

    mkdocs treats it as external and never follows it; the repo has no reason
    to notice it. So a page can go on pointing at `bench/results/latest.jsonl`
    long after the file moves, and the first report is a reader hitting
    GitHub's 404.
    """
    broken = []
    for page in [*_pages(), DOCS / 'README.md']:
        for target in _targets(page):
            if not target.startswith(BLOB):
                continue
            relative = target.removeprefix(f'{BLOB}/').partition('#')[0]
            if not (REPO / relative).exists():
                broken.append(f'{page.relative_to(REPO)} -> {relative}')
    assert not broken, f'links to repo files that no longer exist: {broken}'


def test_links_to_our_own_files_are_all_spelled_the_same_way():
    """One spelling, so the check above cannot be dodged.

    A link at a file in this repo written any other way — `tree/`, `raw/`, a
    permalinked sha, a branch that will vanish — reaches the right page today
    and is skipped by the existence check, which only recognises `blob/main`.
    Issue and PR links are not file links and are left alone.
    """
    file_shaped = re.compile(rf'^{re.escape(REPO_URL)}/(blob|tree|raw|blame)/')
    stray = [
        f'{page.relative_to(REPO)} -> {target}'
        for page in [*_pages(), DOCS / 'README.md']
        for target in _targets(page)
        if file_shaped.match(target) and not target.startswith(f'{BLOB}/')
    ]
    assert not stray, f'links at repo files not written as {BLOB}/<path>: {stray}'


def test_the_convention_is_actually_in_use():
    """A guard on the guards.

    Every assertion above passes vacuously on a docs tree with no outbound
    links at all — including one where a bad refactor stripped them. Pin that
    the arrangement they describe exists.
    """
    urls = [t for page in _pages() for t in _targets(page) if t.startswith(BLOB)]
    assert len(urls) >= 15, f'expected the docs to link out to the repo; found {len(urls)}'


def test_every_figure_is_referenced_and_every_reference_resolves():
    """Both directions, because a figure rots from either end.

    A reference to a chart that was never written is a broken image on the
    published page; a chart nobody references is what `docs/bench.svg` became
    — drawn once for a PR in 2026, hardcoded for light mode, and still in the
    tree three engines later. `bench.plot` writes this directory whole, so
    anything in it that the docs do not name is stale by construction.

    Figures come in light/dark pairs: mkdocs-material's palette toggle stamps
    the host page, which an `<img>`-referenced SVG cannot see, so the page
    carries both and the `#only-*` suffixes choose.
    """
    charts = DOCS / 'charts'
    assert charts.is_dir(), 'docs/charts is gone — the benchmarks page embeds it'
    referenced = set()
    for page in DOCS.rglob('*.md'):
        for match in re.finditer(r'\]\((charts/[^)\s]+?\.svg)(#only-(?:light|dark))?\)', page.read_text()):
            referenced.add(match.group(1))
            assert match.group(2), f'{page.name} embeds {match.group(1)} without a #only-light/#only-dark suffix'
            assert (DOCS / match.group(1)).exists(), (
                f'{page.name} references a figure that does not exist: {match.group(1)}'
            )
    on_disk = {f'charts/{p.name}' for p in charts.glob('*.svg')}
    assert on_disk == referenced, f'figures nobody embeds: {sorted(on_disk - referenced)}'
    for name in sorted(on_disk):
        light = name.replace('-dark.svg', '-light.svg')
        assert light in on_disk, f'{name} has no light twin — the toggle would leave a reader with nothing'


def test_the_home_page_still_carries_its_math_block():
    """`tools.gallery_math --check` also fills the tabs on `docs/index.md`, and
    it fills what it finds — a page whose markers were dropped in an edit stops
    being checked without anything failing. Pin that they are there.

    The content itself is not asserted here; that is the generator's job, and
    `test_the_gallery_math_is_current` runs it.
    """
    from tools import gallery_math

    page = (DOCS / 'index.md').read_text()
    assert gallery_math.HOME_BEGIN in page and gallery_math.HOME_END in page, (
        f'docs/index.md lost its {gallery_math.HOME_BEGIN}/{gallery_math.HOME_END} markers — '
        f'the LaTeX tabs are generated, and an unmarked page silently opts out'
    )


# --------------------------------------------------------------------------
# SPEC §0 — the laws
# --------------------------------------------------------------------------

SPEC = DOCS / 'SPEC.md'

#: A `## 0. The laws` row: `| 7 | text | [§6](#6-absence) |`
_LAW_ROW = re.compile(r'^\|\s*(\d+)\s*\|(.+?)\|([^|]*)\|\s*$', re.MULTILINE)


def _laws() -> list[tuple[str, str, str]]:
    text = SPEC.read_text()
    start = text.index('## 0. The laws')
    return _LAW_ROW.findall(text[start : text.index('\n## 1. ', start)])


def _headings() -> set[str]:
    """Every heading in SPEC.md as GitHub would slug it."""
    slugs = set()
    for line in SPEC.read_text().splitlines():
        if line.startswith('#'):
            title = line.lstrip('#').strip()
            slugs.add(re.sub(r'[^a-z0-9 -]', '', title.lower()).replace(' ', '-'))
    return slugs


def test_every_law_cites_the_section_that_elaborates_it():
    """A law is the canonical statement and the section below is the detail.

    An unlinked law is the failure that would otherwise pass silently: mkdocs
    fails the build on a *dead* anchor, but a row that cites nothing at all
    resolves fine and quietly becomes a second, drifting home for the rule.
    """
    laws = _laws()
    assert len(laws) >= 10, f'expected the law block to be found and populated; got {len(laws)} rows'

    slugs = _headings()
    broken = []
    for number, _, citation in laws:
        targets = re.findall(r'\]\(#([a-z0-9-]+)\)', citation)
        if not targets:
            broken.append(f'law {number} cites no section')
        broken += [f'law {number} -> #{t}' for t in targets if t not in slugs]
    assert not broken, f'laws whose citation does not resolve in SPEC.md: {broken}'


MERMAID = re.compile(r'```mermaid\n(.*?)```', re.DOTALL)
_ID = r'[A-Za-z_][A-Za-z0-9_]*'


def _mermaid_blocks() -> list[tuple[Path, str]]:
    return [(p, block) for p in DOCS.rglob('*.md') for block in MERMAID.findall(p.read_text())]


def test_every_mermaid_edge_points_at_a_node_that_exists():
    """A renamed node must not silently orphan the edge that pointed at it.

    Mermaid does not error on an unknown id — it invents an empty box labelled
    with the id and draws the edge into that. So a rename leaves a diagram that
    still renders, still passes `mkdocs build --strict`, and is wrong: a stray
    box where the real one used to be, and the real one floating unconnected.

    That happened while the second engine was being drawn in — `BIND` became
    `BINDP` in its declaration and nowhere else — and nothing caught it but a
    reader.
    """
    for path, block in _mermaid_blocks():
        declared = set(re.findall(rf'\b({_ID})\s*[\[\("]', block))
        declared |= set(re.findall(rf'subgraph\s+({_ID})', block))
        referenced = set()
        for raw in block.splitlines():
            line = raw.split('%%')[0]
            if '-->' not in line:
                continue
            for part in re.split(r'-->\|[^|]*\||-->', line):
                name = part.strip().split('[')[0].split('(')[0].strip()
                if re.fullmatch(_ID, name):
                    referenced.add(name)
        orphans = sorted(referenced - declared)
        assert not orphans, f'{path.name}: edges point at undeclared nodes {orphans}'


def test_every_mermaid_subgraph_is_closed():
    """One missing `end` swallows the rest of the diagram into the last box."""
    for path, block in _mermaid_blocks():
        opens = len(re.findall(r'^\s*subgraph\b', block, re.MULTILINE))
        closes = len(re.findall(r'^\s*end\s*$', block, re.MULTILINE))
        assert opens == closes, f'{path.name}: {opens} subgraph vs {closes} end'

"""Turn a results JSONL into the markdown that goes in docs/benchmarks.md.

    uv run python -m bench.report bench/results/latest.json

Nothing here recomputes or smooths anything: repeats collapse by *minimum*,
which is the usual choice for a benchmark because noise only ever adds. The
point of this module existing at all is that the published table has one
provenance — a file — instead of being retyped by hand and then outliving the
harness that produced it.
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
from pathlib import Path
from typing import Any

from bench import results as bench_results

#: The published ladder's pair: the eager lane is what the engine is judged
#: against, so the ratio columns are lpspec ÷ linopy.
DEFAULT_ARMS = ('lpspec', 'linopy')

#: The arms this run tabulates and the one the ratio divides by, set from
#: `--arms`. Rebindable because the published ladder is not the only comparison
#: this renders: `--arms lpspec polars` prices the engine choice itself
#: (duckdb-spike.md), and hard-coding the pair would mean a second renderer
#: that drifts from this one.
ARMS = DEFAULT_ARMS
_RATIO_AGAINST = DEFAULT_ARMS[1]


def load(
    path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    records = bench_results.load(path)
    run = next((r for r in records if r.get('record') == 'run'), {})
    gates = [r for r in records if r.get('record') == 'gate']
    timings = [r for r in records if r.get('record') == 'timing']
    loop = [r for r in records if r.get('record') == 'loop']
    return run, gates, timings, loop


Row = dict[str, Any]
Key = tuple[str, str, str, str]


def _key(r: Row) -> Key:
    return (r['case'], r['size'], r.get('sink', 'lp'), r['arm'])


def best(timings: list[Row]) -> dict[Key, Row]:
    """(case, size, sink, arm) -> the fastest repeat."""
    out: dict[Key, Row] = {}
    for r in timings:
        if 'error' in r:
            continue
        key = _key(r)
        if key not in out or r['wall_seconds'] < out[key]['wall_seconds']:
            out[key] = r
    return out


def failures(timings: list[Row]) -> dict[Key, str]:
    """A run that died is a measurement, and the report renders it as one."""
    return {_key(r): r['error'] for r in timings if 'error' in r}


def _si(n: float) -> str:
    for unit, scale in (('M', 1e6), ('k', 1e3)):
        if n >= scale:
            return f'{n / scale:.6g}{unit}'
    return f'{n:.0f}'


def _gb(n: float) -> str:
    return f'{n / 1e9:.2f}'


def _ratio(a: float | None, b: float | None) -> str:
    return f'{a / b:.2f}x' if a and b else '—'


_DENSITY_RUNG = re.compile(r'd\d+$')


def sizes_of(case: str, rows: dict[Key, Row], sink: str = 'lp', *, density: bool = False) -> list[str]:
    """Rung labels for *case*, smallest model first.

    The density sweep is held at one model size, so mixing it into the size
    ladder would sort four densities in among four sizes and read as a single
    monotone column that is really two axes. They get separate tables.
    """
    seen = {
        s: r['counts']['columns']
        for (c, s, k, _), r in rows.items()
        if c == case and k == sink and bool(_DENSITY_RUNG.match(s)) == density
    }
    return sorted(seen, key=lambda s: seen[s])


#: The figures `bench.plot` writes, in the order the page reads them. Paired
#: light/dark because mkdocs-material's toggle stamps the *host* page and an
#: `<img>`-referenced SVG cannot see it; GitHub takes the first of the pair.
FIGURES = (
    ('wall', 'Wall time to a loaded solver, by model size'),
    ('peak', 'Peak resident memory, by model size'),
    ('cases', 'Every model in the corpus, through the highs sink'),
    ('sinks', 'The l rung through every sink, both arms'),
)


def figures() -> str:
    """The figure embeds, as markdown that renders in both places.

    One pointer at the interactive page for the whole set rather than one per
    figure: these are pictures, and reading a value off one is what that page
    is for.
    """
    out = []
    for name, alt in FIGURES:
        out.append(f'![{alt}](charts/{name}-light.svg#only-light)')
        out.append(f'![{alt}](charts/{name}-dark.svg#only-dark)')
        out.append('')
    out.append(
        '*Static, so they render anywhere. The same data with a cursor: [the chart page](benchmarks-scaling.html).*'
    )
    return '\n'.join(out)


#: How each arm reaches each sink, said once so a table can name its own seam.
_SEAM_ENGINE = {
    'lp': 'Both arms write the LP file; only the engine that filled the tables differs.',
    'highs': (
        'Both arms end holding a populated `highspy.Highs` with `run()` never '
        'called, through the same `build_highs` seam. Only the engine that '
        'filled the tables differs.'
    ),
}

_SEAM = {
    'lp': 'lpspec writes the LP file, linopy through its `lp-polars` writer.',
    'highs': (
        'Both arms end holding a populated `highspy.Highs` with `run()` never '
        'called: lpspec through `build_highs`, linopy through '
        '`to_highspy(set_names=False)`. The simplex is the same work whoever '
        'filled the model, so timing it would say nothing about the lane that '
        'filled it.'
    ),
    'gurobi': (
        'Both arms end holding a populated `gurobipy.Model` with `optimize()` '
        'never called: lpspec through `build_gurobi`, linopy through '
        '`to_gurobipy(set_names=False)`. Opt-in — it needs the `[gurobi]` '
        'extra — and the same discipline as the `highs` sink.'
    ),
}


def table(case: str, rows: dict[Key, Row], sink: str = 'lp') -> str:
    """One case's rungs through one sink, as a markdown table.

    The caption is bold rather than a heading: these live inside a collapsed
    ``<details>``, and a heading in there still lands in the table of contents
    — a rail full of entries for tables the page has just called the appendix.
    """
    cols = ARMS
    head = (
        ['variables', 'live', 'rows']
        + [f'wall: {a}' for a in cols]
        + ['wall']
        + [f'peak: {a}' for a in cols]
        + ['peak', 'LP']
    )
    seam = _SEAM[sink] if 'linopy' in cols else _SEAM_ENGINE[sink]
    lines = [
        f'**{case} — {sink} sink**',
        '',
        seam,
        '',
        '| ' + ' | '.join(head) + ' |',
        '|' + '---|' * len(head),
    ]
    for size in sizes_of(case, rows, sink):
        arms = {a: rows.get((case, size, sink, a)) for a in cols}
        ref = next((r for r in arms.values() if r), None)
        if ref is None:
            continue
        wall = {a: (r['wall_seconds'] if r else None) for a, r in arms.items()}
        peak = {a: (r['peak_rss_bytes'] if r else None) for a, r in arms.items()}
        cells = [
            _si(ref['counts']['columns']),
            _live(ref),
            _si(ref['counts']['rows']),
            *(f'{wall[a]:.2f} s' if wall[a] else '—' for a in cols),
            _ratio(wall[ARMS[0]], wall[_RATIO_AGAINST]),
            *(f'{_gb(peak[a])} GB' if peak[a] else '—' for a in cols),
            _ratio(peak[ARMS[0]], peak[_RATIO_AGAINST]),
            f'{ref["lp_bytes"] / 1e6:.0f} MB' if ref.get('lp_bytes') else '—',
        ]
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def _live(r: Row) -> str:
    """What fraction of the coordinate product survived the mask.

    Reported rather than assumed. `dispatch` declares `where: p_max > 0` and
    keeps 100% of its product — the engine pays for a mask that removes
    nothing, and that only shows up if the harness measures it.
    """
    frac = r.get('live_fraction')
    return '—' if frac is None else f'{frac * 100:.0f}%'


def _settling(best: dict[tuple[str, str, str], Row], seen: list[tuple[str, str]]) -> str:
    """How far the first recorded round sits from steady state, per arm.

    Rendered from the results file rather than stated, so a refresh moves it
    (#619). What the pair measures is one recorded round against the best of
    the rest: the harness warms up before recording, so **neither end carries
    the one-time import cost** and this is the loop settling, not a cold start.
    Measuring that cost needs a fresh interpreter per arm, which no rung takes.
    """
    per_arm = []
    for arm in ARMS:
        deltas = sorted(
            (best[(case, size, arm)]['first_build_seconds'] - best[(case, size, arm)]['steady_build_seconds']) * 1000
            for case, size in seen
            if (case, size, arm) in best and best[(case, size, arm)].get('first_build_seconds') is not None
        )
        if deltas:
            per_arm.append(f'{statistics.median(deltas):+.1f} ms on {arm}')
    if not per_arm:
        return ''
    return (
        'The harness warms up before it records, so neither column carries the '
        'one-time import cost: the median gap between them is ' + ' and '.join(per_arm) + '.'
    )


def marginal(loop_rows: list[Row]) -> str:
    """First model in a process, against every model after it.

    Two questions with two answers, and the gap between them is larger than
    most of the differences this file reports — so publishing one figure would
    misreport whichever use case it was not.

    The density rungs are skipped: they are several masks at one model size and
    would render as rows sharing a label. They have their own table.
    """
    best: dict[tuple[str, str, str], Row] = {}
    for r in loop_rows:
        if 'error' in r or r.get('steady_build_seconds') is None:
            continue
        if _DENSITY_RUNG.match(r['size']):
            continue
        key = (r['case'], r['size'], r['arm'])
        if key not in best or r['first_build_seconds'] < best[key]['first_build_seconds']:
            best[key] = r
    if not best:
        return ''

    seen = sorted(
        {(c, s) for c, s, _ in best},
        key=lambda k: best[(k[0], k[1], 'lpspec')].get('nominal_variables', 0),
    )
    lines = [
        '### Marginal cost per model',
        '',
        'Build only, repeated in one process. **first** is the first recorded round '
        'and **steady** the best of the rounds after it, so the pair is what a '
        'rolling horizon pays for its second window against its first. ' + _settling(best, seen),
        '',
        '| case | vars | lpspec: first | lpspec: steady | linopy: first | linopy: steady | steady vs linopy |',
        '|---|---|---|---|---|---|---|',
    ]
    for case, size in seen:
        ours, eager = best.get((case, size, 'lpspec')), best.get((case, size, 'linopy'))
        if not ours or not eager:
            continue
        lines.append(
            '| '
            + ' | '.join(
                [
                    case,
                    _si(ours.get('nominal_variables', 0)),
                    f'{ours["first_build_seconds"] * 1000:.1f} ms',
                    f'**{ours["steady_build_seconds"] * 1000:.1f} ms**',
                    f'{eager["first_build_seconds"] * 1000:.1f} ms',
                    f'{eager["steady_build_seconds"] * 1000:.1f} ms',
                    _ratio(ours['steady_build_seconds'], eager['steady_build_seconds']),
                ]
            )
            + ' |'
        )
    return '\n'.join(lines)


def density(rows: dict[Key, Row]) -> str:
    """One model size, four mask densities — the axis the ladder cannot show.

    A mask is row absence relationally and a NaN-padded dense array eagerly, so
    this is the one comparison where the two lanes are not doing the same work
    in different orders — they are doing different amounts of work.
    """
    cases = [c for c in sorted({c for c, _, _, _ in rows}) if sizes_of(c, rows, 'lp', density=True)]
    if not cases:
        return ''
    cols = ARMS
    head = (
        ['case', 'live', 'variables']
        + [f'wall: {a}' for a in cols]
        + ['wall']
        + [f'peak: {a}' for a in cols]
        + ['peak']
    )
    lines = [
        '### The mask sweep',
        '',
        'One model size, through the `lp` sink. For `nodal`, `live` is how many '
        'of the 12 technologies each node has installed: 12 / 6 / 3 / 1.',
        '',
        '| ' + ' | '.join(head) + ' |',
        '|' + '---|' * len(head),
    ]
    for case in cases:
        for size in reversed(sizes_of(case, rows, 'lp', density=True)):
            arms = {a: rows.get((case, size, 'lp', a)) for a in cols}
            ref = next((r for r in arms.values() if r), None)
            if ref is None:
                continue
            wall = {a: (r['wall_seconds'] if r else None) for a, r in arms.items()}
            peak = {a: (r['peak_rss_bytes'] if r else None) for a, r in arms.items()}
            cells = [
                case,
                _live(ref),
                _si(ref['counts']['columns']),
                *(f'{wall[a]:.2f} s' if wall[a] else '—' for a in cols),
                _ratio(wall[ARMS[0]], wall[_RATIO_AGAINST]),
                *(f'{_gb(peak[a])} GB' if peak[a] else '—' for a in cols),
                _ratio(peak[ARMS[0]], peak[_RATIO_AGAINST]),
            ]
            lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def main(argv: list[str] | None = None) -> int:
    """Print the published page for the given result files.

    The parity gate is enforced by the harness now rather than recorded by it —
    a case whose arms disagree fails its measurements outright, so a file that
    exists at all was gated. Older files still carry the records, which is why
    both branches are here.

    Per-case tables are collapsed, and in ``<details>`` rather than mkdocs'
    ``???`` because these pages are read on GitHub too, where only the HTML
    form folds. The figures are the reading; the numbers stay one click away,
    because a chart nobody can check is decoration.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('results', type=Path, nargs='*', default=[Path('bench/results/latest.json')])
    ap.add_argument(
        '--arms',
        nargs=2,
        default=list(DEFAULT_ARMS),
        metavar=('SUBJECT', 'AGAINST'),
        help='the two arms to tabulate. Ratios are SUBJECT / AGAINST, so the order is the claim.',
    )
    opts = ap.parse_args(argv)

    global ARMS, _RATIO_AGAINST
    ARMS = tuple(opts.arms)
    _RATIO_AGAINST = opts.arms[1]

    run: dict[str, Any] = {}
    gates: list[Row] = []
    timings: list[Row] = []
    loop: list[Row] = []
    for path in opts.results:
        one_run, one_gates, one_timings, one_loop = load(path)
        run = run or one_run
        gates += one_gates
        timings += one_timings
        loop += one_loop
    rows = best(timings)
    failed = failures(timings)

    versions = ', '.join(f'{k} {v}' for k, v in run.get('versions', {}).items() if v)
    print(f'{run.get("platform", "?")}, python {run.get("python", "?")} — {versions}.')
    print()
    if gates:
        print(
            'Parity gate: '
            + '; '.join(
                f'{g["case"]} objectives agree to {g["relative_gap"]:.1e} relative'
                if g['passed']
                else f'{g["case"]} FAILED'
                for g in gates
            )
            + '.'
        )
    else:
        print('Parity gate: enforced at measurement time — every arm below built the same model.')
    print()
    print(figures())
    for case in sorted({c for c, _, _, _ in rows}):
        sinks = [k for k in sorted({k for c, _, k, _ in rows if c == case}) if sizes_of(case, rows, k)]
        if not sinks:
            continue
        print()
        print(f'<details markdown="1">\n<summary><b>{case}</b> — every rung, every sink</summary>\n')
        for sink in sinks:
            print(table(case, rows, sink))
            print()
        print('</details>')
    loop_table = marginal(loop)
    if loop_table:
        print()
        print(loop_table)
    density_table = density(rows)
    if density_table:
        print()
        print(density_table)
    for key, error in sorted(failed.items()):
        print(f'\n<!-- {" ".join(k for k in key if k)}: {error} -->')
    return 0


if __name__ == '__main__':
    sys.exit(main())

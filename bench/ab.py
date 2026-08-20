"""Is a change faster, or is the machine noisy? — A/B with a verdict.

The build is where the tree spends most of its time and the hardest thing to
measure: repeated runs of *identical code* spread 12-55%, because each one
allocates and frees the better part of a gigabyte and the allocator, the page
cache and the thermal state all drift under it. Against that, eyeballing two
minima is not a comparison. Every wrong perf number this repo has published
came from doing exactly that.

    uv run python -m bench.ab record before --case profiled --size l [--phase emit]
    ...switch the code...
    uv run python -m bench.ab record after  --case profiled --size l
    uv run python -m bench.ab compare before after

Four things it does that reading minima does not:

**A verdict, not a ratio.** ``compare`` bootstraps a confidence interval on the
difference and refuses to call a winner when it straddles zero. A 9% gap under
a 30% spread is not a 9% win, and this says so rather than leaving it to
whoever reads the table.

**Interleaving.** ``record --append`` adds to an existing file, so alternating
short runs — before, after, before, after — puts both arms under the same drift
instead of giving the second one whatever the machine had become. Block
ordering is the layout most vulnerable to a machine that changes, and it is the
one you get by default.

**A fingerprint of what actually ran.** Each file records the git HEAD, whether
the tree was dirty, and a hash of the source tree. ``compare`` refuses two files
with the same source hash — which is what a measurement of nothing looks like,
and is easier to produce than it sounds: ``git stash push -- <path>`` on a tree
whose change is already *committed* stashes nothing at all, and both arms then
run the same code.

**Preconditions.** Timing a 1 GB write on a volume at 98% measures the volume,
and CPU already busy is borrowed back from the thing being timed. Both are
checked before the first round rather than wondered about after the last.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import shutil
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

from bench import cases as bench_cases

#: Below this the machine is doing something else, and it is doing it to the
#: measurement. Not fatal — a warning, because "idle" is a judgement and a
#: laptop is never fully one.
MIN_IDLE_FRACTION = 0.85

#: A nearly-full volume makes writes wildly variable, which is how a
#: filesystem measurement once got published as a 16% sort improvement.
MIN_FREE_BYTES = 20 * 1024**3

RESULTS = Path(__file__).parent / 'results' / 'ab'


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='python -m bench.ab')
    sub = parser.add_subparsers(dest='verb', required=True)

    rec = sub.add_parser('record', help='time one arm and save the samples')
    rec.add_argument('name', help='what to call this arm')
    rec.add_argument('--case', required=True, choices=sorted(bench_cases.CASES))
    rec.add_argument('--size', required=True)
    rec.add_argument('--rounds', type=int, default=9)
    rec.add_argument(
        '--phase',
        default='build',
        choices=('build', 'emit'),
        help='what to time. `emit` writes the LP text to /dev/null off an already-built model, '
        'which takes the filesystem out of it — a 1 GB write measures the volume, not the writer',
    )
    rec.add_argument('--append', action='store_true', help='add to an existing arm, for interleaving')
    rec.add_argument('--force', action='store_true', help='record even if the machine looks busy')

    cmp_ = sub.add_parser('compare', help='two arms, with a verdict')
    cmp_.add_argument('before')
    cmp_.add_argument('after')
    cmp_.add_argument('--confidence', type=float, default=0.95)

    args = parser.parse_args(argv)
    return _record(args) if args.verb == 'record' else _compare(args)


# ---------------------------------------------------------------------------
# recording
# ---------------------------------------------------------------------------


def _record(args: argparse.Namespace) -> int:
    complaints = _preconditions()
    for line in complaints:
        print(f'  ! {line}')
    if complaints and not args.force:
        print('\nRefusing to record. Fix these, or pass --force and say so when you quote the number.')
        return 1

    import lpspec as lps

    case = bench_cases.CASES[args.case]
    sources = case.data(case.shape(args.size))
    lps.build(str(case.model), sources).close()  # the first build warms the page cache; it is not a sample

    samples = []
    if args.phase == 'build':
        for _ in range(args.rounds):
            started = time.perf_counter()
            ex = lps.build(str(case.model), sources)
            samples.append(time.perf_counter() - started)
            ex.close()
    else:
        from lpspec.relational.sinks.writers.lp_file import write_lp_file

        ex = lps.build(str(case.model), sources)
        tables = ex._tables()
        for _ in range(args.rounds):
            started = time.perf_counter()
            write_lp_file(tables, '/dev/null')
            samples.append(time.perf_counter() - started)
        ex.close()

    path = RESULTS / f'{args.name}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    held = json.loads(path.read_text()) if args.append and path.exists() else None
    if held is not None and (held['case'], held['size'], held.get('phase')) != (args.case, args.size, args.phase):
        print(f'  ! {path.name} holds a different workload than {args.case}/{args.size} {args.phase}')
        return 1

    record = {
        'case': args.case,
        'size': args.size,
        'phase': args.phase,
        'samples': (held['samples'] if held else []) + samples,
        'source': _source_hash(),
        'commit': _git('rev-parse', '--short', 'HEAD'),
        'dirty': bool(_git('status', '--porcelain')),
        'platform': platform.platform(),
    }
    path.write_text(json.dumps(record, indent=2) + '\n')
    print(f'{args.name}: {len(record["samples"])} samples of {args.case}/{args.size}  -> {path}')
    _line(args.name, record['samples'])
    return 0


def _preconditions() -> list[str]:
    """What is wrong with the machine, in the words of the mistake it causes."""
    complaints = []
    free = shutil.disk_usage(Path.cwd()).free
    if free < MIN_FREE_BYTES:
        complaints.append(f'{free / 1024**3:.1f} GB free — a nearly-full volume makes writes wildly variable')
    idle = _idle_fraction()
    if idle is not None and idle < MIN_IDLE_FRACTION:
        complaints.append(f'{idle:.0%} idle — the rest is being borrowed from whatever you are timing')
    return complaints


def _idle_fraction() -> float | None:
    """Idle CPU as a fraction, or ``None`` where it cannot be read.

    Two samples, because the first is an average since boot on every platform
    that offers one at all — and a machine idle for a week reads as idle now
    whatever it is doing.
    """
    try:
        import psutil
    except ImportError:
        return None
    psutil.cpu_percent(interval=None)
    return 1.0 - psutil.cpu_percent(interval=1.0) / 100.0


def _source_hash() -> str:
    """A hash of the engine source, so two arms cannot silently be one.

    Content rather than a commit, because the mistake this catches is an
    *uncommitted* arm that never got applied — the commit would be identical
    either way and the samples would look like a very quiet machine.
    """
    digest = hashlib.sha256()
    root = Path(__file__).parent.parent / 'src'
    for path in sorted(root.rglob('*.py')):
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def _git(*args: str) -> str:
    try:
        return subprocess.run(['git', *args], capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ''


# ---------------------------------------------------------------------------
# comparing
# ---------------------------------------------------------------------------


def _compare(args: argparse.Namespace) -> int:
    before, after = (_load(name) for name in (args.before, args.after))
    if before is None or after is None:
        return 1
    keyed = ('case', 'size', 'phase')
    if [before.get(k) for k in keyed] != [after.get(k) for k in keyed]:
        print('  ! the two arms are not the same workload')
        return 1
    if before['source'] == after['source']:
        print(f'  ! both arms hash to {before["source"]} — the same code ran twice, and this compares nothing.')
        print('    A `git stash push -- <path>` stashes nothing when the change is already committed.')
        return 1

    phase = before.get('phase', 'build')
    print(f'\n{before["case"]}/{before["size"]} {phase}   {len(before["samples"])} + {len(after["samples"])} samples\n')
    _line(args.before, before['samples'])
    _line(args.after, after['samples'])

    low, high = _interval(before['samples'], after['samples'], args.confidence)
    shift = _statistic(after['samples']) / _statistic(before['samples']) - 1
    print(f'\n  difference      {shift:+.1%}')
    print(f'  {args.confidence:.0%} interval    {low:+.1%} .. {high:+.1%}')
    if low < 0 < high:
        print('\n  NOT RESOLVABLE — the interval crosses zero. More rounds, or a quieter machine.')
        print('  Interleave with `record --append` so both arms meet the same drift.')
    else:
        print(f'\n  {"FASTER" if high < 0 else "SLOWER"} — the interval clears zero.')
    for arm, name in ((before, args.before), (after, args.after)):
        if arm['dirty']:
            print(f'  (note: {name} was recorded against a dirty tree)')
    return 0


def _load(name: str) -> dict[str, Any] | None:
    path = RESULTS / f'{name}.json'
    if not path.exists():
        print(f'  ! no arm called {name!r} in {RESULTS}')
        return None
    return json.loads(path.read_text())


def _statistic(samples: list[float]) -> float:
    """The minimum: noise only ever adds, which is the rule the ladder uses."""
    return min(samples)


def _interval(before: list[float], after: list[float], confidence: float) -> tuple[float, float]:
    """A bootstrap interval on the relative difference of the two minima.

    Resampling rather than a t-test, because the statistic is a *minimum* and
    the distribution behind it is neither normal nor symmetric — a long right
    tail of rounds that met the garbage collector.
    """
    rng = random.Random(0)
    shifts = []
    for _ in range(2000):
        a = _statistic([rng.choice(before) for _ in before])
        b = _statistic([rng.choice(after) for _ in after])
        shifts.append(b / a - 1)
    shifts.sort()
    tail = (1 - confidence) / 2
    return shifts[int(tail * len(shifts))], shifts[int((1 - tail) * len(shifts)) - 1]


def _line(name: str, samples: list[float]) -> None:
    low = min(samples)
    spread = (max(samples) / low - 1) * 100
    print(
        f'  {name:14} min {low * 1000:8.1f} ms   median {statistics.median(samples) * 1000:8.1f}   spread {spread:5.1f}%'
    )


if __name__ == '__main__':
    raise SystemExit(main())

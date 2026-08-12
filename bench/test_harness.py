"""The harness measures what it says it measures.

`test_ladder.py` is the measurement; this is the part of it that has to be true
for a number to mean anything. It is fast, needs no data and no rung, so it
runs on a bare `pytest bench` before anything is timed.
"""

from __future__ import annotations

import os

from bench.workloads import _engine


def test_the_default_arm_clears_the_engine_rather_than_leaving_it() -> None:
    """A set-only switch leaks, and a leak here is a confident wrong number.

    One pytest session is one interpreter, so `LPSPEC_ENGINE` set by an arm
    that names an engine outlives that arm. The default arm has to clear it —
    otherwise the first named engine selects itself for every arm after it and
    a two-engine comparison measures one engine against itself, at ratios near
    1.00 that look like a result.

    The old runner spawned a process per measurement and could not have this
    bug; the docstring saying so outlived the runner it described.
    """
    _engine('polars')
    assert os.environ.get('LPSPEC_ENGINE') == 'polars'

    _engine(None)
    assert 'LPSPEC_ENGINE' not in os.environ, (
        'the default arm left the previous engine selected, so every arm after it measures that one'
    )

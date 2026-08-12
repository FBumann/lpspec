"""The LP sink renders doubles exactly, and writes the same bytes twice.

``lp_file`` writes numbers by casting them to string, because emit is almost
entirely float-to-text and a cast is far cheaper than a format string. That
trade is only free if the cast is shortest-*round-trip* and not merely
shortest, so the property is pinned here rather than left to the golden file —
a golden proves the bytes did not move, not that they are correct.

Reproducibility (#109) is pinned here too, for the same reason: a golden file
proves one write, and the failure mode is two writes of one model differing.
"""

from __future__ import annotations

import hashlib
import math
import struct
import tempfile
from pathlib import Path

import polars as pl
import pytest

import lpspec as lps
from lpspec.relational.sinks.writers import lp_file
from lpspec.relational.sinks.writers.lp_file import _number, _signed
from tests.conftest import DISPATCH_MODEL, override

#: Doubles that break naive formatters: repeating binary fractions, the
#: extremes of the exponent range, a denormal, and the signed zeros.
AWKWARD = [
    0.1,
    1 / 3,
    2 / 3,
    0.3,
    1e-17,
    1.7976931348623157e308,
    2.2250738585072014e-308,
    5e-324,
    123456789.12345679,
    1e20,
    -0.0,
    0.0,
]


def _rendered(render, values: list[float]) -> list[str]:
    """*render* applied to ``values`` as the sink applies it, in order."""
    frame = pl.DataFrame({'v': values}, schema={'v': pl.Float64})
    return frame.select(render(pl.col('v'))).to_series().to_list()


def _signed_text(value: pl.Expr) -> pl.Expr:
    """The sign and magnitude ``_signed`` hands to ``concat_str``, as one string.

    The sink never glues them itself — it passes both into the ``concat_str``
    that builds the whole term — so the property under test is what that
    concatenation produces.
    """
    return pl.concat_str(*_signed(value))


def _bits(x: float) -> bytes:
    """The bit pattern, so ``-0.0`` and ``0.0`` compare unequal."""
    return struct.pack('<d', x)


@pytest.mark.parametrize('value', AWKWARD, ids=repr)
def test_plain_cast_round_trips(value: float) -> None:
    """``_number`` — how bounds and right-hand sides are written."""
    (text,) = _rendered(_number, [value])
    assert _bits(float(text)) == _bits(value)


@pytest.mark.parametrize('value', AWKWARD, ids=repr)
def test_signed_coefficient_round_trips(value: float) -> None:
    """``_signed`` — how objective and matrix coefficients are written.

    The sign is normalised away for ``-0.0`` (see :func:`_signed`), so the
    round-trip is checked on magnitude there: ``+0.0`` and ``-0.0`` are the
    same coefficient, and only one of them is expressible after a ``+``.
    """
    (text,) = _rendered(_signed_text, [value])
    assert text[0] in '+-', f'coefficient {text!r} carries no explicit sign'
    assert text[:2] != '+-', f'coefficient {text!r} carries two signs'
    assert float(text) == value, '-0.0 == 0.0, which is the whole point'


def test_negative_zero_coefficient_is_written_once() -> None:
    """The trap the spelled-out zero in :func:`_signed` exists to close.

    ``-0.0`` is reachable — any negative coefficient times a zero parameter —
    and it satisfies ``>= 0``, so a naive sign arm emits ``+`` in front of a
    cast that still reads ``-0.0``.
    """
    (text,) = _rendered(_signed_text, [-0.0])
    assert text == '+0.0'


def test_extremes_do_not_become_infinite() -> None:
    """A formatter that drops exponent digits turns ``1e308`` into ``inf``."""
    for text in _rendered(_number, [1.7976931348623157e308, 5e-324]):
        assert math.isfinite(float(text))
        assert float(text) != 0.0


def test_written_bounds_are_bit_exact() -> None:
    """End to end: awkward data in, the same doubles back out of the file."""
    upper = [1 / 3, 1e-17]
    cost = [2 / 3, 1.7976931348623157e308]
    data = {
        'p_max': pl.DataFrame({'generator': ['wind', 'gas'], 'value': upper}),
        'cost': pl.DataFrame({'generator': ['wind', 'gas'], 'value': cost}),
        'load': pl.DataFrame({'snapshot': [0], 'value': [0.0]}),
    }
    with tempfile.TemporaryDirectory() as tmp:
        lp = Path(tmp) / 'model.lp'
        with lps.build(DISPATCH_MODEL, data) as ex:
            ex.write(lp)
        text = lp.read_text()

    section = text.split('bounds\n')[1].split('\nend')[0]
    written = sorted(float(line.rsplit('<=', 1)[1]) for line in section.strip().splitlines())
    assert written == sorted(upper)

    objective = text.split('obj:\n')[1].split('\ns.t.')[0]
    coefficients = sorted(float(line.split(' x')[0]) for line in objective.strip().splitlines())
    assert coefficients == sorted(cost)


def test_one_model_writes_the_same_bytes_every_time(tmp_path: Path) -> None:
    """#109 — reproducible output, which is a property of the whole file.

    Sized well past one morsel on purpose. The constraint section is the only
    place ordering can escape: its terms arrive by join, and a parallel engine
    hands back one row's terms in whatever order it finished them, which no
    amount of ordering the rows afterwards repairs. So the model needs many
    terms per row and enough rows for the engine to split the work — a handful
    of constraints would pass whatever the sink did.
    """
    generators = [f'g{i}' for i in range(20)]
    snapshots = 200
    schema = override(DISPATCH_MODEL, **{'dimensions.generator.values': generators})
    data = {
        'p_max': pl.DataFrame({'generator': generators, 'value': [100.0 + i for i in range(len(generators))]}),
        'cost': pl.DataFrame({'generator': generators, 'value': [1.0 + i / 8 for i in range(len(generators))]}),
        'load': pl.DataFrame({'snapshot': list(range(snapshots)), 'value': [50.0 + t % 7 for t in range(snapshots)]}),
    }

    written = []
    with lps.build(schema, data) as ex:
        for attempt in range(3):
            lp = tmp_path / f'{attempt}.lp'
            ex.write(lp)
            written.append(hashlib.sha256(lp.read_bytes()).hexdigest())

    assert len(set(written)) == 1, 'the same model wrote different bytes'

    # A second *build*, not a second write: three writes share one matrix, so
    # they cannot see an engine that hands its rows back in a different order
    # each time. Ordering the matrix inside the engine would hide that; the
    # sink carrying its own sort key is what actually makes it true.
    for _ in range(2):
        with lps.build(schema, data) as rebuilt:
            lp = tmp_path / 'rebuilt.lp'
            rebuilt.write(lp)
            assert hashlib.sha256(lp.read_bytes()).hexdigest() == written[0], (
                'two builds of one model wrote different bytes'
            )


def test_chunking_the_constraint_section_leaves_the_bytes_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The seams are invisible — chunking bounds the writer's peak, not its output.

    The suite's models fit inside one `EMIT_BUDGET`, so without forcing the
    budget down no test ever crosses a seam: a writer that dropped, doubled or
    reordered a boundary row would pass everything else here. A budget of a few
    nonzeros puts a seam every handful of rows.
    """
    generators = [f'g{i}' for i in range(5)]
    snapshots = 40
    schema = override(DISPATCH_MODEL, **{'dimensions.generator.values': generators})
    data = {
        'p_max': pl.DataFrame({'generator': generators, 'value': [100.0 + i for i in range(len(generators))]}),
        'cost': pl.DataFrame({'generator': generators, 'value': [1.0 + i / 8 for i in range(len(generators))]}),
        'load': pl.DataFrame({'snapshot': list(range(snapshots)), 'value': [50.0 + t % 7 for t in range(snapshots)]}),
    }

    with lps.build(schema, data) as ex:
        ex.write(tmp_path / 'one.lp')
        monkeypatch.setattr(lp_file, 'EMIT_BUDGET', 3)
        ex.write(tmp_path / 'many.lp')

    assert (tmp_path / 'one.lp').read_bytes() == (tmp_path / 'many.lp').read_bytes()


def test_section_keywords_survive_sections_far_larger_than_a_buffer(tmp_path: Path) -> None:
    """The sink writes the keywords itself and polars writes the sections.

    Two writers on one handle, alternating. They agree only for as long as
    polars goes through the handle's buffer rather than around it to the file
    descriptor — and a small model proves nothing, because everything fits in
    the buffer and the ordering cannot be observed. So each section here is
    megabytes: if a keyword ever lands somewhere other than between the two
    sections it separates, it lands in the middle of one of them.
    """
    generators = [f'g{i}' for i in range(50)]
    snapshots = 2000
    schema = override(DISPATCH_MODEL, **{'dimensions.generator.values': generators})
    data = {
        'p_max': pl.DataFrame({'generator': generators, 'value': [100.0 + i for i in range(len(generators))]}),
        'cost': pl.DataFrame({'generator': generators, 'value': [1.0 + i / 8 for i in range(len(generators))]}),
        'load': pl.DataFrame({'snapshot': list(range(snapshots)), 'value': [50.0 + t % 7 for t in range(snapshots)]}),
    }

    lp = tmp_path / 'model.lp'
    with lps.build(schema, data) as ex:
        ex.write(lp)
    lines = lp.read_text().splitlines()

    keywords = ['min', 'obj:', 's.t.', 'bounds', 'end']
    at = [i for i, line in enumerate(lines) if line in keywords]
    assert [lines[i] for i in at] == keywords, 'a section keyword is missing, doubled or out of order'

    variables = len(generators) * snapshots
    objective, bounds = at[1], at[3]
    assert lp.stat().st_size > 4_000_000, 'sections too small for the buffer boundary to be crossed'
    assert sum(1 for line in lines[objective + 1 : at[2]] if line.startswith(('+', '-'))) == variables
    assert sum(1 for line in lines[bounds + 1 : at[4]] if ' <= x' in line) == variables

"""The ``lp_file`` sink: the model as LP text.

Portability, debugging, and the differential oracle. Every section is a lazy
frame sunk straight into the open file, so the rendered text is polars' to
stream and no byte is written twice.

Numbers go through polars' float cast, which round-trips exactly: the text a
solver reads back is the double the engine computed.

**Every section is written in label order.** A solver does not care, but a
reader diffing two LP files does, and so does anyone checking that a model
builds the same bytes twice (#109).
"""

from __future__ import annotations

from pathlib import Path
from typing import IO, TYPE_CHECKING

import polars as pl

from lpspec.relational.sinks.tables import SENSE_CODES

if TYPE_CHECKING:
    from lpspec.relational.sinks.tables import ModelTables


#: How the LP format spells each comparison. Derived from the engine's own
#: vocabulary rather than written out, so a sense added there reaches the file
#: or raises here, instead of being rendered as whatever the loop last saw.
#: The format differs on one word: it writes an equality as ``=``.
_LP_SENSE = {sense: '=' if sense == '==' else sense for sense in SENSE_CODES}

#: Nonzeros per constraint chunk. A chunk's rendered lines live in memory until
#: it is sunk, so this is the knob that bounds the writer's peak rather than its
#: speed: chunking at this width takes most of the constraint section out of
#: peak for no change in the bytes written. Wider costs memory for nothing; much
#: narrower pays per-chunk overhead on every range (#189).
EMIT_BUDGET = 2_000_000


def _sink(frame: pl.LazyFrame, f: IO[bytes]) -> None:
    """Append a one-column frame to *f*, one raw line per row.

    A CSV writer with the CSV switched off, straight into the handle the caller
    holds: polars writes through its buffer, so an ``f.write()`` between two
    sinks lands between them and no concatenation pass rereads the file.

    ``maintain_order`` is polars' default, stated rather than inherited because
    the parameter is documented as unstable and a flipped default would make
    the bytes non-reproducible in silence (#109).
    """
    frame.sink_csv(f, include_header=False, quote_style='never', maintain_order=True)


def write_lp_file(model: ModelTables, path: str | Path) -> None:
    """Write the model as LP text.

    ``cols`` is positional, so the bounds section's index is added inside the
    streamed pipeline. The constraint section goes out one row range at a time,
    since a chunk's rendered lines are held until it is sunk.

    ``semi-continuous`` is linopy's spelling of the section; HiGHS's reader
    accepts it beside its own ``semi``, so a file this writes and one the
    eager lane writes are read by the same parsers.
    """
    path = Path(path)
    objective = model.obj.lazy().sort('col').select(_term(pl.col('coeff'), pl.col('col')))
    bounds = (
        model.cols.lazy()
        .with_row_index('col')
        .select(
            pl.concat_str(
                _bound(pl.col('lb'), '-infinity').alias('lb'),
                pl.lit(' <= x').alias('open'),
                _digits(pl.col('col')),
                pl.lit(' <= ').alias('close'),
                _bound(pl.col('ub'), '+infinity').alias('ub'),
            )
        )
    )

    with open(path, 'wb') as f:
        f.write((b'min' if model.objective_sense == 'min' else b'max') + b'\n\nobj:\n')
        if model.objective_constant:
            f.write(f'{model.objective_constant:+.17g}\n'.encode())
        _sink(objective, f)

        f.write(b'\ns.t.\n\n')
        for block in model.row_blocks(EMIT_BUDGET):
            _sink(_constraint_lines(model, block.lo, block.hi, model.matrix_block(block.lo, block.hi)), f)

        f.write(b'\nbounds\n')
        _sink(bounds, f)

        sections = (('binary', 'binary'), ('integer', 'general'), ('semi_continuous', 'semi-continuous'))
        for variable_type, keyword in sections:
            chosen = model.cols.lazy().with_row_index('col').filter(pl.col('vtype') == variable_type)
            if chosen.select(pl.len()).collect().item() == 0:
                continue
            f.write(f'\n{keyword}\n'.encode())
            _sink(chosen.select(pl.concat_str(pl.lit('x'), _digits(pl.col('col')))), f)

        if model.sos.height:
            f.write(b'\nsos\n')
            _sink(_set_lines(model), f)

        f.write(b'\nend\n')


def _set_lines(model: ModelTables) -> pl.LazyFrame:
    """Each special-ordered set as one ``s0: S2 :: x3:1 x4:2`` line.

    linopy's spelling of the section, so a file this writes and a file the
    eager lane writes are read by the same parsers.

    **The one section gathered rather than interleaved.** A set's members have
    to reach one line, where the constraint section sorts one row per output
    line instead; what makes that affordable is that a set is a handful of
    members and a model declares far fewer sets than rows. Order is the
    stream's own, and ``maintain_order`` is what keeps a group's line the same
    bytes twice.

    Written even where the reader may refuse it: HiGHS has no SOS concept and
    its parser says so, which is the honest outcome for a solver that cannot
    answer the question.
    """
    return (
        model.sos.lazy()
        .group_by('set', maintain_order=True)
        .agg(
            pl.col('type').first(),
            pl.concat_str(pl.lit('x'), _digits(pl.col('col')), pl.lit(':'), _digits(pl.col('weight')))
            .str.join(' ')
            .alias('members'),
        )
        .select(
            pl.concat_str(
                pl.lit('s'),
                _digits(pl.col('set')),
                pl.lit(': S'),
                _digits(pl.col('type')),
                pl.lit(' :: '),
                pl.col('members'),
            )
        )
    )


def _constraint_lines(model: ModelTables, lo: int, hi: int, entries: pl.DataFrame) -> pl.LazyFrame:
    """Every constraint line for rows ``[lo, hi)``, one sorted stream.

    One row per *output line*, interleaved by sorting, so nothing gathers a
    row's terms into a string list first — a ``group_by('row')`` into a list
    column and an explode measured a large multiple of this (#520). *entries* is the
    chunk's slice of the matrix from :meth:`ModelTables.matrix_block`, and the
    anti-join gives a termless row the line a solver still needs to parse.

    **The order is one integer, and the only other column.** A row's lines
    occupy ``slots`` consecutive keys — header, placeholder, each term at its
    column index, sense — so one sort settles both the row order and the order
    within a row, which is what #109 pins.

    The key is **chunk-relative**: each range is sunk before the next is built,
    so ``row - lo`` bounds the product by a chunk's height rather than the
    model's, where a global row would be one careless model away from
    overflowing ``Int64`` and reordering the file in silence.

    The terms are sorted although they arrive sorted: the union subsumes the
    order and the bytes are identical without it, but the union sort merges
    pre-ordered runs rather than permuting them, and dropping it costs emit on
    every case measured (#520).
    """
    slots = model.cols.height + 3

    def _key(within: pl.Expr) -> pl.Expr:
        return ((pl.col('row') - lo) * slots + within).alias('key')

    rows = model.rows.lazy().filter(pl.col('row').is_between(lo, hi, closed='left'))
    matrix = entries.lazy()
    header = rows.select(
        _key(pl.lit(0, dtype=pl.Int64)),
        pl.concat_str(pl.lit('c').alias('c'), _digits(pl.col('row')), pl.lit(':').alias('colon')).alias('line'),
    )
    placeholder = rows.join(matrix.select('row'), on='row', how='anti').select(
        _key(pl.lit(1, dtype=pl.Int64)),
        pl.lit('+0 x0').alias('line'),
    )
    terms = matrix.sort('row', 'col').select(
        _key(pl.col('col').cast(pl.Int64) + 2),
        _term(pl.col('coeff'), pl.col('col')).alias('line'),
    )
    footer = rows.select(
        _key(pl.lit(slots - 1, dtype=pl.Int64)),
        pl.concat_str(
            pl.col('sense').replace_strict(_LP_SENSE, return_dtype=pl.String),
            pl.lit(' '),
            _number(pl.col('rhs')),
        ).alias('line'),
    )
    return pl.concat([header, placeholder, terms, footer]).sort('key').select('line')


def _term(coeff: pl.Expr, col: pl.Expr) -> pl.Expr:
    """One ``+1.5 x7`` term, allocated once.

    Chaining ``+`` would make each of the four pieces its own pass over a
    full-width string column.
    """
    return pl.concat_str(*_signed(coeff), pl.lit(' x'), _digits(col))


def _number(value: pl.Expr) -> pl.Expr:
    """A float as LP text."""
    return value.cast(pl.String)


def _signed(value: pl.Expr) -> tuple[pl.Expr, pl.Expr]:
    """A coefficient, sign always explicit — the LP format needs the ``+``.

    Two pieces rather than one finished string: the cast already carries the
    ``-``, so only a non-negative value needs a sign glued on and the sign
    column stays one character wide. Rendering ``abs()`` under a ``when``
    instead would render the magnitude at full width in both arms to discard
    one.

    Zero is spelled out rather than cast because ``-0.0`` is ``>= 0``: it takes
    the ``+`` arm while the cast renders ``-0.0``, giving ``+-0.0``, which no LP
    parser accepts. Any negative coefficient times a zero parameter reaches it.
    """
    return (
        pl.when(value >= 0).then(pl.lit('+')).otherwise(pl.lit('')).alias('sign'),
        pl.when(value == 0).then(pl.lit('0.0')).otherwise(_number(value)).alias('magnitude'),
    )


def _bound(value: pl.Expr, infinite: str) -> pl.Expr:
    """A bound, with the LP format's own spelling for an unbounded one."""
    return pl.when(value.is_infinite()).then(pl.lit(infinite)).otherwise(_number(value))


def _digits(value: pl.Expr) -> pl.Expr:
    """An index as text — never in scientific notation, whatever its size."""
    return value.cast(pl.Int64).cast(pl.String)

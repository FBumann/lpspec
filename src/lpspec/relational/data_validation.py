"""Is the bound data usable? One place, for this lane.

The split this module makes explicit (#351): **spec validation**
(`language/validation.py`) is everything decidable from the file alone and is
where `check()` happens; **data validation** is here — is it there, can it be
read, is it single-valued per coordinate, are its labels real. The two
positions where law 8 grants no default (a divisor, a bound) stay with the
assembly, needing the matrix.

Every function is a pure question over frames and declarations, holding no
executor state, so what counts as usable data can be read without following the
build.

**Scoped to this lane on purpose.** These take tidy polars frames; the eager
lane reads pandas/xarray natively, so it keeps its own checks in
`linopy/loader.py` rather than paying a copy of every parameter to adapt.
The lanes share the *wording* (`lpspec/errors.py`) and the *contract* —
`tests/test_data_parity.py` asserts they reach the same verdict on the same bad
data, which is what keeps the duplication honest.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import polars as pl

from lpspec.errors import DataError, duplicate_coordinate_message, unknown_labels_message

if TYPE_CHECKING:
    from lpspec.relational import plan

#: The dimension frames a check reads labels out of, by dimension name. Only the
#: ones already built: a dimension derived *from* the parameters is not here when
#: a parameter is checked, and has nothing to answer.
Dimensions = Mapping[str, pl.LazyFrame]


def check_one_row_per_coordinate(p: plan.ParameterDeclaration, frame: pl.LazyFrame, dimensions: Dimensions) -> None:
    """A parameter is a function of its dims: one row per coordinate.

    A parameter with **no dims** has exactly one coordinate, so the rule reads
    as "exactly one row" — the case where breaking it is least visible, since a
    dimensionless parameter broadcasts by joining on nothing, which is correct
    for one row and a silent row multiplication for two: duplicate columns for
    one variable in a bound, duplicate mask rows in a where (#166).

    Labels are checked here too, against dimensions that have an index of their
    own; one derived *from* the parameters is not built yet and would have
    nothing to answer, the union of what arrived being its definition (#350).

    Every cheap question runs in one pass over the source. *Naming* an offender
    costs a pass of its own — the duplicate ``group_by`` being the single most
    expensive step of a large build — so those run only on a path about to
    raise. The aggregate names use ``#`` so they cannot collide with a dim's.
    ``.implode()`` on the membership test says "this whole collection" where
    ``is_in`` against a bare Series is ambiguous and deprecated in polars.
    """
    if not p.dims:
        rows = frame.select(pl.len()).collect().item()
        if rows != 1:
            raise DataError(
                f"parameter '{p.name}' is declared with no dims, which means one value "
                f'broadcast everywhere — but its source has {rows} rows. '
                f'Declare the dims it is indexed by, or reduce the source to a single row.'
            )
        return

    known = {d: dimensions[d].select('val').collect()['val'] for d in p.dims if d in dimensions}
    answers = (
        frame.select(
            pl.struct(p.dims).is_duplicated().any().alias('#duplicated'),
            *(pl.col(d).is_in(labels.implode()).all().alias(f'#known {d}') for d, labels in known.items()),
        )
        .collect()
        .row(0, named=True)
    )

    for d, labels in known.items():
        if not answers[f'#known {d}']:
            strangers = frame.filter(~pl.col(d).is_in(labels.implode())).select(pl.col(d).unique()).collect()
            raise DataError(unknown_labels_message(p.name, d, strangers[d].to_list(), labels.to_list()))

    if not answers['#duplicated']:
        return
    duplicated = frame.group_by(p.dims).agg(pl.len().alias('#rows')).filter(pl.col('#rows') > 1).head(3).collect()
    shown = '; '.join(
        ', '.join(f'{d}={row[d]!r}' for d in p.dims) + f' ({row["#rows"]} rows)'
        for row in duplicated.iter_rows(named=True)
    )
    raise DataError(duplicate_coordinate_message(p.name, shown, list(p.dims)))


def check_coordinates_single_valued(d: str, names: list[str], frame: pl.LazyFrame) -> None:
    """One label, one coordinate value — two rows disagreeing is a data bug.

    It names *every* offending coordinate in one pass, rather than raising on
    the first and leaving the rest to be found one build at a time.
    """
    if not names:
        return
    counts = frame.group_by(d).agg(pl.col(c).n_unique().alias(c) for c in names).collect()
    bad = {c: n for c in names if (n := int((counts[c] > 1).sum()))}
    if not bad:
        return
    listed = '; '.join(f"'{c}' ({n} label(s))" for c, n in sorted(bad.items()))
    raise DataError(
        f"dimension '{d}' carries more than one value per label for coordinate(s): "
        f'{listed}. A coordinate is single-valued per label — reduce the source to '
        f'one row per {d}, or model the relation as a parameter instead.'
    )


def check_coordinate_containment(d: str, cname: str, target: str, dimensions: Dimensions) -> None:
    """Every coordinate value must be a label of the dimension it targets.

    A *null* is not a violation — the label belongs to no group, the same
    row-absence idiom the rest of the engine uses. Only a value that is present
    and unknown is a typo, and that one drops terms silently.
    """
    known = dimensions[target].select(pl.col('val').alias(cname))
    bad = (
        dimensions[d]
        .select(cname)
        .filter(pl.col(cname).is_not_null())
        .join(known, on=cname, how='anti')
        .unique()
        .head(5)
        .collect()
    )
    if bad.height == 0:
        return
    shown = ', '.join(repr(v) for v in bad[cname].to_list())
    raise DataError(
        f"dimension '{d}' coordinate '{cname}' has value(s) that are not "
        f"'{target}' coordinates: {shown}. Every value must be a declared "
        f"'{target}' label — otherwise sum(over={d}, group_by={cname}) drops "
        f'those terms in the join that places them, and the model builds and '
        f'solves without them.'
    )

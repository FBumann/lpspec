"""The two engines side by side: what the SQL costs, method for method.

    uv run python -m bench.engine_diff          # the table
    uv run python -m bench.engine_diff --show _translate_fragment

`bench/duckdb-spike.md` priced a duckdb return by counting lines of the polars
engine. That is an estimate of *effort*. This measures the thing effort cannot
stand in for — what the result reads like — by putting each operator's two
implementations next to each other and counting only code.

Both sides are counted the same way: blank lines, comments and docstrings
excluded, so what is compared is the operator rather than the prose around it.
"""

from __future__ import annotations

import argparse
import ast
import inspect
import sys
import textwrap
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lpspec.relational.engines.duck import compiler as duck_compiler
from lpspec.relational.engines.duck import executor as duck_executor
from lpspec.relational.engines.polars import compiler as polars_compiler
from lpspec.relational.engines.polars import executor as polars_executor
from lpspec.relational.engines.polars import labels as polars_labels

#: Operator, then where each engine implements it. `None` means the engine has
#: no counterpart — which is itself a measurement, and the reason `_factored`
#: and the three-way label strategy appear here at all.
PAIRS: list[tuple[str, Any, Any]] = [
    ('frame', polars_compiler.PolarsCompiler.frame, duck_compiler.DuckCompiler.frame),
    (
        '_coordinate_product',
        polars_compiler.PolarsCompiler._coordinate_product,
        duck_compiler.DuckCompiler._coordinate_product,
    ),
    ('parameter_join', polars_compiler.PolarsCompiler.parameter_join, duck_compiler.DuckCompiler.parameter_join),
    (
        '_predicate',
        polars_compiler.PolarsCompiler._predicate,
        (duck_compiler.DuckCompiler._predicate, duck_compiler.DuckCompiler._mark_defined),
    ),
    ('bounds', polars_compiler.PolarsCompiler.bounds, duck_compiler.DuckCompiler.bounds),
    ('expression', polars_compiler.PolarsCompiler.expression, duck_compiler.DuckCompiler.expression),
    (
        '_variable_fragment',
        polars_compiler.PolarsCompiler._variable_fragment,
        duck_compiler.DuckCompiler._variable_fragment,
    ),
    ('_sum_fragment', polars_compiler.PolarsCompiler._sum_fragment, duck_compiler.DuckCompiler._sum_fragment),
    ('_group_fragment', polars_compiler.PolarsCompiler._group_fragment, duck_compiler.DuckCompiler._group_fragment),
    (
        '_translate_fragment',
        polars_compiler.PolarsCompiler._translate_fragment,
        (duck_compiler.DuckCompiler._translate_fragment, duck_compiler.DuckCompiler._moved),
    ),
    ('_filled_edge', polars_compiler.PolarsCompiler._filled_edge, duck_compiler.DuckCompiler._filled_edge),
    ('_edge', polars_compiler.PolarsCompiler._edge, duck_compiler.DuckCompiler._edge),
    ('_vacated', polars_compiler.PolarsCompiler._vacated, duck_compiler.DuckCompiler._vacated),
    ('constant_scalar', polars_compiler.PolarsCompiler.constant_scalar, duck_compiler.DuckCompiler.constant_scalar),
    ('_join_mul', polars_compiler._join_mul, duck_compiler._join_mul),
    ('_negate', polars_compiler._negate, duck_compiler._negate),
    (
        '_propagate_absence',
        polars_compiler._propagate_absence,
        (duck_compiler._propagate_absence, duck_compiler.restrict_to),
    ),
    ('_map_fragments', polars_compiler._map_fragments, duck_compiler._map_fragments),
    ('_compare', polars_compiler._compare, duck_compiler._compare),
    ('labels (all paths)', polars_labels.Labeller.frame, duck_executor.DuckExecutor._label_frame),
    ('_factored', polars_labels.Labeller._factored, None),
    (
        '_build_variable',
        polars_executor.PolarsExecutor._build_variable,
        (duck_executor.DuckExecutor._build_variable, duck_executor.DuckExecutor._sorted_bounds),
    ),
    (
        '_build_constraint',
        polars_executor.PolarsExecutor._build_constraint,
        (
            duck_executor.DuckExecutor._build_constraint,
            duck_executor.DuckExecutor._constant_side,
            duck_executor.DuckExecutor._matrix_side,
        ),
    ),
    ('_build_objective', polars_executor.PolarsExecutor._build_objective, duck_executor.DuckExecutor._build_objective),
]


def code_lines(fn: Any) -> int:
    """Source lines that are neither blank, comment, nor docstring.

    A tuple is one operator an engine spread over named helpers, counted
    together — otherwise splitting a function reads here as a saving.
    """
    if fn is None:
        return 0
    if isinstance(fn, tuple):
        return sum(code_lines(part) for part in fn)
    src = textwrap.dedent(inspect.getsource(fn))
    tree = ast.parse(src)
    doc_spans = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                doc_spans.update(range(body[0].lineno, (body[0].end_lineno or body[0].lineno) + 1))
    out = 0
    for i, raw in enumerate(src.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith('#') or i in doc_spans:
            continue
        out += 1
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--show', help='print both implementations of one operator')
    opts = ap.parse_args(argv)

    if opts.show:
        for name, left, right in PAIRS:
            if name == opts.show or getattr(left, '__name__', '') == opts.show:
                for label, fn in (('polars', left), ('duckdb', right)):
                    print(f'{"=" * 30} {label} {"=" * 30}')
                    parts = fn if isinstance(fn, tuple) else (fn,)
                    print('\n'.join(inspect.getsource(p) for p in parts) if fn else '(no counterpart)')
                return 0
        print(f'no operator named {opts.show!r}')
        return 1

    rows = [(n, code_lines(a), code_lines(b)) for n, a, b in PAIRS]
    width = max(len(r[0]) for r in rows)
    print(f'{"operator".ljust(width)}  polars  duckdb   delta')
    print('-' * (width + 26))
    for name, a, b in rows:
        delta = '—' if b == 0 else f'{b - a:+d}'
        print(f'{name.ljust(width)}  {a:>6}  {b or "—":>6}  {delta:>6}')
    ta, tb = sum(r[1] for r in rows), sum(r[2] for r in rows)
    print('-' * (width + 26))
    print(f'{"total".ljust(width)}  {ta:>6}  {tb:>6}  {tb - ta:>+6}')
    print(f'\nduckdb is {tb / ta:.2f}x the polars line count over the operators both implement.')
    print('Code lines only — blank, comment and docstring lines excluded on both sides.')
    return 0


if __name__ == '__main__':
    sys.exit(main())

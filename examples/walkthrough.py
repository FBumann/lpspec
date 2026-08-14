"""The whole architecture, one model, one stage at a time — run it and read.

docs/ARCHITECTURE.md describes the pipeline; this script *executes* it one stage at
a time and prints the artifact each stage produces. Nothing here is a
reimplementation: every call is the same public entry point ``lps.solve`` takes
internally, so what you see is what actually runs.

    uv run python examples/walkthrough.py

Its output is committed as ``examples/walkthrough.out`` and asserted line for
line by ``tests/test_walkthrough.py``, so the narration cannot go stale
unnoticed: a stage that starts saying something else fails CI, and the diff of
the regenerated file is the record of what changed. Everything printed is
therefore deterministic.

The point it is trying to make is the thesis in docs/ARCHITECTURE.md: a YAML math
spec is a closed AST known before any data is touched. Stages 1-3 happen with
no data bound at all; only stage 4 sees a number.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

import polars as pl

import lpspec as lps
from lpspec.language.expansion import parse_and_expand
from lpspec.language.expression_parser import parse_expression
from lpspec.lowering import lower_program
from lpspec.relational.engines.polars.engine import PolarsEngine
from lpspec.sources import tidy_sources

HERE = Path(__file__).parent
MODEL = HERE / 'walkthrough.yaml'

#: Six snapshots of demand against four generators. Small enough to print.
GENERATORS = ['wind', 'solar', 'gas', 'oil']
SOURCES = {
    'p_max': pl.DataFrame({'generator': GENERATORS, 'value': [100.0, 60.0, 200.0, 0.0]}),
    'load': pl.DataFrame({'snapshot': range(6), 'value': [80.0, 120.0, 150.0, 180.0, 140.0, 100.0]}),
    'cost': pl.DataFrame({'generator': GENERATORS, 'value': [0.0, 0.0, 50.0, 80.0]}),
}
COORDS = {'snapshot': range(6)}

#: Two ways out of the language, caught at two different stages (see stage 7).
_REFUSED = [
    (
        'a helper that is not in the closed built-in set',
        {
            'constraints': {
                'averaged': {
                    'foreach': ['snapshot'],
                    'expression': 'mean(total_supply) <= load',
                }
            }
        },
    ),
    (
        'variable x variable — above the degree-1 ceiling',
        {'objectives': {'total_cost': {'sense': 'minimize', 'expression': 'sum(p * p, over=generator)'}}},
    ),
]


def banner(n: int, title: str, module: str) -> None:
    print(f'\n{_bold(f"[{n}] {title}")}  {_dim(f"({module})")}')


def main() -> None:
    print(__doc__.split('\n\n')[0])
    print(f'\nmodel: {MODEL.relative_to(HERE.parent)}')

    schema = validated_model()
    expanded_ast(schema)
    program = relational_ir(schema)
    with PolarsEngine() as engine:
        model_frames(engine, schema, program)
        lp_file(engine)
        solution(engine)
    refusals()

    print(f'\n{_dim("docs/ARCHITECTURE.md has the rules these stages enforce.")}')


def validated_model() -> lps.Model:
    """Stage 1 — YAML text to a validated model.

    Parses the file, type-checks it against the pydantic schema, and
    name-checks every expression, where string, named expression and macro
    template, used or not. After this call the model is known to be
    well-formed; no data has been touched.
    """
    banner(1, 'YAML text -> validated Model', 'schema.py, validation.py')
    schema = lps.load_model(MODEL)
    print(f'    dimensions   {", ".join(schema.dimensions)}')
    print(f'    parameters   {", ".join(schema.parameters)}')
    print(f'    variables    {", ".join(schema.variables)}')
    print(f'    constraints  {", ".join(schema.constraints)}')
    print(f'    expressions  {", ".join(schema.expressions)}    <- tier 2, still present')
    print(f'    macros       {", ".join(schema.macros)}   <- tier 2, still present')
    return schema


def expanded_ast(schema: lps.Model) -> None:
    """Stage 2 — macros and named expressions substituted away.

    Hard rule 1: the core AST is the whole language. Everything above it is
    pure substitution, which is why a macro costs nothing and cannot make the
    two lanes disagree — neither lane ever sees one.
    """
    banner(2, 'expand macros / named expressions -> core AST', 'expansion.py')
    objective_text = schema.objectives['total_cost'].expression
    print(f'    written      {objective_text!r}')
    print(f'    parsed       {parse_expression(objective_text)}')
    print(f'    expanded     {parse_and_expand(objective_text, schema)}')
    print('                 ^ the macro is gone: sum(p * cost, over=generator)')


def relational_ir(schema: lps.Model) -> Any:
    """Stage 3 — the core AST lowered to the relational plan.

    This is where the language's boundary is *decided*, by attempting the
    lowering, so eligibility can never drift from what the backend supports. It
    needs no data, which is what makes ``lps.check()`` a CI verb for model
    repositories: compile the math, bind nothing.
    """
    banner(3, 'core AST -> relational IR', 'lowering.py')
    program = lower_program(schema)
    print('    Program(')
    for decl in (*program.variables, *program.constraints):
        print(f'      {decl}')
    print(f'      {program.objective}')
    print('    )')
    print('    ^ frozen dataclasses, no macro, no YAML, no linopy, no engine')
    return program


def model_frames(engine: PolarsEngine, schema: lps.Model, program: Any) -> None:
    """Stage 4 — plan plus data to the model frames, the first stage to see a number.

    Sources are adapted to tidy frames (dims..., value) and the engine
    assembles the model from them.

    The private attributes read here are the one place this script reaches past
    the public API: the frames are engine-private by design (hard rule 1 — the
    plan is internal and the query is backend-private), and looking at them is
    the whole point.
    """
    banner(4, 'plan + data -> the model frames', 'relational/engines/polars/engine.py')
    engine.build(program, tidy_sources(schema, SOURCES, COORDS))
    model = engine._tables()
    for name, frame in (
        ('cols', model.cols),
        ('obj', model.obj),
        ('rows', model.rows),
        ('A', model.matrix),
    ):
        print(f'    {name:<20} {frame.height:>4} rows')
    print('\n    cols/rows/A/obj = the LP itself, in COO form:')
    print('    a column is a bound and a cost, a row a sense and a rhs,')
    print('    and A is every nonzero coefficient as (row, col, coeff).')

    variables = engine._variables['p'].collect()
    n_full = 6 * 4
    print('\n    where "p_max > 0" is not a mask array — it is row absence:')
    print(f'    p has {variables.height} rows, not {n_full}: retired oil never becomes a column.')
    print(_indent(variables.sort('var_label').head(4)))


def lp_file(engine: PolarsEngine) -> None:
    """Stage 5 — the same frames, second sink.

    The other one (the ``highs`` solver, stage 6) hands COO batches to highspy
    without ever forming a full CSR here.
    """
    banner(5, 'sink: stream the frames to an LP file', 'relational/sinks/lp_file.py')
    with tempfile.TemporaryDirectory() as tmp:
        lp = Path(tmp) / 'model.lp'
        engine.write(lp)
        text = lp.read_text().splitlines()
        print(_indent('\n'.join(text[:12])))
        print(f'    ... ({len(text)} lines total)')


def solution(engine: PolarsEngine) -> None:
    """Stage 6 — batches to highspy, and the solution read back by label join.

    Never densified. ``primal()`` hands back a frame: the engine's own shape,
    and the one a caller can pass on without this package depending on their
    library. Sorted here because a join's output order is the planner's
    business, not a fact about the architecture this file is narrating.
    """
    banner(6, 'sink: batches -> highspy -> solution frames', 'relational/sinks/highs.py')
    result = engine.solve()
    print(f'    status     {result.status} ({result.termination_condition})')
    print(f'    objective  {result.objective:,.1f}')
    print(_indent(result.primal('p').sort('snapshot', 'generator').head(6)))


def refusals() -> None:
    """Stage 7 — what the language refuses, and which stage catches it.

    Every rejection is a product statement: the error names the construct and
    its rewrite. Never a silent fallback, never a redirect to the other lane —
    both lanes accept exactly the same language (hard rule 3).

    Each model is run through ``lps.check()`` — stages 1-3, no data bound — and
    then, only if that passes, through a build. Both are caught by ``check()``,
    which is what makes it a CI verb: a model repository can compile-check its
    math with no data in the runner. The build arm stays because which stage
    catches what is a real property of the design, and printing it is how this
    script would tell you if that changed.

    ``LanguageError`` is a ``ValueError`` subclass, so that is what is caught.
    """
    banner(7, 'and what the language refuses', 'validation.py, lowering.py')
    for label, patch in _REFUSED:
        print(f'\n    {label}:')
        model = {**_raw(MODEL), **patch}
        try:
            lps.check(model)
        except ValueError as exc:
            _refusal('check()', exc)
            continue
        try:
            lps.build(model, SOURCES, coords=COORDS).close()
        except ValueError as exc:
            _refusal('build()', exc)


def _refusal(verb: str, exc: Exception) -> None:
    print(f'    {_dim(f"caught by {verb} as {type(exc).__name__}")}')
    print(_indent(str(exc), '      '))


def _bold(text: str) -> str:
    return f'\033[1m{text}\033[0m' if sys.stdout.isatty() else text


def _dim(text: str) -> str:
    return f'\033[2m{text}\033[0m' if sys.stdout.isatty() else text


def _indent(obj: object, pad: str = '    ') -> str:
    return '\n'.join(pad + line for line in str(obj).splitlines())


def _raw(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text())


if __name__ == '__main__':
    main()

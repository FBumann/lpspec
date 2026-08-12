"""The engines, and how a name becomes one.

Everything in here implements `relational.engine.Engine`; everything above it
in `relational/` is what an implementation answers to.

**The set is closed.** No `register_engine()`, for the same reason the sinks
have no `register_sink()`: an installed package that can change which engine
`lps.build` uses is hard rule 5's failure mode one level down — what a model
costs would depend on Python-side state the file cannot see. Adding an engine
is a pull request here.

Resolution is lazy: the import is the expensive part, so an engine nobody
selects costs nothing to have named here.

**Both engines ship, and both are installed.** Neither is behind an extra, for
the same reason `DEFAULT_ENGINE` is not: an engine a plain `pip install lpspec`
cannot select is not a choice a caller has, and a default nobody has installed
is not a default. What having both costs is two packages — `duckdb`, and
`pyarrow` for the frames it and polars hand each other — paid once at install
rather than guarded at every use.

**`LPSPEC_ENGINE` is the only way to choose**, and `lps.build` takes no engine
parameter. That is deliberate rather than minimal: the engines build the same
model integer for integer (`tests/test_engine_parity.py`), so the choice cannot
change the answer — only what computing it costs. `coords` belongs in the call
because it decides *what model is built*; an engine decides nothing, and a knob
that cannot change the answer does not belong in the signature that produces
one.

It also keeps the choice where it belongs. Which engine suits a machine is an
operational fact about that machine, and committing it into the code that
describes the math couples the two. An environment can say "run everything on
polars here" without touching a line.

This is not the session state hard rule 4 forbids: that rule is about
Python-side state changing what a file *means*, and nothing here can.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lpspec.relational.engine import Engine

#: name → ``module:attribute``. The default is first, and is the one every
#: published benchmark and every unasked CI measurement is taken on.
ENGINES: dict[str, str] = {
    'duckdb': 'lpspec.relational.engines.duck.executor:DuckExecutor',
    'polars': 'lpspec.relational.engines.polars.executor:PolarsExecutor',
}

DEFAULT_ENGINE = 'duckdb'

#: The only switch. Unset means `DEFAULT_ENGINE`.
ENV_VAR = 'LPSPEC_ENGINE'


def resolve(engine: str | None = None) -> type[Engine]:
    """The engine class to build with: *engine*, else `LPSPEC_ENGINE`, else the default.

    The argument exists for the tests and the benchmark harness, which need a
    named engine without an environment; nothing on the public path passes it.

    An `ImportError` out of here is a broken install, not a missing extra —
    both engines' packages are runtime dependencies — so it is left to
    propagate with the module name the interpreter already put in it.
    """
    import importlib

    from_env = False
    if engine is None:
        engine = os.environ.get(ENV_VAR) or DEFAULT_ENGINE
        from_env = engine != DEFAULT_ENGINE
    if engine not in ENGINES:
        known = ', '.join(repr(n) for n in ENGINES)
        # naming the *source* matters here: an unknown name in the environment
        # is a typo in a shell profile, and reads as a library bug otherwise
        where = f' (from {ENV_VAR})' if from_env else ''
        msg = f'unknown engine {engine!r}{where} — available: {known}'
        raise ValueError(msg)
    module, _, attribute = ENGINES[engine].partition(':')
    return getattr(importlib.import_module(module), attribute)

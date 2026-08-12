"""Sinks: how a built model leaves the engine. See README.md.

**Two families.** A *solver* takes the tables and runs them (``solvers/``,
chosen by name); a *writer* renders them to a file (``writers/``, chosen by
suffix). They are directories rather than a convention, so
``tests/test_architecture.py`` reads membership off the path.

``tables.py`` is what both read, and neither family imports the other.
"""

from lpspec.relational.sinks.solvers import SOLVERS, solver
from lpspec.relational.sinks.tables import COLS, DTYPES, MATRIX, OBJ, ROWS, VTYPE, ModelTables, compress_rows
from lpspec.relational.sinks.writers import PLANNED_WRITERS, WRITERS, writer

__all__ = [
    'COLS',
    'DTYPES',
    'MATRIX',
    'OBJ',
    'PLANNED_WRITERS',
    'ROWS',
    'SOLVERS',
    'VTYPE',
    'WRITERS',
    'ModelTables',
    'compress_rows',
    'solver',
    'writer',
]

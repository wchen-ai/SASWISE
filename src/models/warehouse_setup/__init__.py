"""Warehouse setup package — model analysis and block management.

This package analyses a neural network and organises it into blocks,
and provides the high-level Python API used by ``main.py``,
``model_assembler.py``, and the test suite.

CLI entry points
----------------

1. Generate hierarchy::

       python -m src.models.warehouse_setup.generate_hierarchy --state_dict <path> --out <dir>

2. Block analysis::

       python -m src.models.warehouse_setup.block_analysis --model_hierarchy <path> --out <dir>

3. Create variants::

       python -m src.models.warehouse_setup.create_variant --block_analysis <path> --state_dict <path> --out <dir>

Python API
----------

The names below are re-exported from :mod:`src.models.warehouse_core`,
the historical implementation that used to live next to this package
as ``kitchen_setup.py``. Python's importer prefers a regular package
over a sibling module of the same name, which silently shadowed the
file; it has since been renamed to ``warehouse_core.py`` so that the
re-exports below resolve correctly.

The package and the function ``build_warehouse`` deliberately have
different names so that ``from src.models.warehouse_setup import
build_warehouse`` is unambiguous.
"""

from .hierarchy.build_hierarchy import build_model_hierarchy_from_state_dict
from .hierarchy.save_hierarchy import save_model_hierarchy

from ..warehouse_core import (
    ModelHierarchy,
    analyze_block_parameters,
    build_warehouse,
    parse_user_block_indexing,
    save_block_analysis,
)

__all__ = [
    "ModelHierarchy",
    "analyze_block_parameters",
    "build_model_hierarchy_from_state_dict",
    "build_warehouse",
    "parse_user_block_indexing",
    "save_block_analysis",
    "save_model_hierarchy",
]

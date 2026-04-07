"""Backwards-compatible shim.

All real metadata, dependencies, and tool configuration live in
``pyproject.toml`` (PEP 621). This file exists only so that legacy
tooling that still invokes ``python setup.py ...`` continues to work.
"""

from setuptools import setup

setup()

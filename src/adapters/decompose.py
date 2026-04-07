"""Decomposition strategies — split a model into blocks.

A block is a non-overlapping subset of the model's ``state_dict`` keys
(parameters and buffers). The full set of blocks must cover every
``state_dict`` key exactly once.

Three strategies are provided:

* :func:`decompose_balanced` — partition the keys in iteration order
  into ``num_blocks`` groups with approximately equal parameter
  counts. Works for any architecture; default for the
  ``adopt_checkpoint`` CLI.

* :func:`decompose_top_level` — every direct child of the model is its
  own block. Natural for ResNets, U-Nets, and other architectures with
  a small number of named children.

* :func:`decompose_manual` — the user supplies a list of regex
  patterns; each ``state_dict`` key is assigned to the first matching
  block.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, List

import torch
from torch import nn


@dataclass
class BlockSpec:
    """A single block in the warehouse.

    Attributes
    ----------
    id : int
        Zero-based block index.
    name : str
        Human-readable label (e.g. ``"layer3"`` or ``"block_2"``).
    state_dict_keys : list[str]
        Keys from ``model.state_dict()`` that belong to this block.
    num_params : int
        Total element count across all keys.
    """

    id: int
    name: str
    state_dict_keys: List[str] = field(default_factory=list)
    num_params: int = 0


def _state_dict_sizes(model: nn.Module) -> List[tuple[str, int]]:
    """Return ``[(key, numel), …]`` in ``state_dict`` iteration order."""
    return [(k, v.numel()) for k, v in model.state_dict().items()]


def decompose_balanced(model: nn.Module, num_blocks: int) -> List[BlockSpec]:
    """Partition the model into ``num_blocks`` roughly equal-sized blocks.

    Walks ``model.state_dict()`` in iteration order and greedily
    accumulates keys into the current block until its parameter count
    is at least ``total / num_blocks``, then opens the next block.
    """
    if num_blocks < 1:
        raise ValueError(f"num_blocks must be >= 1, got {num_blocks}")

    items = _state_dict_sizes(model)
    if not items:
        raise ValueError("Model has no parameters or buffers in its state_dict")

    total = sum(n for _, n in items)
    target = max(1, total // num_blocks)

    blocks: List[BlockSpec] = []
    current_keys: List[str] = []
    current_count = 0
    block_id = 0

    for key, count in items:
        current_keys.append(key)
        current_count += count
        # Close the current block if we are at or past the target and we
        # still have at least one block left to open.
        if current_count >= target and block_id < num_blocks - 1:
            blocks.append(
                BlockSpec(
                    id=block_id,
                    name=f"block_{block_id}",
                    state_dict_keys=current_keys,
                    num_params=current_count,
                )
            )
            block_id += 1
            current_keys = []
            current_count = 0

    # Whatever remains becomes the final block.
    if current_keys:
        blocks.append(
            BlockSpec(
                id=block_id,
                name=f"block_{block_id}",
                state_dict_keys=current_keys,
                num_params=current_count,
            )
        )

    # If the loop closed early (e.g. because target was very small)
    # there might be fewer than num_blocks blocks. That is acceptable
    # but we report it so the user knows.
    return blocks


def decompose_top_level(model: nn.Module) -> List[BlockSpec]:
    """Make every direct child of ``model`` a block.

    State-dict keys not owned by any direct child (rare; e.g. raw
    parameters defined on the root module) are appended to the last
    block.
    """
    children = list(model.named_children())
    if not children:
        raise ValueError("Model has no direct children to use as blocks")

    state_dict = model.state_dict()
    assigned: set[str] = set()
    blocks: List[BlockSpec] = []

    for block_id, (child_name, _child) in enumerate(children):
        prefix = f"{child_name}."
        keys = [k for k in state_dict.keys() if k == child_name or k.startswith(prefix)]
        num = sum(state_dict[k].numel() for k in keys)
        assigned.update(keys)
        blocks.append(
            BlockSpec(
                id=block_id,
                name=child_name,
                state_dict_keys=keys,
                num_params=num,
            )
        )

    # Sweep up anything that did not match a top-level child (e.g.
    # parameters registered directly on ``model``).
    leftover = [k for k in state_dict.keys() if k not in assigned]
    if leftover:
        blocks[-1].state_dict_keys.extend(leftover)
        blocks[-1].num_params += sum(state_dict[k].numel() for k in leftover)

    return blocks


def decompose_manual(model: nn.Module, patterns: Iterable[str]) -> List[BlockSpec]:
    """Assign each state-dict key to the first matching regex.

    Parameters
    ----------
    patterns : Iterable[str]
        Regex patterns, one per block. The order defines the block
        ids.

    Raises
    ------
    ValueError
        If any state-dict key matches none of the patterns.
    """
    compiled = [re.compile(p) for p in patterns]
    if not compiled:
        raise ValueError("decompose_manual requires at least one pattern")

    state_dict = model.state_dict()
    block_keys: dict[int, List[str]] = {i: [] for i in range(len(compiled))}

    for key in state_dict.keys():
        for i, pat in enumerate(compiled):
            if pat.search(key):
                block_keys[i].append(key)
                break
        else:
            raise ValueError(
                f"State-dict key {key!r} did not match any block pattern. "
                f"Add a catch-all pattern such as '.*' as the last entry."
            )

    blocks: List[BlockSpec] = []
    for i, keys in block_keys.items():
        num = sum(state_dict[k].numel() for k in keys)
        blocks.append(
            BlockSpec(
                id=i,
                name=f"block_{i}",
                state_dict_keys=keys,
                num_params=num,
            )
        )
    return blocks

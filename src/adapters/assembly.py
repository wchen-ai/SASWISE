"""Build a sub-model by grafting variants from a warehouse into a fresh model.

A *block configuration* is a ``dict[int, int]`` mapping ``block_id`` to
``variant_id``. Assembling a sub-model means:

1. Call ``model_factory()`` to obtain a fresh ``nn.Module``.
2. For each ``(block_id, variant_id)`` in the configuration, load the
   variant's state-dict slice from the warehouse and copy its tensors
   into the corresponding parameters/buffers of the model.

Only one variant slice is held in memory at a time, plus the assembled
model itself. Total memory cost is therefore the size of the model.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import torch
from torch import nn

from .disk_warehouse import DiskWarehouse


def assemble_submodel(
    model_factory: Callable[[], nn.Module],
    block_config: Dict[int, int],
    warehouse: DiskWarehouse,
    map_location: str | torch.device = "cpu",
    base_state_dict: Optional[Dict[str, torch.Tensor]] = None,
) -> nn.Module:
    """Construct a single sub-model from a block configuration.

    Parameters
    ----------
    model_factory
        Zero-argument callable returning a fresh ``nn.Module``.
    block_config
        ``{block_id: variant_id}``. Every block in the warehouse must
        appear in the dict.
    warehouse
        The on-disk variant store.
    map_location
        Device to load variant tensors onto. Use the GPU device when
        you intend to immediately move the model there.
    base_state_dict
        Optional fallback weights for any state-dict key not covered
        by any block. Useful when the decomposition deliberately
        excludes some parameters from variantisation (e.g. frozen
        normalisation layers).
    """
    if set(block_config.keys()) != {s.id for s in warehouse.block_specs}:
        missing = {s.id for s in warehouse.block_specs} - set(block_config.keys())
        extra = set(block_config.keys()) - {s.id for s in warehouse.block_specs}
        raise ValueError(
            f"block_config must cover every block exactly once. "
            f"Missing: {sorted(missing)}, extra: {sorted(extra)}"
        )

    model = model_factory()

    if base_state_dict is not None:
        model.load_state_dict(base_state_dict, strict=False)

    # Build the merged state dict for the requested configuration.
    merged: Dict[str, torch.Tensor] = {}
    for block_id, variant_id in block_config.items():
        slice_state = warehouse.load_variant(block_id, variant_id, map_location=map_location)
        merged.update(slice_state)

    # ``strict=False`` because some non-floating tensors (BN
    # num_batches_tracked) may not be in the slice if the user excluded
    # them, and the model may have additional registered tensors.
    missing, unexpected = model.load_state_dict(merged, strict=False)
    if unexpected:
        raise RuntimeError(
            f"Variant state contained unexpected keys absent from the model: "
            f"{unexpected[:5]}{'…' if len(unexpected) > 5 else ''}"
        )
    return model

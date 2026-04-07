"""Materialise initial variants by cloning the base checkpoint and perturbing.

For each block ``b`` and each variant index ``v``::

    variant_0[k] = base_state_dict[k]                          (verbatim)
    variant_v[k] = base_state_dict[k] + N(0, σ · mean|w|)      (v >= 1)

Variant ``0`` is always the unmodified base, so a freshly-initialised
warehouse already produces the original checkpoint as one of its sub-models.
"""

from __future__ import annotations

from typing import List

import torch
from torch import nn

from .decompose import BlockSpec
from .disk_warehouse import DiskWarehouse


def initialize_variants(
    model: nn.Module,
    block_specs: List[BlockSpec],
    num_variants: int,
    warehouse: DiskWarehouse,
    noise_std: float = 1e-3,
    seed: int = 0,
) -> None:
    """Write ``num_variants`` initial variants per block to ``warehouse``.

    Parameters
    ----------
    model
        The base model with its pretrained weights already loaded.
    block_specs
        Output of one of the ``decompose_*`` functions.
    num_variants
        Number of variants per block. Must match ``warehouse.num_variants``.
    warehouse
        Destination ``DiskWarehouse``. Must already be created on disk.
    noise_std
        Multiplicative scale of the Gaussian perturbation. The actual
        per-tensor noise is ``noise_std * mean(|w|) * randn``.
    seed
        Generator seed; the same value yields reproducible variants.
    """
    if num_variants != warehouse.num_variants:
        raise ValueError(
            f"num_variants ({num_variants}) does not match "
            f"warehouse.num_variants ({warehouse.num_variants})"
        )

    base_state = model.state_dict()
    generator = torch.Generator(device="cpu").manual_seed(seed)

    for spec in block_specs:
        for variant_id in range(num_variants):
            slice_state = {}
            for key in spec.state_dict_keys:
                w = base_state[key].detach().cpu()
                if variant_id == 0 or noise_std == 0.0:
                    slice_state[key] = w.clone()
                    continue
                # Skip non-floating tensors (BN num_batches_tracked, etc.).
                if not w.is_floating_point():
                    slice_state[key] = w.clone()
                    continue
                scale = noise_std * w.abs().mean().clamp_min(1e-8)
                noise = torch.randn(w.shape, generator=generator, dtype=w.dtype) * scale
                slice_state[key] = w + noise
            warehouse.save_variant(spec.id, variant_id, slice_state)

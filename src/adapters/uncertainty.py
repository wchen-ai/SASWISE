"""Streaming ensemble inference with uncertainty quantification.

For each sampled block configuration the script:

1. Assembles a sub-model on the GPU.
2. Runs the user's ``infer_fn`` over the validation loader.
3. Discards the sub-model and folds the output into a running mean
   and running second-moment via Welford's algorithm.

Only **one** sub-model is in GPU memory at any time. The full
per-configuration outputs are *never* stored — memory cost is
``O(output_size)`` regardless of how many configurations are sampled.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from types import ModuleType
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from ._internal import to_device
from .assembly import assemble_submodel
from .disk_warehouse import DiskWarehouse


@dataclass
class EnsemblePrediction:
    """Aggregated ensemble prediction with uncertainty.

    Attributes
    ----------
    mean
        Running mean of per-configuration outputs, shape
        ``(N, …output_dims…)``.
    std
        Per-element standard deviation across configurations, same
        shape as ``mean``.
    predictive_entropy
        For classification only: entropy of the softmaxed mean.
        Shape ``(N,)``. ``None`` for non-classification tasks.
    num_configurations
        Number of sub-models actually evaluated.
    block_configurations
        The sampled configurations, in the order evaluated.
    """

    mean: torch.Tensor
    std: torch.Tensor
    num_configurations: int
    predictive_entropy: Optional[torch.Tensor] = None
    block_configurations: List[Dict[int, int]] = field(default_factory=list)


def _sample_configurations(
    warehouse: DiskWarehouse, num_configurations: int, seed: int = 0
) -> List[Dict[int, int]]:
    """Sample (or exhaustively enumerate) block configurations."""
    block_ids = [s.id for s in warehouse.block_specs]
    total = warehouse.num_configurations()

    if num_configurations >= total:
        return [
            dict(zip(block_ids, combo))
            for combo in itertools.product(
                range(warehouse.num_variants), repeat=warehouse.num_blocks
            )
        ]

    rng = random.Random(seed)
    configurations: List[Dict[int, int]] = []
    seen: set[tuple[int, ...]] = set()
    while len(configurations) < num_configurations:
        cfg = tuple(rng.randrange(warehouse.num_variants) for _ in block_ids)
        if cfg in seen:
            continue
        seen.add(cfg)
        configurations.append(dict(zip(block_ids, cfg)))
    return configurations


def _run_one_config(
    user_module: ModuleType,
    warehouse: DiskWarehouse,
    block_config: Dict[int, int],
    device: torch.device,
    base_state_dict: Optional[Dict[str, torch.Tensor]],
) -> torch.Tensor:
    """Assemble + evaluate a single sub-model. Returns the concatenated output."""
    model = assemble_submodel(
        user_module.model_factory, block_config, warehouse,
        map_location="cpu", base_state_dict=base_state_dict,
    ).to(device).eval()

    _, val_loader = user_module.get_dataloaders()
    pieces: List[torch.Tensor] = []
    with torch.no_grad():
        for batch in val_loader:
            batch_d = to_device(batch, device)
            out = user_module.infer_fn(model, batch_d)
            pieces.append(out.detach().cpu())
    full = torch.cat(pieces, dim=0)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return full


def infer_with_uncertainty(
    warehouse: DiskWarehouse,
    user_module: ModuleType,
    num_configurations: int = 64,
    task: str = "auto",
    device: str | torch.device = "cuda",
    seed: int = 0,
    base_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    show_progress: bool = True,
) -> EnsemblePrediction:
    """Run ensembled inference and return mean + uncertainty.

    Parameters
    ----------
    warehouse
        On-disk variant store to draw configurations from.
    user_module
        User-supplied module exposing ``model_factory``, ``get_dataloaders``,
        and ``infer_fn``.
    num_configurations
        Maximum number of sub-models to evaluate. If this is at least
        ``warehouse.num_configurations()``, the full Cartesian product
        is enumerated; otherwise that many distinct configurations are
        sampled uniformly without replacement.
    task
        ``"classification"`` enables predictive-entropy output;
        ``"regression"`` and ``"segmentation"`` produce only
        ``(mean, std)``; ``"auto"`` infers from output rank
        (rank-2 → classification).
    """
    device = torch.device(device)
    configurations = _sample_configurations(warehouse, num_configurations, seed=seed)

    running_mean: Optional[torch.Tensor] = None
    running_m2: Optional[torch.Tensor] = None
    n_seen = 0

    for i, cfg in enumerate(configurations):
        full_out = _run_one_config(user_module, warehouse, cfg, device, base_state_dict)

        n_seen += 1
        if running_mean is None:
            running_mean = full_out.clone()
            running_m2 = torch.zeros_like(full_out)
        else:
            delta = full_out - running_mean
            running_mean = running_mean + delta / n_seen
            delta2 = full_out - running_mean
            running_m2 = running_m2 + delta * delta2

        if show_progress:
            print(f"[{i + 1}/{len(configurations)}] cfg={cfg} -> shape={tuple(full_out.shape)}")

    assert running_mean is not None and running_m2 is not None
    variance = running_m2 / max(n_seen - 1, 1)
    std = variance.clamp_min(0).sqrt()

    pred = EnsemblePrediction(
        mean=running_mean,
        std=std,
        num_configurations=n_seen,
        block_configurations=configurations,
    )

    is_classification = task == "classification" or (
        task == "auto" and running_mean.dim() == 2
    )
    if is_classification:
        probs = F.softmax(running_mean, dim=-1)
        log_probs = probs.clamp_min(1e-12).log()
        pred.predictive_entropy = -(probs * log_probs).sum(dim=-1)

    return pred

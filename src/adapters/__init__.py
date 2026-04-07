"""SASWISE adapters — adopt arbitrary pre-trained checkpoints into a warehouse.

This subpackage exposes a small, model-agnostic API for the
"I have a pretrained checkpoint, just adopt it" use case:

1. **Decompose** the model into ``B`` blocks
   (:func:`decompose_balanced`, :func:`decompose_top_level`,
   :func:`decompose_manual`).
2. **Initialize** ``V`` variants per block by cloning the base parameters
   and adding small Gaussian noise (:func:`initialize_variants`).
3. **Fine-tune** the warehouse with the diversification consistency
   loss, holding only one sub-model in GPU memory at a time
   (:func:`finetune_warehouse`).
4. **Infer with uncertainty** by streaming a sample of block
   configurations through the validation set, accumulating mean and
   variance via Welford's algorithm (:func:`infer_with_uncertainty`).

The adapter does not modify any of the existing SASWISE-UE modules; it
is a clean alternative to the manual ``warehouse_setup`` /
``train_diversification`` / ``ensemble_evaluator`` pipeline for users
who want to point at a checkpoint and a Python file and call it a day.
"""

from .assembly import assemble_submodel
from .decompose import (
    BlockSpec,
    decompose_balanced,
    decompose_manual,
    decompose_top_level,
)
from .disk_warehouse import DiskWarehouse
from .finetune import ConsistencyBalance, finetune_warehouse
from .uncertainty import EnsemblePrediction, infer_with_uncertainty
from .variant_init import initialize_variants

__all__ = [
    "BlockSpec",
    "ConsistencyBalance",
    "DiskWarehouse",
    "EnsemblePrediction",
    "assemble_submodel",
    "decompose_balanced",
    "decompose_manual",
    "decompose_top_level",
    "finetune_warehouse",
    "infer_with_uncertainty",
    "initialize_variants",
]

"""Assemble-train-disassemble fine-tuning loop with adaptive consistency weighting.

Algorithm sketch::

    for epoch in 1..E:
        for batch in train_loader:
            λ      = balancer.current_weight()                 # warm-up + EMA-balanced
            cfg_a  = random configuration                      # update target
            cfg_b  = random configuration  (cfg_b ≠ cfg_a)     # consistency reference
            model_a = assemble_submodel(cfg_a)                 # trainable
            model_b = assemble_submodel(cfg_b)                 # frozen reference
            out_a   = infer_fn(model_a, batch)
            out_b   = infer_fn(model_b, batch).detach()
            loss    = loss_fn(out_a, batch) + λ · consistency_fn(out_a, out_b)
            loss.backward();  optimizer.step()
            for block_id in cfg_a:
                save the corresponding slice of model_a back to disk
            balancer.observe(loss_fn.item(), consistency_fn.item())
            del model_a, model_b
            torch.cuda.empty_cache()

Only ``model_a`` and ``model_b`` are ever resident in GPU memory.

The consistency weight ``λ`` is computed by a small
:class:`ConsistencyBalance` policy. By default it tracks an
exponentially-smoothed magnitude of each loss and sets ``λ`` so the
consistency term contributes a fixed fraction (``target_ratio``, default
0.1) of the total loss, after a linear warm-up of 100 steps. Pass a
plain ``float`` to recover the old constant-weight behaviour.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from types import ModuleType
from typing import Callable, Dict, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from ._internal import to_device
from .assembly import assemble_submodel
from .disk_warehouse import DiskWarehouse


# ---------------------------------------------------------------------------
# Consistency-loss balancing
# ---------------------------------------------------------------------------


@dataclass
class ConsistencyBalance:
    """Configuration for the consistency-loss weight schedule.

    The training objective is::

        L_total = L_task + λ · L_cons

    In **adaptive** mode (the default), ``λ`` is recomputed every step
    so that the consistency term contributes ``target_ratio`` of the
    total loss magnitude::

        λ = (target_ratio / (1 − target_ratio)) · EMA[L_task] / EMA[L_cons]

    A linear warm-up scales ``λ`` from zero to its adaptive value over
    the first ``warmup_steps`` batches, so the model fits the data
    primarily through ``L_task`` before the consistency term begins to
    pull sub-models toward each other.

    In **fixed** mode (``adaptive=False``), ``λ`` is the constant
    ``fixed_weight`` (still subject to the warm-up if
    ``warmup_steps > 0``).

    Attributes
    ----------
    adaptive
        Use loss-magnitude-aware ``λ`` if True; otherwise use a fixed
        ``λ``.
    target_ratio
        Target fraction of total loss contributed by consistency.
        Used only when ``adaptive=True``. Must lie strictly in
        ``(0, 1)``. Default: ``0.1`` (consistency ≈ 10% of total).
    fixed_weight
        Constant ``λ`` when ``adaptive=False``.
    warmup_steps
        Number of optimization steps over which ``λ`` linearly ramps
        from 0 to its target. Set to 0 to disable the warm-up.
    ema_decay
        Smoothing factor for the loss magnitude estimates. Higher
        values produce smoother but slower-reacting weights.
    min_weight, max_weight
        Hard bounds on ``λ``. The adaptive value is clipped to this
        interval to prevent runaway scaling under degenerate losses.
    """

    adaptive: bool = True
    target_ratio: float = 0.1
    fixed_weight: float = 0.1
    warmup_steps: int = 100
    ema_decay: float = 0.9
    min_weight: float = 0.0
    max_weight: float = 100.0

    def __post_init__(self) -> None:
        if self.adaptive and not (0.0 < self.target_ratio < 1.0):
            raise ValueError(
                f"target_ratio must be in (0, 1) when adaptive=True, got {self.target_ratio}"
            )
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError(f"ema_decay must be in [0, 1), got {self.ema_decay}")
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {self.warmup_steps}")
        if self.min_weight < 0 or self.max_weight < self.min_weight:
            raise ValueError(
                f"Bounds must satisfy 0 <= min_weight <= max_weight, "
                f"got min={self.min_weight}, max={self.max_weight}"
            )


class _ConsistencyBalancer:
    """Lag-1 EMA tracker that produces the current consistency weight.

    The "lag-1" pattern means: ``current_weight()`` reads the EMAs as
    they were *before* the most recent observation, so the weight
    applied to step ``t``'s gradient is computed from the loss
    statistics of steps ``0..t-1``. This avoids using a loss to scale
    its own gradient.
    """

    def __init__(self, config: ConsistencyBalance) -> None:
        self.config = config
        self._task_ema: Optional[float] = None
        self._cons_ema: Optional[float] = None
        self._step = 0

    def current_weight(self) -> float:
        cfg = self.config
        if not cfg.adaptive:
            base = cfg.fixed_weight
        elif self._task_ema is None or self._cons_ema is None:
            base = 0.0
        else:
            ratio = cfg.target_ratio
            base = (ratio / (1.0 - ratio)) * (self._task_ema / max(self._cons_ema, 1e-12))
            base = max(cfg.min_weight, min(cfg.max_weight, base))

        if cfg.warmup_steps > 0:
            warmup = min(1.0, self._step / cfg.warmup_steps)
        else:
            warmup = 1.0
        return float(warmup * base)

    def observe(self, task_loss: float, cons_loss: float) -> None:
        cfg = self.config
        decay = cfg.ema_decay
        if self._task_ema is None:
            self._task_ema = float(task_loss)
            self._cons_ema = float(cons_loss)
        else:
            self._task_ema = decay * self._task_ema + (1.0 - decay) * float(task_loss)
            self._cons_ema = decay * self._cons_ema + (1.0 - decay) * float(cons_loss)
        self._step += 1

    @property
    def step(self) -> int:
        return self._step


# ---------------------------------------------------------------------------
# Default consistency loss (overridable by user_module.consistency_fn)
# ---------------------------------------------------------------------------


def _default_consistency_fn(out_a: torch.Tensor, out_b: torch.Tensor) -> torch.Tensor:
    """Default consistency loss: MSE between raw outputs.

    Robust across classification (logits), regression, and segmentation.
    For classification specifically, KL of softmaxed outputs is sharper;
    define ``consistency_fn`` in your user module if you want it.
    """
    return F.mse_loss(out_a, out_b)


def _sample_distinct_configurations(
    block_ids: list[int], num_variants: int
) -> tuple[Dict[int, int], Dict[int, int]]:
    """Return two random configurations that differ in at least one block."""
    cfg_a = {bid: random.randrange(num_variants) for bid in block_ids}
    cfg_b = {bid: random.randrange(num_variants) for bid in block_ids}
    if num_variants > 1 and cfg_a == cfg_b:
        flip_block = random.choice(block_ids)
        cfg_b[flip_block] = (cfg_b[flip_block] + 1) % num_variants
    return cfg_a, cfg_b


def _save_block_slice_from_model(
    model: nn.Module,
    spec_keys: list[str],
    warehouse: DiskWarehouse,
    block_id: int,
    variant_id: int,
) -> None:
    state_dict = model.state_dict()
    slice_state = {k: state_dict[k].detach().cpu() for k in spec_keys}
    warehouse.save_variant(block_id, variant_id, slice_state)


def finetune_warehouse(
    warehouse: DiskWarehouse,
    user_module: ModuleType,
    epochs: int = 5,
    lr: float = 1e-4,
    consistency: Union[ConsistencyBalance, float, None] = None,
    device: Union[str, torch.device] = "cuda",
    log_every: int = 50,
    base_state_dict: Optional[Dict[str, torch.Tensor]] = None,
) -> None:
    """Fine-tune every variant in the warehouse with the diversification loss.

    Parameters
    ----------
    warehouse
        The on-disk variant store to update in place.
    user_module
        A user-supplied Python module (loaded by
        :func:`adapters._internal.load_user_module`) that exposes
        ``model_factory``, ``get_dataloaders``, ``infer_fn``, and
        ``loss_fn``. May optionally expose ``consistency_fn``;
        otherwise the default MSE consistency is used.
    epochs
        Number of passes over the training set.
    lr
        Adam learning rate. A fresh optimiser is created for every
        sampled configuration; persistent optimiser state across
        configurations would defeat block isolation.
    consistency
        Consistency-loss balancing policy. Accepts either a
        :class:`ConsistencyBalance` instance, a plain ``float``
        (interpreted as a fixed ``λ``), or ``None`` (the default,
        which uses ``ConsistencyBalance()`` — adaptive λ aiming for
        ``target_ratio=0.1`` of the total loss with a 100-step
        linear warm-up).
    device
        Device to assemble sub-models on.
    log_every
        Print one line every ``log_every`` batches.
    base_state_dict
        Optional fallback weights for state-dict keys not covered by
        any block (forwarded to :func:`assemble_submodel`).
    """
    if consistency is None:
        consistency = ConsistencyBalance()
    elif isinstance(consistency, (int, float)):
        consistency = ConsistencyBalance(adaptive=False, fixed_weight=float(consistency))
    elif not isinstance(consistency, ConsistencyBalance):
        raise TypeError(
            "consistency must be ConsistencyBalance, float, or None; "
            f"got {type(consistency).__name__}"
        )
    balancer = _ConsistencyBalancer(consistency)

    train_loader, _ = user_module.get_dataloaders()
    consistency_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = getattr(
        user_module, "consistency_fn", _default_consistency_fn
    )

    block_ids = [s.id for s in warehouse.block_specs]
    num_variants = warehouse.num_variants
    device = torch.device(device)

    step = 0
    for epoch in range(epochs):
        for batch in train_loader:
            # Lag-1 weight: derived from the EMAs as of the previous step.
            weight = balancer.current_weight()

            cfg_a, cfg_b = _sample_distinct_configurations(block_ids, num_variants)

            # Trainable model_a.
            model_a = assemble_submodel(
                user_module.model_factory, cfg_a, warehouse,
                map_location="cpu", base_state_dict=base_state_dict,
            ).to(device).train()
            optim = torch.optim.Adam(model_a.parameters(), lr=lr)

            # Frozen reference model_b.
            with torch.no_grad():
                model_b = assemble_submodel(
                    user_module.model_factory, cfg_b, warehouse,
                    map_location="cpu", base_state_dict=base_state_dict,
                ).to(device).eval()

            batch_d = to_device(batch, device)
            out_a = user_module.infer_fn(model_a, batch_d)
            with torch.no_grad():
                out_b = user_module.infer_fn(model_b, batch_d)

            task_loss = user_module.loss_fn(out_a, batch_d)
            cons_loss = consistency_fn(out_a, out_b)
            loss = task_loss + weight * cons_loss

            optim.zero_grad()
            loss.backward()
            optim.step()

            # Persist only the variants we just updated.
            for block_id, variant_id in cfg_a.items():
                spec = warehouse.block_by_id(block_id)
                _save_block_slice_from_model(
                    model_a, spec.state_dict_keys, warehouse, block_id, variant_id,
                )

            # Update EMAs *after* the gradient step.
            balancer.observe(task_loss.item(), cons_loss.item())

            if step % log_every == 0:
                print(
                    f"epoch {epoch + 1}/{epochs}  step {step:>6d}  "
                    f"loss={loss.item():.4f}  task={task_loss.item():.4f}  "
                    f"cons={cons_loss.item():.4f}  λ={weight:.4f}"
                )
            step += 1

            del model_a, model_b, optim
            if device.type == "cuda":
                torch.cuda.empty_cache()

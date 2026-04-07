"""Private helpers shared by the adapter modules.

Nothing in this module is part of the public adapter API.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

REQUIRED_USER_CALLABLES: tuple[str, ...] = (
    "model_factory",
    "load_base_checkpoint",
    "get_dataloaders",
    "infer_fn",
    "loss_fn",
)


def to_device(batch: Any, device: str | torch.device) -> Any:
    """Move a (possibly nested) batch onto ``device``.

    Handles tensors, tuples, lists, and dicts. Anything else is left
    untouched, on the assumption that the user's ``infer_fn`` knows
    what to do with it.
    """
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, (list, tuple)):
        moved = [to_device(b, device) for b in batch]
        return type(batch)(moved)
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    return batch


def load_user_module(path: str | Path) -> ModuleType:
    """Import a user-supplied Python file as a module and validate its API.

    The user file must define five top-level callables:
    ``model_factory``, ``load_base_checkpoint``, ``get_dataloaders``,
    ``infer_fn``, and ``loss_fn``. ``consistency_fn`` is optional.
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"User module not found: {path}")

    module_name = f"_saswise_user_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    missing = [name for name in REQUIRED_USER_CALLABLES if not hasattr(module, name)]
    if missing:
        raise AttributeError(
            f"User module {path} is missing required callables: {missing}. "
            f"Required: {list(REQUIRED_USER_CALLABLES)}"
        )
    return module

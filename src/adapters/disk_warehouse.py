"""Disk-resident warehouse — one ``.pt`` file per ``(block, variant)``.

The warehouse layout under ``root/`` is::

    root/
    ├── warehouse_metadata.json     # block specs + num_variants
    ├── block_00/
    │   ├── variant_00.pt           # state dict slice for block 0, variant 0
    │   ├── variant_01.pt
    │   └── …
    ├── block_01/
    │   └── …
    └── …

Each variant ``.pt`` file holds a ``dict[str, torch.Tensor]`` containing
exactly the keys belonging to its block. Loading a variant therefore
costs only the size of one block, never the full model.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import torch

from .decompose import BlockSpec

METADATA_FILENAME = "warehouse_metadata.json"


class DiskWarehouse:
    """Persistent on-disk store of variant slices.

    The warehouse never holds more than a single variant tensor in
    memory at a time; everything else lives on disk under ``root``.
    """

    def __init__(
        self,
        root: str | Path,
        block_specs: List[BlockSpec],
        num_variants: int,
    ) -> None:
        if num_variants < 1:
            raise ValueError(f"num_variants must be >= 1, got {num_variants}")
        if not block_specs:
            raise ValueError("block_specs must be non-empty")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.block_specs = list(block_specs)
        self.num_variants = num_variants
        self._save_metadata()

    # ------------------------------------------------------------------ I/O

    def variant_path(self, block_id: int, variant_id: int) -> Path:
        """Filesystem path for a single ``(block, variant)`` slice."""
        return self.root / f"block_{block_id:02d}" / f"variant_{variant_id:02d}.pt"

    def save_variant(
        self,
        block_id: int,
        variant_id: int,
        state_dict: Dict[str, torch.Tensor],
    ) -> None:
        """Write a variant slice to disk."""
        path = self.variant_path(block_id, variant_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Always store on CPU so reloading is device-independent.
        cpu_state = {k: v.detach().cpu() for k, v in state_dict.items()}
        torch.save(cpu_state, path)

    def load_variant(
        self,
        block_id: int,
        variant_id: int,
        map_location: str | torch.device = "cpu",
    ) -> Dict[str, torch.Tensor]:
        """Read a variant slice from disk."""
        path = self.variant_path(block_id, variant_id)
        if not path.exists():
            raise FileNotFoundError(f"Variant not found on disk: {path}")
        return torch.load(path, map_location=map_location)

    # ------------------------------------------------------------------ metadata

    def _save_metadata(self) -> None:
        meta = {
            "num_variants": self.num_variants,
            "block_specs": [
                {
                    "id": s.id,
                    "name": s.name,
                    "state_dict_keys": s.state_dict_keys,
                    "num_params": s.num_params,
                }
                for s in self.block_specs
            ],
        }
        (self.root / METADATA_FILENAME).write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, root: str | Path) -> "DiskWarehouse":
        """Reconstruct a warehouse handle from a directory on disk."""
        root = Path(root)
        meta_path = root / METADATA_FILENAME
        if not meta_path.exists():
            raise FileNotFoundError(
                f"No {METADATA_FILENAME} found under {root}; not a warehouse directory."
            )
        meta = json.loads(meta_path.read_text())
        specs = [
            BlockSpec(
                id=s["id"],
                name=s["name"],
                state_dict_keys=list(s["state_dict_keys"]),
                num_params=int(s["num_params"]),
            )
            for s in meta["block_specs"]
        ]
        wh = cls.__new__(cls)
        wh.root = root
        wh.block_specs = specs
        wh.num_variants = int(meta["num_variants"])
        return wh

    # ------------------------------------------------------------------ accessors

    @property
    def num_blocks(self) -> int:
        return len(self.block_specs)

    def num_configurations(self) -> int:
        """Total number of distinct sub-models the warehouse can produce."""
        return self.num_variants ** self.num_blocks

    def block_by_id(self, block_id: int) -> BlockSpec:
        for spec in self.block_specs:
            if spec.id == block_id:
                return spec
        raise KeyError(f"No block with id={block_id}")

    def __repr__(self) -> str:
        return (
            f"DiskWarehouse(root={self.root!r}, "
            f"blocks={self.num_blocks}, variants={self.num_variants}, "
            f"configurations={self.num_configurations()})"
        )

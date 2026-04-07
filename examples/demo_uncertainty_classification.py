"""SASWISE-UE end-to-end uncertainty demo on a small classification task.

This script runs the full adopter pipeline using the Python API
(no CLI subprocesses) and reports the calibration story you actually
care about: how well does the ensemble's predictive uncertainty
correlate with its prediction errors?

Pipeline
--------

1. Train a small MLP base classifier for a few epochs.
2. Decompose the trained model into 4 balanced blocks via
   :func:`adapters.decompose_balanced`.
3. Materialise 3 perturbed variants per block on disk
   (3⁴ = 81 sub-models in the warehouse).
4. Fine-tune the warehouse with adaptive consistency balancing
   (``target_ratio=0.1``, 100-step warm-up).
5. Evaluate 32 randomly sampled sub-models on the validation set,
   streaming the per-sample mean, standard deviation, and predictive
   entropy via Welford.
6. Compute calibration metrics:
   - ensemble accuracy
   - mean predictive entropy split by correct vs incorrect
   - Pearson correlation between predictive entropy and 0/1 error
   - AUROC of "is this prediction wrong" given predictive entropy
   - a risk–coverage table at five coverage levels

Data
----

By default the script tries to download MNIST via torchvision. If
that fails (no network), it transparently falls back to a synthetic
two-Gaussian classification dataset that exercises the same pipeline
in ~30 seconds.

Run it
------

.. code-block:: bash

    # Default: MNIST if available, synthetic otherwise.
    python examples/demo_uncertainty_classification.py

    # Force the synthetic dataset (offline / fast).
    python examples/demo_uncertainty_classification.py --dataset synthetic

    # Force MNIST (will fail loudly if no network).
    python examples/demo_uncertainty_classification.py --dataset mnist
"""

from __future__ import annotations

import argparse
import math
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset

# Make the repo root importable when running from anywhere.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.adapters import (  # noqa: E402
    ConsistencyBalance,
    DiskWarehouse,
    decompose_balanced,
    finetune_warehouse,
    infer_with_uncertainty,
    initialize_variants,
)


# =============================================================================
# Models
# =============================================================================


class MLP(nn.Module):
    """A small four-layer MLP — large enough to split into 4 blocks."""

    def __init__(self, in_features: int, num_classes: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


# =============================================================================
# Datasets
# =============================================================================


def _try_load_mnist(data_root: Path) -> Tuple[DataLoader, DataLoader, int, int]:
    """Try to download MNIST. Returns (train, val, in_features, num_classes)."""
    from torchvision import datasets, transforms

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_ds = datasets.MNIST(str(data_root), train=True, download=True, transform=transform)
    val_ds = datasets.MNIST(str(data_root), train=False, download=True, transform=transform)
    return (
        DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0),
        DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=0),
        28 * 28,
        10,
    )


def _make_synthetic(seed: int = 0) -> Tuple[DataLoader, DataLoader, int, int]:
    """A 3-class Gaussian mixture in 20 dimensions, with intentional overlap.

    Class clusters are placed close enough that some validation points
    are intrinsically ambiguous — exactly the situation where we want
    high predictive uncertainty.
    """
    g = torch.Generator().manual_seed(seed)
    in_dim = 20
    num_classes = 3
    n_train_per_class = 800
    n_val_per_class = 200

    # Class centres on a small simplex so the classes overlap.
    centres = torch.tensor(
        [
            [+1.5, +1.5] + [0.0] * (in_dim - 2),
            [-1.5, +1.5] + [0.0] * (in_dim - 2),
            [+0.0, -1.5] + [0.0] * (in_dim - 2),
        ]
    )

    def _sample(n_per_class: int):
        xs = []
        ys = []
        for c, centre in enumerate(centres):
            x = centre + torch.randn(n_per_class, in_dim, generator=g) * 1.2
            y = torch.full((n_per_class,), c, dtype=torch.long)
            xs.append(x)
            ys.append(y)
        x = torch.cat(xs, dim=0)
        y = torch.cat(ys, dim=0)
        perm = torch.randperm(x.size(0), generator=g)
        return x[perm], y[perm]

    x_train, y_train = _sample(n_train_per_class)
    x_val, y_val = _sample(n_val_per_class)
    return (
        DataLoader(TensorDataset(x_train, y_train), batch_size=128, shuffle=True),
        DataLoader(TensorDataset(x_val, y_val), batch_size=256, shuffle=False),
        in_dim,
        num_classes,
    )


def get_dataset(name: str, data_root: Path) -> Tuple[DataLoader, DataLoader, int, int, str]:
    """Pick a dataset. Returns (train, val, in_features, num_classes, label)."""
    if name == "synthetic":
        train, val, in_dim, n_cls = _make_synthetic()
        return train, val, in_dim, n_cls, "synthetic Gaussian mixture (3 classes, 20 features)"
    if name == "mnist":
        train, val, in_dim, n_cls = _try_load_mnist(data_root)
        return train, val, in_dim, n_cls, f"MNIST (10 classes, 28×28 → {in_dim} features)"
    if name == "auto":
        try:
            train, val, in_dim, n_cls = _try_load_mnist(data_root)
            return train, val, in_dim, n_cls, f"MNIST (10 classes, 28×28 → {in_dim} features)"
        except Exception as e:  # noqa: BLE001 — any failure → fall back
            print(f"[demo] MNIST unavailable ({type(e).__name__}: {e}); falling back to synthetic.")
            train, val, in_dim, n_cls = _make_synthetic()
            return train, val, in_dim, n_cls, "synthetic Gaussian mixture (3 classes, 20 features)"
    raise ValueError(f"Unknown dataset {name!r}")


# =============================================================================
# Phase 1 — train the base checkpoint
# =============================================================================


def train_base_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    device: torch.device,
) -> float:
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.to(device)
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y)
            optim.zero_grad()
            loss.backward()
            optim.step()
        acc = _val_accuracy(model, val_loader, device)
        print(f"  base epoch {epoch + 1}/{epochs}  val_acc={acc:.4f}")
    return _val_accuracy(model, val_loader, device)


def _val_accuracy(model: nn.Module, val_loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.numel()
    return correct / total


# =============================================================================
# Phase 2 — adopter pipeline (decompose → init → finetune → infer)
# =============================================================================


def make_user_module(
    base_model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    in_features: int,
    num_classes: int,
):
    """Wrap an in-memory model + loaders into the 5-callable user contract."""

    def model_factory():
        return MLP(in_features=in_features, num_classes=num_classes)

    def load_base_checkpoint(model, path):
        model.load_state_dict(torch.load(path, map_location="cpu"))

    def get_dataloaders():
        return train_loader, val_loader

    def infer_fn(model, batch):
        x, _ = batch
        return model(x)

    def loss_fn(out, batch):
        _, y = batch
        return F.cross_entropy(out, y)

    def consistency_fn(out_a, out_b):
        return F.kl_div(
            F.log_softmax(out_a, dim=-1),
            F.softmax(out_b, dim=-1),
            reduction="batchmean",
        )

    return SimpleNamespace(
        model_factory=model_factory,
        load_base_checkpoint=load_base_checkpoint,
        get_dataloaders=get_dataloaders,
        infer_fn=infer_fn,
        loss_fn=loss_fn,
        consistency_fn=consistency_fn,
    )


# =============================================================================
# Phase 3 — calibration metrics
# =============================================================================


def _pearson_r(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().double()
    b = b.flatten().double()
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    num = (a_centered * b_centered).sum()
    den = a_centered.pow(2).sum().sqrt() * b_centered.pow(2).sum().sqrt()
    if den.item() == 0.0:
        return float("nan")
    return float(num / den)


def _auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """AUROC computed via the rank-sum identity. ``labels`` is 0/1."""
    scores = scores.flatten().double()
    labels = labels.flatten().long()
    n_pos = int(labels.sum().item())
    n_neg = int(labels.numel() - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = torch.argsort(scores)
    ranks = torch.empty_like(scores)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=torch.float64)
    pos_rank_sum = ranks[labels == 1].sum().item()
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def calibration_report(
    pred_mean_logits: torch.Tensor,
    pred_entropy: torch.Tensor,
    pred_std: torch.Tensor,
    val_labels: torch.Tensor,
) -> None:
    pred_classes = pred_mean_logits.argmax(dim=-1)
    correct = (pred_classes == val_labels).float()
    error = 1.0 - correct
    accuracy = correct.mean().item()

    print(f"\n  ensemble accuracy           : {accuracy:.4f}")
    print(f"  mean predictive entropy     : {pred_entropy.mean().item():.4f}")
    print(f"    on correct samples        : {pred_entropy[correct == 1].mean().item():.4f}")
    print(f"    on incorrect samples      : {pred_entropy[correct == 0].mean().item():.4f}")

    # Per-sample std summed over class dim — a single uncertainty scalar.
    std_per_sample = pred_std.norm(p=2, dim=-1)
    print(f"  mean per-sample std (||·||₂): {std_per_sample.mean().item():.4f}")

    # ----- correlations -----
    pearson_ent_err = _pearson_r(pred_entropy, error)
    pearson_std_err = _pearson_r(std_per_sample, error)
    print(f"\n  Pearson r(entropy, error)   : {pearson_ent_err:+.4f}   "
          f"(positive = high entropy ↔ wrong)")
    print(f"  Pearson r(std,     error)   : {pearson_std_err:+.4f}")

    # AUROC: can predictive entropy detect misclassifications?
    auroc_ent = _auroc(pred_entropy, error.long())
    auroc_std = _auroc(std_per_sample, error.long())
    print(f"\n  AUROC of misclassification (using entropy as score): {auroc_ent:.4f}")
    print(f"  AUROC of misclassification (using std    as score) : {auroc_std:.4f}")
    print(f"    (0.5 = uncertainty is uninformative; 1.0 = perfectly identifies all errors)")

    # ----- risk-coverage curve -----
    # Sort samples by ascending entropy, then look at error rate within
    # the most-confident X% of predictions.
    print("\n  Risk-coverage curve (selective prediction):")
    print("    coverage   error rate   #samples kept")
    order = torch.argsort(pred_entropy)
    for cov in [1.00, 0.80, 0.60, 0.40, 0.20]:
        k = max(1, int(round(cov * pred_entropy.numel())))
        keep = order[:k]
        cov_err = error[keep].mean().item()
        print(f"    {cov:5.0%}      {cov_err:6.4f}       {k:>6d}")


def collect_val_labels(val_loader: DataLoader) -> torch.Tensor:
    parts = []
    for _, y in val_loader:
        parts.append(y)
    return torch.cat(parts, dim=0)


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("Pipeline")[0].strip())
    parser.add_argument("--dataset", choices=["auto", "mnist", "synthetic"], default="auto")
    parser.add_argument("--data-root", type=Path, default=Path("~/data").expanduser())
    parser.add_argument("--base-epochs", type=int, default=2,
                        help="Epochs for the base classifier (default: 2)")
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--num-variants", type=int, default=3)
    parser.add_argument("--noise-std", type=float, default=5e-2,
                        help="Initial perturbation scale (default: 5e-2; "
                             "larger gives more diverse variants).")
    parser.add_argument("--finetune-epochs", type=int, default=2)
    parser.add_argument("--target-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--num-configurations", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    print("=" * 64)
    print("SASWISE-UE — uncertainty calibration demo")
    print("=" * 64)
    print(f"device         : {device}")
    print(f"seed           : {args.seed}")

    # ---------- data ----------
    train_loader, val_loader, in_features, num_classes, label = get_dataset(
        args.dataset, args.data_root,
    )
    print(f"dataset        : {label}")
    print(f"  train samples: {len(train_loader.dataset)}")
    print(f"  val   samples: {len(val_loader.dataset)}")
    print(f"  features     : {in_features}")
    print(f"  classes      : {num_classes}")

    # ---------- 1. base ----------
    print("\n[1/4] Training base classifier")
    base_model = MLP(in_features=in_features, num_classes=num_classes)
    base_acc = train_base_model(base_model, train_loader, val_loader, args.base_epochs, device)
    print(f"  base val_acc : {base_acc:.4f}")

    with tempfile.TemporaryDirectory() as tmp:
        base_path = Path(tmp) / "base.pt"
        torch.save(base_model.state_dict(), base_path)

        # ---------- 2. adopter init ----------
        print("\n[2/4] Decomposing the base model and materialising variants")
        specs = decompose_balanced(base_model.cpu(), num_blocks=args.num_blocks)
        print(f"  decomposed into {len(specs)} blocks "
              f"(requested {args.num_blocks}; mismatch happens on tiny models):")
        for s in specs:
            print(f"    block {s.id}  {s.num_params:>7d} params  "
                  f"({len(s.state_dict_keys)} tensors)")

        warehouse = DiskWarehouse(
            root=Path(tmp) / "warehouse",
            block_specs=specs,
            num_variants=args.num_variants,
        )
        initialize_variants(
            base_model.cpu(), specs, args.num_variants, warehouse,
            noise_std=args.noise_std, seed=args.seed,
        )
        print(f"  warehouse: {len(specs)} blocks × {args.num_variants} variants "
              f"= {warehouse.num_configurations()} possible sub-models")

        # ---------- 3. finetune ----------
        print("\n[3/4] Fine-tuning the warehouse with adaptive consistency weighting")
        user_module = make_user_module(
            base_model, train_loader, val_loader, in_features, num_classes,
        )
        finetune_warehouse(
            warehouse=warehouse,
            user_module=user_module,
            epochs=args.finetune_epochs,
            lr=1e-4,
            consistency=ConsistencyBalance(
                target_ratio=args.target_ratio,
                warmup_steps=args.warmup_steps,
                ema_decay=0.9,
            ),
            device=str(device),
            log_every=max(1, len(train_loader) // 4),
        )

        # ---------- 4. ensembled inference + uncertainty ----------
        print("\n[4/4] Ensembled inference (streaming Welford)")
        pred = infer_with_uncertainty(
            warehouse=warehouse,
            user_module=user_module,
            num_configurations=args.num_configurations,
            task="classification",
            device=str(device),
            seed=args.seed,
            show_progress=False,
        )
        print(f"  configurations sampled: {pred.num_configurations}")
        print(f"  mean shape            : {tuple(pred.mean.shape)}")

        # ---------- 5. calibration metrics ----------
        print("\n----- Calibration metrics -----")
        val_labels = collect_val_labels(val_loader)
        calibration_report(
            pred_mean_logits=pred.mean,
            pred_entropy=pred.predictive_entropy,
            pred_std=pred.std,
            val_labels=val_labels,
        )

    print("\n" + "=" * 64)
    print("Done.")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Minimal SASWISE adoption example: a 4-layer MLP on MNIST.

This is a *user module* — the kind of file you point at with
``scripts/adopt_checkpoint.py --user-module``. It demonstrates the
five-callable contract that the adapter expects.

To try the full pipeline end-to-end::

    # 1. Bootstrap a base checkpoint by training a few epochs.
    python examples/adopt_mnist_mlp.py train --epochs 2 --output base_mlp.pt

    # 2. Decompose into 4 balanced blocks with 3 variants each.
    python scripts/adopt_checkpoint.py init \\
        --user-module   examples/adopt_mnist_mlp.py \\
        --checkpoint    base_mlp.pt \\
        --output-dir    experiment/mnist_mlp \\
        --decomposition auto-balanced:4 \\
        --variants      3

    # 3. Diversify with the consistency loss.
    python scripts/adopt_checkpoint.py finetune \\
        --user-module   examples/adopt_mnist_mlp.py \\
        --warehouse     experiment/mnist_mlp/warehouse \\
        --epochs        1

    # 4. Ensembled inference + uncertainty (4^3 = 64 sub-models).
    python scripts/adopt_checkpoint.py infer \\
        --user-module       examples/adopt_mnist_mlp.py \\
        --warehouse         experiment/mnist_mlp/warehouse \\
        --num-configurations 64 \\
        --task              classification \\
        --output            mnist_mlp_predictions.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATA_ROOT = Path("~/data").expanduser()


# ---------------------------------------------------------------------- model


class MLP(nn.Module):
    """A small four-layer MLP — large enough to split into multiple blocks."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


# --------------------------------------------------------- the 5-callable API


def model_factory() -> nn.Module:
    """Return a fresh, randomly-initialised model."""
    return MLP()


def load_base_checkpoint(model: nn.Module, path: str) -> None:
    """Load the user's pretrained weights into a freshly-built model."""
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)


def get_dataloaders() -> Tuple[DataLoader, DataLoader]:
    """Return ``(train_loader, val_loader)``."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_ds = datasets.MNIST(DATA_ROOT, train=True, download=True, transform=transform)
    val_ds = datasets.MNIST(DATA_ROOT, train=False, download=True, transform=transform)
    return (
        DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2),
        DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=2),
    )


def infer_fn(model: nn.Module, batch) -> torch.Tensor:
    """One forward pass. ``batch`` is whatever the dataloader yields."""
    x, _ = batch
    return model(x)


def loss_fn(out: torch.Tensor, batch) -> torch.Tensor:
    """Per-batch task loss for fine-tuning."""
    _, y = batch
    return F.cross_entropy(out, y)


# Optional: override the default MSE consistency with a sharper KL term.
def consistency_fn(out_a: torch.Tensor, out_b: torch.Tensor) -> torch.Tensor:
    """KL(softmax(a) || softmax(b)) — sharper than MSE for classification."""
    return F.kl_div(
        F.log_softmax(out_a, dim=-1),
        F.softmax(out_b, dim=-1),
        reduction="batchmean",
    )


# ------------------------------------- helper: train a base checkpoint to use


def _train_base_checkpoint(epochs: int, output: str) -> None:
    """Train a fresh MLP on MNIST and save its state dict.

    Run with ``python examples/adopt_mnist_mlp.py train --epochs 2 ...``
    to bootstrap a checkpoint that the adopter can then ingest.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_factory().to(device)
    train_loader, val_loader = get_dataloaders()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = F.cross_entropy(out, y)
            optim.zero_grad()
            loss.backward()
            optim.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(dim=-1)
                correct += (preds == y).sum().item()
                total += y.numel()
        print(f"epoch {epoch + 1}/{epochs}  val_acc={correct / total:.4f}")

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output)
    print(f"Saved base checkpoint to {output}")


def _main() -> None:
    parser = argparse.ArgumentParser(description="MNIST MLP — utility entry point.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_train = sub.add_parser("train", help="Train a base checkpoint.")
    p_train.add_argument("--epochs", type=int, default=2)
    p_train.add_argument("--output", default="base_mlp.pt")
    args = parser.parse_args()
    if args.cmd == "train":
        _train_base_checkpoint(args.epochs, args.output)


if __name__ == "__main__":
    _main()

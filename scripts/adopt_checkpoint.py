"""SASWISE adopter — three-subcommand CLI for adopting any pretrained model.

The script reads a pre-trained checkpoint, decomposes the model into
``B`` blocks, materialises ``V`` initial variants per block on disk,
fine-tunes the warehouse with the diversification consistency loss,
and finally produces an ensembled prediction with calibrated
uncertainty — all without ever holding more than one sub-model in GPU
memory at a time.

Usage
-----

.. code-block:: bash

    # 1. Decompose checkpoint and materialise initial variants.
    python scripts/adopt_checkpoint.py init \\
        --user-module   examples/adopt_mnist_mlp.py \\
        --checkpoint    pretrained.pt \\
        --output-dir    experiment/my_run \\
        --decomposition auto-balanced:6 \\
        --variants      4

    # 2. Fine-tune the warehouse.
    python scripts/adopt_checkpoint.py finetune \\
        --user-module   examples/adopt_mnist_mlp.py \\
        --warehouse     experiment/my_run/warehouse \\
        --epochs        5 \\
        --consistency-weight 0.1

    # 3. Run ensembled inference + uncertainty.
    python scripts/adopt_checkpoint.py infer \\
        --user-module       examples/adopt_mnist_mlp.py \\
        --warehouse         experiment/my_run/warehouse \\
        --num-configurations 64 \\
        --task              classification \\
        --output            predictions.pt

The user module is a single Python file that defines five callables:
``model_factory``, ``load_base_checkpoint``, ``get_dataloaders``,
``infer_fn``, and ``loss_fn`` — see ``examples/adopt_mnist_mlp.py`` for
a minimal example.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Make the repository root importable when this script is run directly.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.adapters import (  # noqa: E402  (sys.path tweak above)
    ConsistencyBalance,
    DiskWarehouse,
    decompose_balanced,
    decompose_top_level,
    finetune_warehouse,
    infer_with_uncertainty,
    initialize_variants,
)
from src.adapters._internal import load_user_module  # noqa: E402


# ---------------------------------------------------------------------- helpers


def _parse_decomposition(spec: str):
    """Translate a ``--decomposition`` string into a callable.

    Supported forms:

    * ``auto-balanced:K`` — partition into K parameter-balanced blocks.
    * ``auto-top-level``  — every direct child of the model is a block.
    """
    if spec.startswith("auto-balanced:"):
        try:
            k = int(spec.split(":", 1)[1])
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid auto-balanced spec {spec!r}; expected 'auto-balanced:<int>'"
            ) from exc
        return ("balanced", k)
    if spec == "auto-top-level":
        return ("top-level", None)
    raise argparse.ArgumentTypeError(
        f"Unsupported decomposition {spec!r}. "
        "Use 'auto-balanced:K' or 'auto-top-level'."
    )


# ----------------------------------------------------------------- subcommands


def cmd_init(args: argparse.Namespace) -> int:
    """Initialise a warehouse from a pretrained checkpoint."""
    user_module = load_user_module(args.user_module)
    print(f"Loaded user module: {Path(args.user_module).resolve()}")

    model = user_module.model_factory()
    print(f"Built fresh model: {type(model).__name__}")
    user_module.load_base_checkpoint(model, args.checkpoint)
    print(f"Loaded base checkpoint: {args.checkpoint}")

    decomp_kind, decomp_arg = args.decomposition
    if decomp_kind == "balanced":
        block_specs = decompose_balanced(model, decomp_arg)
    else:
        block_specs = decompose_top_level(model)
    print(f"Decomposed model into {len(block_specs)} blocks ({decomp_kind})")
    for spec in block_specs:
        print(
            f"  block {spec.id:>2d}  '{spec.name}'  "
            f"{spec.num_params:>10d} elements over {len(spec.state_dict_keys)} tensors"
        )

    warehouse_root = Path(args.output_dir) / "warehouse"
    warehouse = DiskWarehouse(
        root=warehouse_root,
        block_specs=block_specs,
        num_variants=args.variants,
    )
    print(f"\nWriting {args.variants} variants per block to {warehouse_root} ...")
    initialize_variants(
        model=model,
        block_specs=block_specs,
        num_variants=args.variants,
        warehouse=warehouse,
        noise_std=args.noise_std,
        seed=args.seed,
    )
    print(
        f"Done. Warehouse contains {warehouse.num_blocks} blocks "
        f"× {warehouse.num_variants} variants = "
        f"{warehouse.num_configurations()} possible sub-models."
    )
    return 0


def cmd_finetune(args: argparse.Namespace) -> int:
    """Fine-tune every variant in the warehouse with the consistency loss."""
    user_module = load_user_module(args.user_module)
    warehouse = DiskWarehouse.load(args.warehouse)
    print(f"Loaded warehouse: {warehouse}")

    consistency = ConsistencyBalance(
        adaptive=(args.consistency_mode == "adaptive"),
        target_ratio=args.target_ratio,
        fixed_weight=args.fixed_weight,
        warmup_steps=args.warmup_steps,
        ema_decay=args.ema_decay,
        min_weight=args.min_weight,
        max_weight=args.max_weight,
    )
    mode_label = (
        f"adaptive (target_ratio={consistency.target_ratio}, "
        f"warmup={consistency.warmup_steps}, ema_decay={consistency.ema_decay})"
        if consistency.adaptive
        else f"fixed (λ={consistency.fixed_weight}, warmup={consistency.warmup_steps})"
    )
    print(f"Consistency balancing: {mode_label}")

    finetune_warehouse(
        warehouse=warehouse,
        user_module=user_module,
        epochs=args.epochs,
        lr=args.lr,
        consistency=consistency,
        device=args.device,
        log_every=args.log_every,
    )
    print("Fine-tuning complete.")
    return 0


def cmd_infer(args: argparse.Namespace) -> int:
    """Run ensembled inference and save predictions + uncertainty."""
    user_module = load_user_module(args.user_module)
    warehouse = DiskWarehouse.load(args.warehouse)
    print(f"Loaded warehouse: {warehouse}")

    pred = infer_with_uncertainty(
        warehouse=warehouse,
        user_module=user_module,
        num_configurations=args.num_configurations,
        task=args.task,
        device=args.device,
        seed=args.seed,
        show_progress=not args.quiet,
    )

    payload = {
        "mean": pred.mean,
        "std": pred.std,
        "num_configurations": pred.num_configurations,
        "block_configurations": pred.block_configurations,
    }
    if pred.predictive_entropy is not None:
        payload["predictive_entropy"] = pred.predictive_entropy

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)

    print(f"\nSaved ensemble predictions to {output_path}")
    print(f"  mean shape : {tuple(pred.mean.shape)}")
    print(f"  std  shape : {tuple(pred.std.shape)}")
    print(f"  configs    : {pred.num_configurations}")
    if pred.predictive_entropy is not None:
        print(f"  entropy    : shape {tuple(pred.predictive_entropy.shape)}")
    return 0


# --------------------------------------------------------------------- argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="adopt_checkpoint",
        description=__doc__.split("Usage")[0].strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---------------- init ----------------
    p_init = sub.add_parser(
        "init",
        help="Decompose a checkpoint and materialise initial variants.",
    )
    p_init.add_argument("--user-module", required=True,
                        help="Path to the user .py file with the 5 required callables.")
    p_init.add_argument("--checkpoint", required=True,
                        help="Path to the pretrained state-dict file.")
    p_init.add_argument("--output-dir", required=True,
                        help="Directory in which to create the warehouse.")
    p_init.add_argument("--decomposition", type=_parse_decomposition,
                        default=("balanced", 6),
                        help="auto-balanced:K (default) or auto-top-level.")
    p_init.add_argument("--variants", type=int, default=4,
                        help="Variants per block (default: 4).")
    p_init.add_argument("--noise-std", type=float, default=1e-3,
                        help="Std of the multiplicative Gaussian perturbation (default: 1e-3).")
    p_init.add_argument("--seed", type=int, default=0)
    p_init.set_defaults(func=cmd_init)

    # ---------------- finetune ----------------
    p_ft = sub.add_parser(
        "finetune",
        help="Fine-tune the warehouse with the diversification consistency loss.",
    )
    p_ft.add_argument("--user-module", required=True)
    p_ft.add_argument("--warehouse", required=True,
                      help="Path to the warehouse directory created by 'init'.")
    p_ft.add_argument("--epochs", type=int, default=5)
    p_ft.add_argument("--lr", type=float, default=1e-4)
    p_ft.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p_ft.add_argument("--log-every", type=int, default=50)

    # Consistency-loss balancing.
    p_ft.add_argument(
        "--consistency-mode",
        choices=["adaptive", "fixed"],
        default="adaptive",
        help="adaptive: λ is auto-balanced from EMA loss magnitudes (default). "
             "fixed: λ stays at --fixed-weight throughout training.",
    )
    p_ft.add_argument(
        "--target-ratio",
        type=float,
        default=0.1,
        help="Adaptive mode only: target fraction of total loss contributed "
             "by the consistency term. Default: 0.1 (consistency ≈ 10%% of total).",
    )
    p_ft.add_argument(
        "--fixed-weight",
        type=float,
        default=0.1,
        help="Fixed mode only: constant value of λ. Default: 0.1.",
    )
    p_ft.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
        help="Number of steps over which λ ramps linearly from 0 to its target. "
             "Set to 0 to disable. Default: 100.",
    )
    p_ft.add_argument(
        "--ema-decay",
        type=float,
        default=0.9,
        help="Adaptive mode only: smoothing factor for the loss-magnitude EMAs. "
             "Higher = smoother but slower-reacting. Default: 0.9.",
    )
    p_ft.add_argument(
        "--min-weight",
        type=float,
        default=0.0,
        help="Adaptive mode only: lower bound on λ. Default: 0.",
    )
    p_ft.add_argument(
        "--max-weight",
        type=float,
        default=100.0,
        help="Adaptive mode only: upper bound on λ. Default: 100.",
    )
    p_ft.set_defaults(func=cmd_finetune)

    # ---------------- infer ----------------
    p_in = sub.add_parser(
        "infer",
        help="Run ensembled inference and save predictions + uncertainty.",
    )
    p_in.add_argument("--user-module", required=True)
    p_in.add_argument("--warehouse", required=True)
    p_in.add_argument("--num-configurations", type=int, default=64)
    p_in.add_argument("--task", choices=["classification", "regression", "segmentation", "auto"],
                      default="auto")
    p_in.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p_in.add_argument("--seed", type=int, default=0)
    p_in.add_argument("--output", default="predictions.pt")
    p_in.add_argument("--quiet", action="store_true")
    p_in.set_defaults(func=cmd_infer)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

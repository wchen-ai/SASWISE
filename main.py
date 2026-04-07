"""SASWISE fine-tuning entry point.

Loads a YAML config, builds a warehouse from a pretrained checkpoint,
assembles a model from an optional block-configuration specification,
and hands the result to :class:`src.training.fine_tuner.FineTuner`.

Example
-------
.. code-block:: bash

    python main.py --config config.yaml --block-config configs/my_blocks.yaml
"""

import argparse
from pathlib import Path

from src.models.warehouse_setup import build_warehouse
from src.models.model_assembler import model_assembler
from src.training.fine_tuner import FineTuner
from src.utils.helpers import load_config
from src.utils.logger import setup_logging


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the fine-tuning workflow."""
    parser = argparse.ArgumentParser(description="Fine-tuning workflow")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--block_config",
        type=str,
        default=None,
        help="Path to block_config file for model assembling",
    )
    return parser.parse_args()


def main():
    # Parse arguments and load configuration
    args = parse_args()
    config = load_config(args.config)
    
    # Setup logging
    logger = setup_logging(config['logging'])
    logger.info("Starting fine-tuning workflow")
    
    # Step 1: Warehouse Setup
    logger.info("Building warehouse (loading and analysing model)")
    warehouse = build_warehouse(
        model_path=config['model']['pretrained_path'],
        model_type=config['model']['model_type']
    )

    # Step 2: Model assembly
    logger.info("Assembling model according to block configuration")
    block_config = load_config(args.block_config) if args.block_config else {}
    assembled_model = model_assembler(block_config=block_config, warehouse=warehouse)

    # Step 3: Fine-tuning
    logger.info("Starting fine-tuning process")
    fine_tuner = FineTuner(
        model=assembled_model,
        config=config['training'],
        warehouse=warehouse
    )
    fine_tuner.train()
    
    logger.info("Fine-tuning workflow completed")


if __name__ == "__main__":
    main() 
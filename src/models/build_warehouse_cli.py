import os
import torch
import json
from pathlib import Path
from monai.networks.nets import ViT
from typing import Dict, List, Any
import re


def create_warehouse_structure(block_analysis_file: str, base_dir: str = "project"):
    """Create warehouse folder structure and save block variants."""
    # Create base directory structure
    project_dir = Path(base_dir)
    warehouse_dir = project_dir / "warehouse"
    warehouse_dir.mkdir(parents=True, exist_ok=True)

    # Create ViT model
    model = ViT(
        in_channels=1,
        img_size=(96, 96, 96),
        patch_size=(16, 16, 16),
        hidden_size=768,
        mlp_dim=3072,
        num_layers=12,
        num_heads=12,
        num_classes=2
    )

    # Parse block analysis file
    blocks = {}
    current_block = None
    current_variants = 0

    with open(block_analysis_file, 'r') as f:
        for line in f:
            # Match block header with variant count
            block_match = re.match(r"Block (\d+)<(\d+)>:", line)
            if block_match:
                current_block = int(block_match.group(1))
                current_variants = int(block_match.group(2))
                blocks[current_block] = {
                    'variants': current_variants,
                    'nodes': [],
                }
                continue

            # Match nodes
            if current_block and line.strip().startswith("- "):
                node = line.strip()[2:].strip()
                blocks[current_block]['nodes'].append(node)

    # Create block directories and save variants
    state_dict = model.state_dict()

    for block_id, block_info in blocks.items():
        # Create block directory
        block_dir = warehouse_dir / f"block_{block_id}"
        block_dir.mkdir(exist_ok=True)

        # Extract parameters for this block
        block_state = {}
        total_params = 0
        for node in block_info['nodes']:
            # Find all parameters belonging to this node
            node_params = {}
            for param_name, param in state_dict.items():
                if param_name.startswith(node):
                    param_shape = list(param.shape)
                    param_size = param.numel()
                    node_params[param_name] = {
                        'shape': param_shape,
                        'size': param_size
                    }
                    total_params += param_size

        # Save block info with parameter shapes and sizes
        info = {
            'num_variants': block_info['variants'],
            'nodes': block_info['nodes'],
            'parameters': total_params,
            'parameter_info': node_params
        }
        with open(block_dir / "info.json", 'w') as f:
            json.dump(info, f, indent=2)

        # Save variant metadata
        for variant in range(1, block_info['variants'] + 1):
            variant_info = {
                'block': block_id,
                'variant_id': variant,
                'nodes': block_info['nodes'],
                'total_parameters': total_params
            }
            with open(block_dir / f"variant_{variant}.json", 'w') as f:
                json.dump(variant_info, f, indent=2)

    # Save warehouse info
    warehouse_info = {
        'total_blocks': len(blocks),
        'blocks': blocks,
        'total_parameters': sum(p.numel() for p in state_dict.values()),
        'model_config': {
            'in_channels': 1,
            'img_size': [96, 96, 96],
            'patch_size': [16, 16, 16],
            'hidden_size': 768,
            'mlp_dim': 3072,
            'num_layers': 12,
            'num_heads': 12,
            'num_classes': 2
        }
    }
    with open(warehouse_dir / "warehouse_info.json", 'w') as f:
        json.dump(warehouse_info, f, indent=2)

    return project_dir


if __name__ == "__main__":
    block_analysis_file = "block_analysis_ViT_20250221_155726.txt"
    project_dir = create_warehouse_structure(block_analysis_file)
    print(f"Warehouse structure created at: {project_dir}")
    print("\nStructure created:")
    print("project/")
    print("└── warehouse/")
    for block in sorted(os.listdir(project_dir / "warehouse")):
        if block.startswith("block_"):
            print(f"    ├── {block}/")
            block_path = project_dir / "warehouse" / block
            for file in sorted(os.listdir(block_path)):
                print(f"    │   ├── {file}")
    print("    └── warehouse_info.json") 
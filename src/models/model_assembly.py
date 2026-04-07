import os
import json
import torch
from pathlib import Path
from typing import List, Dict, Any


def assemble_model(path_to_saved_model: str, block_config: List[int], output_path: str) -> None:
    """Assemble a model by combining different variants according to the block_config.
    
    Args:
        path_to_saved_model: Path to the original saved model or pretrained weights
        block_config: List of variant numbers, with length matching number of blocks
        output_path: Path where to save the assembled model
        
    The block_config list should have the same length as the number of blocks in variant_info.json,
    and each number should be within the valid range of variants for that block.
    """
    # Load the original model state dict
    original_state_dict = torch.load(path_to_saved_model, map_location='cpu')
    
    # Get the experiment root directory (two levels up from the model file)
    experiment_dir = Path(path_to_saved_model).parent.parent
    variants_dir = experiment_dir / "variants"
    
    # Load variant info
    with open(variants_dir / "variant_info.json", 'r') as f:
        variant_info = json.load(f)
    
    # Validate block_config length
    num_blocks = len(variant_info['blocks'])
    if len(block_config) != num_blocks:
        raise ValueError(f"BlockConfig length ({len(block_config)}) does not match number of blocks ({num_blocks})")
    
    # Create new state dict for the assembled model
    assembled_state_dict = {}
    
    # For each block, load the specified variant and update the state dict
    for block_id, variant_id in enumerate(block_config, start=1):
        block_info = variant_info['blocks'][str(block_id)]
        
        # Validate variant number
        if variant_id < 1 or variant_id > block_info['num_variants']:
            raise ValueError(f"Invalid variant number {variant_id} for block {block_id}. "
                           f"Must be between 1 and {block_info['num_variants']}")
        
        # Get the variant state dict path
        variant_path = block_info['variants'][variant_id - 1]['state_dict']
        
        # Load the variant state dict
        variant_state_dict = torch.load(variant_path, map_location='cpu')
        
        # Update the assembled state dict with this variant's parameters
        assembled_state_dict.update(variant_state_dict)
    
    # Save the assembled model
    torch.save(assembled_state_dict, output_path)


def test_assemble_model():
    """Test the assemble_model function using actual MNIST experiment files."""
    # Define paths
    path_to_saved_model = "experiment/MNIST/models/resnet18_mnist_init.pth"
    output_path = "experiment/MNIST/models/current_submodel_model.pth"
    
    # Create block_config with all 1's (first variant for each block)
    # From the variant_info.json we know there are 5 blocks
    block_config = [1, 1, 1, 1, 1]
    
    try:
        print("Testing assemble_model with actual MNIST experiment files...")
        assemble_model(path_to_saved_model, block_config, output_path)
        print(f"✓ Model assembled successfully")
        print(f"✓ Assembleed model saved to: {output_path}")
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")


if __name__ == "__main__":
    test_assemble_model() 
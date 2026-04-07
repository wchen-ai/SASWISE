"""Creator module for creating variant folders."""

import re
import json
import torch
from pathlib import Path
from typing import Dict, Any, List, Set


def get_block_nodes(block_analysis_path: str) -> Dict[int, Dict[str, Any]]:
    """Extract block information from block analysis file.
    
    Args:
        block_analysis_path: Path to the block analysis file
        
    Returns:
        dict: Block information with nodes and variants
    """
    blocks = {}
    current_block = None
    
    with open(block_analysis_path, 'r') as f:
        for line in f:
            if line.startswith("Block "):
                # Extract block number and variants
                block_match = re.match(r"Block (\d+)<(\d+)>:", line)
                if block_match:
                    current_block = int(block_match.group(1))
                    num_variants = int(block_match.group(2))
                    blocks[current_block] = {
                        'num_variants': num_variants,
                        'nodes': set()
                    }
            elif current_block is not None and line.strip().startswith("- "):
                # Extract node paths
                node = line.strip("- \n")
                blocks[current_block]['nodes'].add(node)
                
    return blocks


def create_variant_folders(block_analysis_path: str, root_dir: str, state_dict_path: str) -> Dict[str, Any]:
    """Create variant folders and state dicts based on block analysis.
    
    Args:
        block_analysis_path: Path to the block analysis file
        root_dir: Root directory where variants will be created
        state_dict_path: Path to the original state dict file
        
    Returns:
        dict: Information about created variant structure
    """
    # Create variant directories
    variants_dir = Path(root_dir) / "variants"
    variants_dir.mkdir(exist_ok=True)
    
    # Load original state dict
    state_dict = torch.load(state_dict_path, map_location='cpu')
    
    # Get block information
    blocks = get_block_nodes(block_analysis_path)
    
    variant_info = {
        'root_dir': str(variants_dir),
        'blocks': {}
    }
    
    # Create block directories and state dicts
    for block_id, block_info in blocks.items():
        block_dir = variants_dir / f"block_{block_id}"
        block_dir.mkdir(exist_ok=True)
        
        variant_info['blocks'][block_id] = {
            'num_variants': block_info['num_variants'],
            'nodes': list(block_info['nodes']),
            'variants': []
        }
        
        # Create state dict for each variant
        for variant in range(1, block_info['num_variants'] + 1):
            # Create a copy of state dict with only the nodes for this block
            block_state_dict = {
                key: value for key, value in state_dict.items()
                if any(key.startswith(node + '.') or key == node for node in block_info['nodes'])
            }
            
            # Save state dict
            state_dict_file = block_dir / f"variant_{variant}_state_dict.pt"
            torch.save(block_state_dict, state_dict_file)
            
            variant_info['blocks'][block_id]['variants'].append({
                'variant': variant,
                'state_dict': str(state_dict_file),
                'status': "initialized"
            })
    
    # Save variant info to a single JSON file
    info_file = variants_dir / "variant_info.json"
    with open(info_file, 'w') as f:
        json.dump(variant_info, f, indent=2)
    
    return variant_info 
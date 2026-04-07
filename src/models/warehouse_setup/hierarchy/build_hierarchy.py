"""Build hierarchy from model state dictionary.

This module provides functions to build a hierarchical representation of a model's
structure from its state dictionary. The hierarchy preserves the natural structure
of layers and components while tracking parameter counts at each level.

Example hierarchy structure:
{
    'layer1': {
        'params': ['layer1.weight', 'layer1.bias'],
        'children': {
            'conv': {...},
            'bn': {...}
        }
    }
}
"""

from typing import Dict, Any, Tuple
import torch
from collections import defaultdict


def build_model_hierarchy_from_state_dict(state_dict_path: str) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """Build a hierarchical representation of model structure from a state dictionary file.
    
    Args:
        state_dict_path: Path to the PyTorch state dictionary file
        
    Returns:
        Tuple containing:
        - hierarchy: Dictionary representing the model's hierarchical structure
        - param_counts: Dictionary containing parameter counts for each node
        
    Example:
        >>> hierarchy, counts = build_model_hierarchy_from_state_dict('model.pt')
        >>> print(counts['layer1.conv'])  # Parameters in layer1's conv
    """
    # Load state dict
    state_dict = torch.load(state_dict_path, map_location='cpu')
    
    # Initialize hierarchy and parameter counts
    hierarchy = {}
    param_counts = defaultdict(int)
    
    # First pass: Count parameters for each node
    for key, param in state_dict.items():
        parts = key.split('.')
        current_path = []
        
        for part in parts[:-1]:  # Exclude the parameter name
            current_path.append(part)
            path_str = '.'.join(current_path)
            param_counts[path_str] += param.numel()
    
    # Second pass: Build hierarchy
    for key in state_dict.keys():
        parts = key.split('.')
        current_dict = hierarchy
        current_path = []
        
        for part in parts[:-1]:  # Exclude the parameter name
            current_path.append(part)
            path_str = '.'.join(current_path)
            
            if path_str not in current_dict:
                current_dict[path_str] = {
                    'params': [],
                    'children': {}
                }
            
            # Add parameter to the immediate parent's param list
            if part == parts[-2]:  # If this is the immediate parent
                current_dict[path_str]['params'].append(key)
            
            current_dict = current_dict[path_str]['children']
    
    return hierarchy, dict(param_counts) 
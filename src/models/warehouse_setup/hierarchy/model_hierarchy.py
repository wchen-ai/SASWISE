from typing import Dict, Any
import torch
from collections import defaultdict


class ModelHierarchy:
    """Class representing the hierarchical structure of a model."""
    
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.hierarchy = self._build_hierarchy()
        self.parameter_counts = self._count_parameters()
    
    def _build_hierarchy(self) -> Dict[str, Any]:
        """Build a tree structure representing the model hierarchy."""
        hierarchy = {}
        
        for name, module in self.model.named_modules():
            if not name:  # Skip root
                continue
                
            parts = name.split('.')
            current = hierarchy
            
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            current[parts[-1]] = {
                'type': type(module).__name__,
                'children': {}
            }
        
        return hierarchy
    
    def _count_parameters(self) -> Dict[str, int]:
        """Count parameters for each node in the hierarchy."""
        counts = defaultdict(int)
        
        for name, param in self.model.named_parameters():
            module_path = '.'.join(name.split('.')[:-1])
            counts[module_path] += param.numel()
            
            # Add to parent counts
            while '.' in module_path:
                module_path = '.'.join(module_path.split('.')[:-1])
                counts[module_path] += param.numel()
        
        return dict(counts)
    
    def get_node_info(self, node_path: str) -> Dict[str, Any]:
        """Get information about a specific node in the hierarchy."""
        parts = node_path.split('.')
        current = self.hierarchy
        
        for part in parts:
            if part not in current:
                raise ValueError(f"Invalid node path: {node_path}")
            current = current[part]
        
        return {
            'type': current['type'],
            'parameters': self.parameter_counts.get(node_path, 0),
            'children': list(current['children'].keys())
        } 
"""Save model hierarchy to a file for block annotation.

This module provides functions to save a model's hierarchical structure to a text file
in a format that's easy to read and annotate. Each node in the hierarchy is written
with empty brackets [ ] where users can add block indices.

Example output format:
    - layer1 [ ] (params: 64)
      - conv [ ] (params: 32)
      - bn [ ] (params: 32)
"""

from typing import Dict, Any, Optional
from pathlib import Path


def save_model_hierarchy(hierarchy: Dict[str, Any], param_counts: Dict[str, int], 
                        output_path: str, root_dir: Optional[str] = None) -> None:
    """Save the model hierarchy to a text file with brackets for block indexing.
    
    Args:
        hierarchy: Dictionary representing the model's hierarchical structure
        param_counts: Dictionary containing parameter counts for each node
        output_path: Path where to save the hierarchy text file
        root_dir: Optional root directory to prepend to output_path
        
    Example:
        >>> save_model_hierarchy(hierarchy, counts, 'model_hierarchy.txt')
        # Creates a file with nodes and parameter counts:
        # - layer1 [ ] (params: 64)
        #   - conv [ ] (params: 32)
        #   - bn [ ] (params: 32)
    """
    if root_dir:
        output_path = str(Path(root_dir) / output_path)
        Path(root_dir).mkdir(parents=True, exist_ok=True)
        
    with open(output_path, 'w') as f:
        f.write("Model Hierarchy:\n")
        f.write("Instructions: Add block indices in the brackets [ ] for nodes you want to group.\n")
        f.write("Rules:\n")
        f.write("1. If a node has an index, all its children belong to that block\n")
        f.write("2. If multiple siblings have the same index, they form one block\n")
        f.write("3. Empty brackets [ ] mean this node will inherit its block index from its parent node\n\n")
        
        def write_hierarchy(d, level=0):
            def sort_key(k):
                # Split the key into parts
                parts = k.split('.')
                # Convert numerical parts to integers for sorting
                return [int(p) if p.isdigit() else p for p in parts]
            
            for k, v in sorted(d.items(), key=lambda x: sort_key(x[0])):
                indent = '  ' * level
                f.write(f"{indent}- {k} [ ] (params: {param_counts.get(k, 0):,})\n")
                write_hierarchy(v['children'], level + 1)
        
        write_hierarchy(hierarchy) 
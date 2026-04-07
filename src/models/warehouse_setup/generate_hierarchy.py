#!/usr/bin/env python3
"""Generate Model Hierarchy Tool.

This script generates a hierarchical representation of a neural network model's structure
from its state dictionary. The hierarchy is saved in a text file where users can annotate
each node with block indices.

Example usage:
    python -m src.models.warehouse_setup.generate_hierarchy \\
        --state_dict path/to/model.pt \\
        --out output/directory

The output file will be named 'model_hierarchy_<model>_<timestamp>.txt' and will contain:
- Model layer hierarchy with parameter counts
- Empty brackets [ ] for block annotation
- Instructions for annotation
"""

import argparse
from pathlib import Path
from datetime import datetime
from .hierarchy.build_hierarchy import build_model_hierarchy_from_state_dict
from .hierarchy.save_hierarchy import save_model_hierarchy


def main():
    parser = argparse.ArgumentParser(description='Generate model hierarchy from state dictionary')
    parser.add_argument('--state_dict', type=str, required=True,
                      help='Path to the model state dict file')
    parser.add_argument('--out', type=str, required=True,
                      help='Directory where the hierarchy file will be saved')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Generate hierarchy from state dict
        print(f"Building hierarchy from state dict: {args.state_dict}")
        hierarchy, param_counts = build_model_hierarchy_from_state_dict(args.state_dict)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"model_hierarchy_ResNet18_{timestamp}.txt"
        output_path = output_dir / output_file
        
        # Save hierarchy with annotation placeholders
        save_model_hierarchy(hierarchy, param_counts, str(output_path))
        
        print(f"\nHierarchy has been saved to: {output_path}")
        print("\nPlease annotate the hierarchy file by:")
        print("1. Adding block indices in the brackets [ ]")
        print("2. Use empty brackets [ ] for nodes that should inherit from their parent")
        print("3. Group components logically (e.g., conv layers, batch norms, etc.)")
        print(f"\nTotal model parameters: {sum(count for count in param_counts.values()):,}")
        
        return str(output_path)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None


if __name__ == "__main__":
    main() 
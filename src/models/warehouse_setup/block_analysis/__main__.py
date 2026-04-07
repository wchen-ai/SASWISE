"""Main entry point for block analysis."""

import argparse
from pathlib import Path
from .parser import parse_user_block_indexing
from .analyzer import analyze_block_parameters
from .saver import save_block_analysis


def main():
    parser = argparse.ArgumentParser(description='Generate block analysis from annotated hierarchy')
    parser.add_argument('--model_hierarchy', type=str, required=True,
                      help='Path to the annotated model hierarchy file')
    parser.add_argument('--out', type=str, required=True,
                      help='Directory where the block analysis will be saved')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        print(f"Analyzing block assignments in: {args.model_hierarchy}")
        
        # Parse block indexing from hierarchy file
        block_assignment, void_nodes, overlap_nodes, empty_box_nodes = parse_user_block_indexing(args.model_hierarchy)
        
        # Analyze block parameters
        analysis = analyze_block_parameters(args.model_hierarchy, block_assignment, void_nodes, overlap_nodes, empty_box_nodes)
        
        # Generate output filename based on input
        input_name = Path(args.model_hierarchy).name
        output_file = input_name.replace("model_hierarchy", "block_analysis")
        output_path = output_dir / output_file
        
        # Save block analysis
        save_block_analysis(analysis, str(output_path))
        print(f"\nBlock analysis has been saved to: {output_path}")
        
        # Return the output path for potential chaining
        return str(output_path)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None


if __name__ == "__main__":
    main() 
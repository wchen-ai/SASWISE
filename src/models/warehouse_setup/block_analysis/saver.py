"""Saver module for block analysis."""

from typing import Dict, Any


def save_block_analysis(analysis: Dict[str, Any], output_file: str) -> None:
    """Save the block analysis results to a file.
    
    Args:
        analysis (dict): Analysis results from analyze_block_parameters
        output_file (str): Path to save the analysis results
    """
    with open(output_file, 'w') as f:
        f.write("Block Analysis:\n\n")

        # Write block information with variant counts
        for block, info in sorted(analysis['block_info'].items()):
            # Default to 1 variant if not specified
            variants = info.get('variants', 1)
            f.write(f"Block {block}<{variants}>:\n")
            f.write(f"Total Parameters: {info['total_params']:,}\n")
            f.write("Nodes:\n")
            for node in sorted(info['nodes']):
                f.write(f"  - {node}\n")
            f.write("\n")

        # Write empty box information
        if analysis['empty_box_info']['nodes']:
            f.write("\nEmpty Box Nodes (all descendants are assigned or empty boxes):\n")
            for node in sorted(analysis['empty_box_info']['nodes']):
                f.write(f"  - {node}\n")
            f.write("\n")

        # Write void node information
        if analysis['void_info']['nodes']:
            f.write("\nNodes with no block assignment:\n")
            f.write(f"Total Parameters: {analysis['void_info']['total_params']:,}\n")
            for node in sorted(analysis['void_info']['nodes']):
                f.write(f"  - {node}\n")
            f.write("\n")

        # Write overlap information
        if analysis['overlap_info']:
            f.write("\nNodes assigned to multiple blocks:\n")
            for node, info in sorted(analysis['overlap_info'].items()):
                f.write(f"  - {node} (blocks: {info['blocks']}, params: {info['params']:,})\n")
            f.write("\n")

        # Write validation information
        f.write("\nParameter Validation:\n")
        f.write(f"Total leaf node parameters across all blocks: {analysis['validation_info']['leaf_params']:,}\n")
        f.write(f"Total number of leaf nodes: {len(analysis['validation_info']['all_leaf_nodes'])}\n")
        f.write(f"Total parameters across all blocks: {analysis['total_params']:,}\n") 
from typing import Dict, Any, Optional, Tuple, List
import torch
from collections import defaultdict
import argparse
from datetime import datetime
from pathlib import Path
import re
import json

from src.models.model_loader import load_pretrained_model, get_model_info


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


def build_warehouse(
    model_path: str,
    model_type: str,
    device: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load pretrained weights and output the model hierarchy.
    
    Args:
        model_path: Path to pretrained model weights
        model_type: Type of model to load
        device: Device to load model on
    
    Returns:
        Dictionary containing:
        - model: The loaded model
        - hierarchy: ModelHierarchy instance
        - info: General model information
    """
    # Load the model
    model = load_pretrained_model(model_path, model_type, device)
    
    # Create hierarchy
    hierarchy = ModelHierarchy(model)
    
    # Get general model info
    info = get_model_info(model)
    
    return {
        'model': model,
        'hierarchy': hierarchy,
        'info': info
    }


def build_model_hierarchy_from_state_dict(state_dict_path: str) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """
    Build a hierarchical representation of model structure from a state dictionary file.
    
    Args:
        state_dict_path: Path to the PyTorch state dictionary file
        
    Returns:
        Tuple containing:
        - hierarchy: Dictionary representing the model's hierarchical structure
        - param_counts: Dictionary containing parameter counts for each node
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


def save_model_hierarchy(hierarchy: Dict[str, Any], param_counts: Dict[str, int], output_path: str, root_dir: Optional[str] = None):
    """
    Save the model hierarchy to a text file with brackets for block indexing.
    
    Args:
        hierarchy: Dictionary representing the model's hierarchical structure
        param_counts: Dictionary containing parameter counts for each node
        output_path: Path where to save the hierarchy text file
        root_dir: Optional root directory to prepend to output_path
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


def parse_user_block_indexing(hierarchy_file):
    """Parse the user's block indexing from the hierarchy file.
    
    Args:
        hierarchy_file (str): Path to the hierarchy file with user's block indexing
        
    Returns:
        tuple: (block_assignment, void_nodes, overlap_nodes, empty_box_nodes)
            - block_assignment: Dictionary mapping block indices to lists of node paths
            - void_nodes: List of nodes with no block assignment and not all descendants indexed
            - overlap_nodes: Dictionary mapping nodes to their multiple block assignments
            - empty_box_nodes: List of nodes that are empty boxes (all descendants are assigned or empty boxes)
    """
    block_assignment = {}  # Maps block index to list of nodes
    node_blocks = {}  # Maps node to list of blocks it belongs to
    void_nodes = []
    node_children = {}  # Maps node to its children
    node_descendants = defaultdict(set)  # Maps node to all its descendants
    empty_box_nodes = []  # Nodes that are empty boxes
    
    def get_parent_block(node_path):
        # Split the path into components
        parts = node_path.split('.')
        # Try increasingly shorter paths until we find a parent with a block
        while len(parts) > 1:
            parts.pop()
            parent = '.'.join(parts)
            if parent in node_blocks and node_blocks[parent]:
                return parent, node_blocks[parent][0]  # Return parent path and its first block
        return None, None

    # First pass: collect node relationships and explicit block assignments
    with open(hierarchy_file, 'r') as f:
        for line in f:
            if '[' not in line or ']' not in line:
                continue
            
            # Skip header lines
            if line.strip().startswith(('Instructions', 'Rules', 'Model')):
                continue
            
            # Extract node path and block index
            node = line.split('[')[0].strip('- \t')
            block_str = line.split('[')[1].split(']')[0].strip()
            
            # Record parent-child relationship and build descendant tree
            if '.' in node:
                parts = node.split('.')
                # Add node as descendant to all its ancestors
                for i in range(1, len(parts)):
                    ancestor = '.'.join(parts[:i])
                    node_descendants[ancestor].add(node)
                
                # Record immediate parent-child relationship
                parent = '.'.join(parts[:-1])
                if parent not in node_children:
                    node_children[parent] = []
                node_children[parent].append(node)
            
            # Record block assignment
            if block_str:  # Has explicit block index
                try:
                    block = int(block_str)
                    if block not in block_assignment:
                        block_assignment[block] = []
                    block_assignment[block].append(node)
                    if node not in node_blocks:
                        node_blocks[node] = []
                    node_blocks[node].append(block)
                except ValueError:
                    continue
            else:  # Empty brackets
                node_blocks[node] = []
                void_nodes.append(node)
    
    # Second pass: handle inheritance and check for overlaps
    overlap_nodes = {}
    
    # First check explicit block assignments for overlaps with parents
    for node, blocks in node_blocks.items():
        if blocks:  # Node has explicit block assignment
            parent_node, parent_block = get_parent_block(node)
            if parent_block is not None and parent_block not in blocks:
                # Node has different block than parent - mark as overlap
                if node not in overlap_nodes:
                    overlap_nodes[node] = []
                overlap_nodes[node] = blocks + [parent_block]
                # Add to both blocks in block_assignment
                if parent_block not in block_assignment:
                    block_assignment[parent_block] = []
                if node not in block_assignment[parent_block]:
                    block_assignment[parent_block].append(node)
    
    # Third pass: identify empty box nodes
    def is_empty_box(node):
        descendants = node_descendants.get(node, set())
        if not descendants:
            # If this is a leaf node with empty brackets, it's not an empty box
            return False
        
        # Get all descendants that have block assignments
        assigned_descendants = set()
        for block_nodes in block_assignment.values():
            assigned_descendants.update(set(block_nodes))
        
        # Check if all immediate children are either:
        # 1. Assigned to a block
        # 2. Already marked as empty boxes
        # 3. Have all their descendants assigned
        children = node_children.get(node, [])
        for child in children:
            if child in assigned_descendants:
                continue
            if child in empty_box_nodes:
                continue
            # Check if all descendants of this child are assigned
            child_descendants = node_descendants.get(child, set())
            if not child_descendants or not all(desc in assigned_descendants for desc in child_descendants):
                return False
        return True
    
    # Iterate through nodes multiple times to handle nested empty boxes
    max_iterations = 10  # Prevent infinite loops
    for _ in range(max_iterations):
        found_new_empty_box = False
        for node in list(void_nodes):  # Use list to avoid modifying during iteration
            if is_empty_box(node):
                empty_box_nodes.append(node)
                void_nodes.remove(node)
                found_new_empty_box = True
        if not found_new_empty_box:
            break
    
    # Handle inheritance for remaining void nodes
    for node in list(void_nodes):  # Use list to avoid modifying during iteration
        parent_node, parent_block = get_parent_block(node)
        if parent_block is not None:
            if parent_block not in block_assignment:
                block_assignment[parent_block] = []
            block_assignment[parent_block].append(node)
            node_blocks[node] = [parent_block]
            void_nodes.remove(node)
    
    return block_assignment, void_nodes, overlap_nodes, empty_box_nodes


def analyze_block_parameters(hierarchy_file, block_assignment, void_nodes, overlap_nodes, empty_box_nodes):
    """Analyze parameters for each block based on the hierarchy file and block assignments.
    
    Args:
        hierarchy_file (str): Path to the hierarchy file
        block_assignment (dict): Dictionary mapping block indices to lists of node paths
        void_nodes (list): List of nodes with no block assignment
        overlap_nodes (dict): Dictionary mapping nodes to their multiple block assignments
        empty_box_nodes (list): List of nodes that are empty boxes
        
    Returns:
        dict: Analysis results containing:
            - block_info: Dictionary mapping block indices to their parameter counts and nodes
            - void_info: Information about void nodes and their parameters
            - overlap_info: Information about overlapping nodes
            - empty_box_info: Information about empty box nodes (no parameter counting)
            - total_params: Total parameters across all blocks (leaf nodes only)
            - validation_info: Comparison between block parameters and actual model parameters
    """
    analysis = {
        'block_info': {},
        'void_info': {'total_params': 0, 'nodes': []},
        'overlap_info': {},
        'empty_box_info': {'nodes': []},  # No parameter counting for empty boxes
        'total_params': 0,
        'validation_info': {'leaf_params': 0, 'all_leaf_nodes': set()}
    }
    
    # Extract parameter counts and build node relationships
    node_params = {}
    node_children = defaultdict(list)
    all_nodes = set()
    
    with open(hierarchy_file, 'r') as f:
        for line in f:
            if '[' not in line or ']' not in line:
                continue
            if line.strip().startswith(('Instructions', 'Rules', 'Model')):
                continue
                
            # Extract node and parameters
            node = line.split('[')[0].strip('- ')
            if '(params:' in line:
                params = int(line.split('(params:')[1].split(')')[0].strip().replace(',', ''))
                node_params[node] = params
            
            # Build parent-child relationships
            all_nodes.add(node)
            if '.' in node:
                parent = '.'.join(node.split('.')[:-1])
                node_children[parent].append(node)
    
    # Identify leaf nodes (nodes with no children)
    leaf_nodes = {node for node in all_nodes if node not in node_children}
    analysis['validation_info']['all_leaf_nodes'] = leaf_nodes
    
    # Analyze parameters for each block (counting only leaf nodes)
    for block, nodes in block_assignment.items():
        block_leaf_params = 0
        block_leaf_nodes = []
        
        for node in nodes:
            # If node is a leaf node, count its parameters
            if node in leaf_nodes and node in node_params:
                block_leaf_params += node_params[node]
                block_leaf_nodes.append(node)
                analysis['validation_info']['leaf_params'] += node_params[node]
        
        analysis['block_info'][block] = {
            'total_params': block_leaf_params,
            'nodes': nodes,
            'leaf_nodes': block_leaf_nodes
        }
        analysis['total_params'] += block_leaf_params
    
    # Analyze void nodes (only leaf nodes)
    for node in void_nodes:
        if node in leaf_nodes and node in node_params:
            analysis['void_info']['total_params'] += node_params[node]
            analysis['void_info']['nodes'].append(node)
    
    # Record empty box nodes (without parameter counting)
    analysis['empty_box_info']['nodes'] = empty_box_nodes
    
    # Analyze overlapping nodes (only leaf nodes)
    for node, blocks in overlap_nodes.items():
        if node in leaf_nodes and node in node_params:
            analysis['overlap_info'][node] = {
                'blocks': blocks,
                'params': node_params[node]
            }
    
    return analysis


def save_block_analysis(analysis, output_file):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Warehouse setup with multiple modes')
    
    # Add mutually exclusive arguments for different modes
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--state_dict_path', type=str,
                         help='Path to the model state dict file to generate hierarchy')
    mode_group.add_argument('--model_hierarchy_path', type=str,
                         help='Path to the annotated model hierarchy file to generate block analysis')
    mode_group.add_argument('--block_plating_path', type=str,
                         help='Path to the block analysis file to create variant folders')
    
    # Add root directory argument
    parser.add_argument('--root_dir', type=str, default='experiment/MNIST',
                      help='Root directory for all outputs')
    
    args = parser.parse_args()
    root_dir = Path(args.root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)

    try:
        if args.state_dict_path:
            # Mode 1: Generate hierarchy from state dict
            print(f"Building hierarchy from state dict: {args.state_dict_path}")
            hierarchy, param_counts = build_model_hierarchy_from_state_dict(args.state_dict_path)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"model_hierarchy_ResNet18_{timestamp}.txt"
            
            # Save hierarchy in the root directory
            save_model_hierarchy(hierarchy, param_counts, output_file, args.root_dir)
            
            print(f"\nHierarchy has been saved to: {root_dir / output_file}")
            print("\nPlease annotate the hierarchy file by:")
            print("1. Adding block indices in the brackets [ ]")
            print("2. Use empty brackets [ ] for nodes that should inherit from their parent")
            print("3. Group components logically (e.g., conv layers, batch norms, etc.)")
            print(f"\nTotal model parameters: {sum(count for count in param_counts.values()):,}")

        elif args.model_hierarchy_path:
            # Mode 2: Generate block analysis from annotated hierarchy
            print(f"Analyzing block assignments in: {args.model_hierarchy_path}")
            
            # Parse block indexing
            block_assignment, void_nodes, overlap_nodes, empty_box_nodes = parse_user_block_indexing(args.model_hierarchy_path)
            
            # Analyze block parameters
            analysis = analyze_block_parameters(args.model_hierarchy_path, block_assignment, void_nodes, overlap_nodes, empty_box_nodes)
            
            # Generate output filename based on input
            output_file = Path(args.model_hierarchy_path).name.replace("model_hierarchy", "block_analysis")
            output_path = root_dir / output_file
            
            # Save block analysis
            save_block_analysis(analysis, str(output_path))
            print(f"\nBlock analysis has been saved to: {output_path}")

        else:  # args.block_plating_path
            # Mode 3: Create block variant folders
            print(f"Creating block variant structure from: {args.block_plating_path}")
            
            # Create variant directories
            variants_dir = root_dir / "variants"
            variants_dir.mkdir(exist_ok=True)
            
            # Parse the block analysis file and create variant structure
            with open(args.block_plating_path, 'r') as f:
                for line in f:
                    if line.startswith("Block "):
                        # Extract block number and variants
                        block_match = re.match(r"Block (\d+)<(\d+)>:", line)
                        if block_match:
                            block_id = block_match.group(1)
                            num_variants = int(block_match.group(2))
                            
                            # Create block directory
                            block_dir = variants_dir / f"block_{block_id}"
                            block_dir.mkdir(exist_ok=True)
                            
                            # Create variant placeholders
                            for variant in range(1, num_variants + 1):
                                variant_file = block_dir / f"variant_{variant}.json"
                                with open(variant_file, 'w') as sf:
                                    json.dump({
                                        "block": int(block_id),
                                        "variant": variant,
                                        "status": "initialized"
                                    }, sf, indent=2)
            
            print(f"\nBlock variant structure created at: {variants_dir}")
            print("\nDirectory structure:")
            print(f"{variants_dir}/")
            for block_dir in sorted(variants_dir.glob("block_*")):
                print(f"├── {block_dir.name}/")
                for variant_file in sorted(block_dir.glob("variant_*.json")):
                    print(f"│   ├── {variant_file.name}")
            print("└── (end)")

    except Exception as e:
        print(f"Error: {str(e)}") 
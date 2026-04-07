"""Analyzer module for block analysis."""

from collections import defaultdict
from typing import Dict, List, Any


def analyze_block_parameters(hierarchy_file: str, block_assignment: Dict[int, List[str]], 
                           void_nodes: List[str], overlap_nodes: Dict[str, List[int]], 
                           empty_box_nodes: List[str]) -> Dict[str, Any]:
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
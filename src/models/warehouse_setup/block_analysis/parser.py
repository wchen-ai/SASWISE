"""Parser module for block analysis."""

from collections import defaultdict
from typing import Dict, List, Tuple, Set


def parse_user_block_indexing(hierarchy_file: str) -> Tuple[Dict[int, List[str]], List[str], Dict[str, List[int]], List[str]]:
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
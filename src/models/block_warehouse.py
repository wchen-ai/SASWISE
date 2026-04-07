import os
import re
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional
import shutil
import json


class BlockWarehouse:
    """Class for managing block variants and building submodels."""
    
    def __init__(self, model: torch.nn.Module, block_analysis_file: str, variants_dir: str = "block_variants"):
        """
        Initialize BlockWarehouse.
        
        Args:
            model: The base model to create variants from
            block_analysis_file: Path to the block analysis file
            variants_dir: Directory to store block variants
        """
        self.model = model
        self.block_analysis_file = block_analysis_file
        self.variants_dir = Path(variants_dir)
        self.variants_dir.mkdir(parents=True, exist_ok=True)
        
        # Load block analysis
        self.blocks = self._load_block_analysis()
        
        # Create variant management system
        self.variant_registry = {}
        self._initialize_variant_registry()
    
    def _load_block_analysis(self) -> Dict[int, Dict[str, Any]]:
        """Load and parse the block analysis file."""
        blocks = {}
        current_block = None
        
        with open(self.block_analysis_file, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            # Match block header with variant count
            block_match = re.match(r"Block (\d+)<(\d+)>:", line)
            if block_match:
                current_block = int(block_match.group(1))
                variants = int(block_match.group(2))
                blocks[current_block] = {
                    'variants': variants,
                    'nodes': [],
                    'parameters': 0
                }
                continue
            
            # Match parameter count
            if current_block and "Total Parameters:" in line:
                param_count = int(line.split(":")[1].strip().replace(",", ""))
                blocks[current_block]['parameters'] = param_count
                continue
            
            # Match nodes
            if current_block and line.strip().startswith("- "):
                node = line.strip()[2:].strip()
                blocks[current_block]['nodes'].append(node)
        
        return blocks
    
    def _initialize_variant_registry(self):
        """Initialize the variant registry with default entries."""
        for block_id, block_info in self.blocks.items():
            self.variant_registry[block_id] = {
                'total_variants': block_info['variants'],
                'available_variants': []
            }
    
    def _get_block_state_dict(self, block_id: int) -> Dict[str, torch.Tensor]:
        """Extract state dict for a specific block."""
        block_state = {}
        block_nodes = self.blocks[block_id]['nodes']
        
        full_state = self.model.state_dict()
        for node in block_nodes:
            # Find all parameters belonging to this node
            for param_name, param in full_state.items():
                if param_name.startswith(node):
                    block_state[param_name] = param.clone()
        
        return block_state
    
    def create_variant(self, block_id: int, variant_id: int):
        """
        Create a variant (copy) of a block.
        
        Args:
            block_id: Block number
            variant_id: Variant number to create
        """
        if block_id not in self.blocks:
            raise ValueError(f"Block {block_id} not found")
            
        if variant_id > self.blocks[block_id]['variants']:
            raise ValueError(f"Variant {variant_id} exceeds maximum variants for block {block_id}")
        
        # Extract block state
        block_state = self._get_block_state_dict(block_id)
        
        # Save variant
        variant_path = self.variants_dir / f"block_{block_id}_variant_{variant_id}.pt"
        torch.save(block_state, variant_path)
        
        # Update registry
        if variant_id not in self.variant_registry[block_id]['available_variants']:
            self.variant_registry[block_id]['available_variants'].append(variant_id)
    
    def create_all_variants(self):
        """Create all possible variants for all blocks."""
        for block_id, block_info in self.blocks.items():
            for variant_id in range(1, block_info['variants'] + 1):
                self.create_variant(block_id, variant_id)
    
    def build_submodel(self, block_config: Dict[int, int]) -> Dict[str, torch.Tensor]:
        """
        Build a submodel from specified variants.
        
        Args:
            block_config: Dictionary mapping block numbers to variant numbers
            
        Returns:
            Combined state dict for the specified submodel
        """
        submodel_state = {}
        
        # Validate block_config
        for block_id, variant_id in block_config.items():
            if block_id not in self.blocks:
                raise ValueError(f"Block {block_id} not found")
            if variant_id not in self.variant_registry[block_id]['available_variants']:
                raise ValueError(f"Variant {variant_id} not available for block {block_id}")
        
        # Combine variants
        for block_id, variant_id in block_config.items():
            variant_path = self.variants_dir / f"block_{block_id}_variant_{variant_id}.pt"
            block_state = torch.load(variant_path)
            submodel_state.update(block_state)
        
        return submodel_state
    
    def save_submodel(self, block_config: Dict[int, int], output_path: str):
        """
        Save a complete submodel to a file.
        
        Args:
            block_config: Dictionary mapping block numbers to variant numbers
            output_path: Path to save the submodel state dict
        """
        submodel_state = self.build_submodel(block_config)
        torch.save(submodel_state, output_path)
    
    def list_available_variants(self) -> Dict[int, List[int]]:
        """
        List all available variants for each block.
        
        Returns:
            Dictionary mapping block numbers to lists of available variant numbers
        """
        return {
            block_id: info['available_variants']
            for block_id, info in self.variant_registry.items()
        }
    
    def get_variant_info(self, block_id: int, variant_id: int) -> Dict[str, Any]:
        """
        Get information about a specific variant.
        
        Args:
            block_id: Block number
            variant_id: Variant number
            
        Returns:
            Dictionary containing variant information
        """
        if block_id not in self.blocks:
            raise ValueError(f"Block {block_id} not found")
            
        variant_path = self.variants_dir / f"block_{block_id}_variant_{variant_id}.pt"
        if not variant_path.exists():
            raise ValueError(f"Variant {variant_id} not found for block {block_id}")
            
        return {
            'block': block_id,
            'variant': variant_id,
            'path': str(variant_path),
            'parameters': self.blocks[block_id]['parameters'],
            'nodes': self.blocks[block_id]['nodes']
        }
    
    def cleanup(self):
        """Remove all variant files and the variants directory."""
        if self.variants_dir.exists():
            shutil.rmtree(self.variants_dir)
        self._initialize_variant_registry()
    
    def save_division_info(self, output_file):
        """Save block division and variant information to a text file."""
        # First analyze the blocks
        self.analyze_blocks()
        
        with open(output_file, 'w') as f:
            f.write("Block Division and Variant Information:\n")
            f.write("=====================================\n\n")
            
            # Write total parameters
            f.write(f"Total Model Parameters: {self.total_params:,}\n\n")
            
            # Write block information
            f.write("Block Information:\n")
            f.write("=================\n")
            for block_id in sorted(self.block_info.keys()):
                info = self.block_info[block_id]
                f.write(f"\nBlock {block_id}:\n")
                f.write(f"  Parameters: {info['params']:,}\n")
                f.write(f"  Nodes: {len(info['nodes'])}\n")
                f.write("  Components:\n")
                for node in sorted(info['nodes']):
                    f.write(f"    - {node}\n")
            
            f.write("\nVariant Files Location:\n")
            f.write(f"  {self.variants_dir}\n\n")
            
            # Write example block_config format
            f.write("Example BlockConfig Format:\n")
            f.write("  block_config = {\n")
            for block_id in sorted(self.block_info.keys()):
                f.write(f"    {block_id}: 1,  # Select variant number for block {block_id}\n")
            f.write("  }\n")

    def analyze_blocks(self):
        """Analyze blocks from the hierarchy file and store information."""
        self.block_info = {}
        self.total_params = 0
        
        # Read and parse the hierarchy file
        with open(self.block_analysis_file, 'r') as f:
            lines = f.readlines()
        
        current_block = None
        for line in lines:
            if '[' in line and ']' in line:
                # Extract block number if present
                block_match = line.split('[')[1].split(']')[0].strip()
                if block_match and block_match.isdigit():
                    current_block = int(block_match)
                    if current_block not in self.block_info:
                        self.block_info[current_block] = {
                            'params': 0,
                            'nodes': set()
                        }
                
                # Extract node name and parameters
                node = line.split('[')[0].strip('- ').strip()
                if '(params:' in line:
                    params = int(line.split('params:')[1].split(')')[0].strip().replace(',', ''))
                    if current_block is not None:
                        self.block_info[current_block]['params'] += params
                        self.block_info[current_block]['nodes'].add(node)
                        self.total_params += params 
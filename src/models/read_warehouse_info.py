import json
from pathlib import Path

def read_warehouse_info(warehouse_path: str = "project/warehouse"):
    """Read and display warehouse information."""
    info_path = Path(warehouse_path) / "warehouse_info.json"
    
    with open(info_path, 'r') as f:
        info = json.load(f)
    
    print("Warehouse Info:")
    print(f"\nTotal Blocks: {info['total_blocks']}")
    print(f"Total Parameters: {info['total_parameters']:,}")
    
    print("\nModel Configuration:")
    for key, value in info['model_config'].items():
        print(f"  {key}: {value}")
    
    print("\nBlocks:")
    for block_id, block_info in sorted(info['blocks'].items()):
        print(f"\nBlock {block_id}:")
        print(f"  Number of Variants: {block_info['variants']}")
        print(f"  Nodes:")
        for node in block_info['nodes']:
            print(f"    - {node}")
        
        # Read and display block-specific info
        block_info_path = Path(warehouse_path) / f"block_{block_id}" / "info.json"
        with open(block_info_path, 'r') as f:
            detailed_info = json.load(f)
        print(f"  Total Parameters: {detailed_info['parameters']:,}")

if __name__ == "__main__":
    read_warehouse_info() 
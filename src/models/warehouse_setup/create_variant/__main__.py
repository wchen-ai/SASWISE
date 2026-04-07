"""Main entry point for creating variant folders."""

import argparse
from pathlib import Path
from .creator import create_variant_folders


def main():
    parser = argparse.ArgumentParser(description='Create variant folders from block analysis')
    parser.add_argument('--block_analysis', type=str, required=True,
                      help='Path to the block analysis file')
    parser.add_argument('--out', type=str, required=True,
                      help='Root directory where variants will be created')
    parser.add_argument('--state_dict', type=str, required=True,
                      help='Path to the original state dict file')
    
    args = parser.parse_args()
    
    try:
        print(f"Creating block variant structure from: {args.block_analysis}")
        print(f"Using state dict from: {args.state_dict}")
        
        # Create variant folders
        variant_info = create_variant_folders(args.block_analysis, args.out, args.state_dict)
        
        # Print directory structure
        variants_dir = Path(variant_info['root_dir'])
        print(f"\nBlock variant structure created at: {variants_dir}")
        print("\nDirectory structure:")
        print(f"{variants_dir}/")
        for block_id, block_info in sorted(variant_info['blocks'].items()):
            print(f"├── block_{block_id}/")
            for variant in block_info['variants']:
                state_dict_file = Path(variant['state_dict']).name
                print(f"│   ├── {state_dict_file}")
        print("├── variant_info.json")
        print("└── (end)")
        
        return variant_info
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None


if __name__ == "__main__":
    main() 
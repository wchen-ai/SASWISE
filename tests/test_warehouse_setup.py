import pytest
import torch
import tempfile
from pathlib import Path
from datetime import datetime
from monai.networks.nets import UNet, UNETR, ViT
from monai.networks.layers import Norm

from src.models.warehouse_setup import (
    build_warehouse,
    ModelHierarchy,
    build_model_hierarchy_from_state_dict,
    save_model_hierarchy,
    parse_user_block_indexing,
    analyze_block_parameters,
    save_block_analysis,
)


@pytest.fixture
def sample_unet_model():
    """Create a sample UNet model for testing."""
    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64),
        strides=(2, 2),
        num_res_units=2,
        norm=Norm.INSTANCE
    )
    return model


@pytest.fixture
def model_path(sample_unet_model):
    """Save a sample model and return its path."""
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        torch.save(sample_unet_model.state_dict(), f.name)
        return f.name


def test_model_state_dict():
    """Test that the model state dict is correctly structured."""
    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=2,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.INSTANCE,
    )

    # Get state dict
    state_dict = model.state_dict()
    
    # Build hierarchy based on numerical indices
    hierarchy = {}
    param_counts = {}
    
    for key in state_dict.keys():
        parts = key.split('.')
        # Find the numerical index parts
        current_dict = hierarchy
        current_path = []
        
        for part in parts:
            current_path.append(part)
            path_str = '.'.join(current_path)
            
            # If this part contains a number, it's a grouping level
            if any(c.isdigit() for c in part):
                if path_str not in current_dict:
                    current_dict[path_str] = {
                        'params': [],
                        'children': {}
                    }
                    param_counts[path_str] = 0
                current_dict = current_dict[path_str]['children']
                
                # Add parameter count
                param_counts[path_str] += state_dict[key].numel()
        
        # Add the parameter to its immediate parent's param list
        parent_path = '.'.join(current_path[:-1])
        if parent_path in hierarchy:
            hierarchy[parent_path]['params'].append(key)

    # Save hierarchy to file
    with open('model_hierarchy.txt', 'w') as f:
        f.write("Model Hierarchy:\n")
        
        def write_hierarchy(d, level=0, f=f):
            for k, v in sorted(d.items()):
                indent = '  ' * level
                f.write(f"{indent}- {k} (params: {param_counts.get(k, 0)})\n")
                if v['params']:
                    for param in sorted(v['params']):
                        f.write(f"{indent}  * {param}\n")
                write_hierarchy(v['children'], level + 1, f)
        
        write_hierarchy(hierarchy)

    # Verify key components are present
    assert any('model.0' in key for key in state_dict.keys()), "Input block not found"
    assert any('model.1.submodule' in key for key in state_dict.keys()), "Downsampling path not found"
    assert any('model.2' in key for key in state_dict.keys()), "Output block not found"
    assert any('.adn.' in key for key in state_dict.keys()), "ADN blocks not found"


def test_model_hierarchy_initialization(sample_unet_model):
    """Test ModelHierarchy initialization with a UNet model."""
    hierarchy = ModelHierarchy(sample_unet_model)
    
    # Test that hierarchy was created
    assert hierarchy.hierarchy is not None
    assert isinstance(hierarchy.hierarchy, dict)
    
    # Test that parameter counts were computed
    assert hierarchy.parameter_counts is not None
    assert isinstance(hierarchy.parameter_counts, dict)
    
    # Check for some expected UNet components in hierarchy
    assert any('down' in key for key in hierarchy.hierarchy.keys())
    assert any('up' in key for key in hierarchy.hierarchy.keys())


def test_model_hierarchy_parameter_counting(sample_unet_model):
    """Test parameter counting in ModelHierarchy."""
    hierarchy = ModelHierarchy(sample_unet_model)
    
    # Total parameters from hierarchy should match model's total parameters
    total_params_model = sum(p.numel() for p in sample_unet_model.parameters())
    total_params_hierarchy = sum(count for count in hierarchy.parameter_counts.values())
    
    assert total_params_hierarchy >= total_params_model  # >= because parameters are counted multiple times in hierarchy (for parent nodes)


def test_get_node_info(sample_unet_model):
    """Test getting information about specific nodes in the hierarchy."""
    hierarchy = ModelHierarchy(sample_unet_model)
    
    # Test getting info for an existing node (adjust path based on your UNet structure)
    # Example: first downsampling block
    node_info = hierarchy.get_node_info('down.0.conv')
    assert isinstance(node_info, dict)
    assert 'type' in node_info
    assert 'parameters' in node_info
    assert 'children' in node_info
    
    # Test getting info for non-existent node
    with pytest.raises(ValueError):
        hierarchy.get_node_info('nonexistent.path')


def test_warehouse_setup_with_unet(model_path):
    """Test build_warehouse function with a UNet model."""
    # Add UNet to the model types
    from src.models.model_loader import _get_model_class
    original_model_classes = _get_model_class.__globals__['model_classes'].copy()
    _get_model_class.__globals__['model_classes']['unet'] = lambda: UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64),
        strides=(2, 2),
        num_res_units=2,
        norm=Norm.INSTANCE
    )
    
    try:
        # Run warehouse setup
        warehouse = build_warehouse(
            model_path=model_path,
            model_type='unet',
            device='cpu'
        )
        
        # Check warehouse contents
        assert 'model' in warehouse
        assert 'hierarchy' in warehouse
        assert 'info' in warehouse
        
        # Check model type
        assert isinstance(warehouse['model'], UNet)
        
        # Check hierarchy
        assert isinstance(warehouse['hierarchy'], ModelHierarchy)
        
        # Check info
        assert 'total_parameters' in warehouse['info']
        assert 'trainable_parameters' in warehouse['info']
        assert 'layers' in warehouse['info']
        assert 'device' in warehouse['info']
        
    finally:
        # Restore original model classes
        _get_model_class.__globals__['model_classes'] = original_model_classes
        
        # Clean up temporary file
        Path(model_path).unlink()


def test_warehouse_setup_invalid_model_type():
    """Test build_warehouse with invalid model type."""
    with pytest.raises(ValueError):
        build_warehouse(
            model_path='dummy_path',
            model_type='invalid_type',
            device='cpu'
        )


def test_model_hierarchy_structure(sample_unet_model):
    """Test the structure of hierarchy created for UNet."""
    hierarchy = ModelHierarchy(sample_unet_model)
    
    # Helper function to check node properties
    def check_node_properties(node):
        assert 'type' in node
        assert 'children' in node
        assert isinstance(node['children'], dict)
    
    # Check main components of UNet
    main_hierarchy = hierarchy.hierarchy
    
    # Check encoder path (down)
    assert any(key.startswith('down') for key in main_hierarchy.keys())
    down_path = next(key for key in main_hierarchy.keys() if key.startswith('down'))
    check_node_properties(main_hierarchy[down_path])
    
    # Check decoder path (up)
    assert any(key.startswith('up') for key in main_hierarchy.keys())
    up_path = next(key for key in main_hierarchy.keys() if key.startswith('up'))
    check_node_properties(main_hierarchy[up_path])
    
    # Check for expected layer types
    layer_types = set()
    for name, module in sample_unet_model.named_modules():
        if name:  # Skip root
            parts = name.split('.')
            current = main_hierarchy
            for part in parts[:-1]:
                assert part in current, f"Missing path component: {part}"
                current = current[part]['children']
            assert parts[-1] in current, f"Missing leaf node: {parts[-1]}"
            layer_types.add(current[parts[-1]]['type'])
    
    # Check for essential UNet components
    essential_types = {'Conv2d', 'BatchNorm2d', 'MaxPool2d', 'ConvTranspose2d'}
    assert essential_types.issubset(layer_types), f"Missing essential layer types. Found: {layer_types}"


def test_unetr_hierarchy():
    """Test hierarchy building with MONAI UNETR model."""
    # Create UNETR model
    model = UNETR(
        in_channels=1,
        out_channels=2,
        img_size=(96, 96, 96),
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        pos_embed='conv',
        norm_name='instance',
        res_block=True
    )

    # Save model state dict to temporary file
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        torch.save(model.state_dict(), f.name)
        state_dict_path = f.name

    try:
        # Build hierarchy
        hierarchy, param_counts = build_model_hierarchy_from_state_dict(state_dict_path)
        
        # Generate filename with timestamp and model name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"model_hierarchy_UNETR_{timestamp}.txt"
        
        # Save hierarchy
        save_model_hierarchy(hierarchy, param_counts, output_file)
        
        # Verify key components are present
        assert any('vit' in key.lower() for key in hierarchy.keys()), "ViT components not found"
        assert any('encoder' in key.lower() for key in hierarchy.keys()), "Encoder not found"
        assert any('decoder' in key.lower() for key in hierarchy.keys()), "Decoder not found"
        
        # Verify parameter counts
        total_params = sum(p.numel() for p in model.parameters())
        hierarchy_params = sum(count for count in param_counts.values())
        assert hierarchy_params >= total_params  # >= because parameters are counted in parent nodes too
        
        # Print total parameters for inspection
        print(f"\nUNETR Model Statistics:")
        print(f"Total parameters: {total_params:,}")
        print(f"Hierarchy saved to: {output_file}")
        
    finally:
        # Cleanup temporary state dict file
        Path(state_dict_path).unlink() 


def test_manual_block_indexing():
    """Test manual block indexing workflow with UNETR model."""
    # Create UNETR model
    model = UNETR(
        in_channels=1,
        out_channels=2,
        img_size=(96, 96, 96),
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        pos_embed='conv',
        norm_name='instance',
        res_block=True
    )

    # Create temporary files
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f_model:
        torch.save(model.state_dict(), f_model.name)
        state_dict_path = f_model.name

    try:
        # First, generate hierarchy with empty brackets
        hierarchy, param_counts = build_model_hierarchy_from_state_dict(state_dict_path)
        hierarchy_file = "model_hierarchy_test.txt"
        save_model_hierarchy(hierarchy, param_counts, hierarchy_file)
        
        print("\nModel hierarchy has been saved to:", hierarchy_file)
        print("Please add block indices in the brackets and then confirm completion.")
        
        # Simulate user adding indices by modifying the file
        # In real usage, this would be done manually by the user
        with open(hierarchy_file, 'r') as f:
            lines = f.readlines()
        
        # Simulate user adding indices
        modified_lines = []
        for line in lines:
            if 'vit.blocks' in line:
                line = line.replace('[ ]', '[1]')  # Mark transformer blocks as block 1
            elif 'encoder' in line:
                line = line.replace('[ ]', '[2]')  # Mark encoders as block 2
            elif 'decoder' in line:
                line = line.replace('[ ]', '[3]')  # Mark decoders as block 3
            modified_lines.append(line)
        
        with open(hierarchy_file, 'w') as f:
            f.writelines(modified_lines)
        
        # Parse the user's block indexing
        blocks = parse_user_block_indexing(hierarchy_file)
        
        # Analyze block parameters
        block_info = analyze_block_parameters(blocks, param_counts)
        
        # Save block analysis
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_file = f"block_analysis_{timestamp}.txt"
        save_block_analysis(block_info, analysis_file)
        
        # Print results
        print("\nBlock analysis has been saved to:", analysis_file)
        with open(analysis_file, 'r') as f:
            print(f.read())
            
    finally:
        # Cleanup temporary files
        Path(state_dict_path).unlink()
        Path(hierarchy_file).unlink()
        Path(analysis_file).unlink() 


def test_block_plating():
    """Test block plating function with the existing hierarchy file."""
    try:
        # Use the existing hierarchy file
        hierarchy_file = "model_hierarchy_UNETR_20250220_221504.txt"
        
        # Parse the user's block indexing
        block_assignment, void_nodes, overlap_nodes, empty_box_nodes = parse_user_block_indexing(hierarchy_file)
        
        # Analyze block parameters
        analysis = analyze_block_parameters(hierarchy_file, block_assignment, void_nodes, overlap_nodes, empty_box_nodes)
        
        # Save block analysis - name based on input hierarchy file
        analysis_file = hierarchy_file.replace("model_hierarchy", "block_analysis")
        save_block_analysis(analysis, analysis_file)
        
        # Print analysis results
        print("\nBlock Analysis Results:")
        with open(analysis_file, 'r') as f:
            print(f.read())
        
        # Verify empty box nodes
        assert 'vit' in empty_box_nodes or 'vit.blocks' in empty_box_nodes, "Expected 'vit' or 'vit.blocks' to be empty boxes"
        
        # Verify that decoder and encoder nodes are empty boxes if their children are assigned
        for node in empty_box_nodes:
            if node.startswith(('decoder', 'encoder')):
                print(f"Empty box node found: {node}")
        
        # Verify blocks
        assert all(i in block_assignment for i in range(1, 8)), "Expected blocks 1 through 7 to be present"
        
        # Verify empty box info in analysis
        assert analysis['empty_box_info']['nodes'], "Expected empty box nodes in analysis"
        assert analysis['empty_box_info']['total_params'] > 0, "Expected non-zero parameters for empty box nodes"
        
    finally:
        # Don't delete the files since we're using existing ones
        pass 


def test_unet_block_plating():
    """Test block plating function with a UNet model."""
    try:
        # Create UNet model with a more complex architecture
        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.INSTANCE
        )

        # Calculate actual model parameters
        actual_params = sum(p.numel() for p in model.parameters())

        # Save model state dict to temporary file
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            torch.save(model.state_dict(), f.name)
            state_dict_path = f.name

        # Use the specified hierarchy file
        hierarchy_file = "model_hierarchy_UNet_20250221_153149.txt"
        print("\nAnalyzing block assignments in:", hierarchy_file)
        
        # Parse the user's block indexing
        block_assignment, void_nodes, overlap_nodes, empty_box_nodes = parse_user_block_indexing(hierarchy_file)
        
        # Analyze block parameters
        analysis = analyze_block_parameters(hierarchy_file, block_assignment, void_nodes, overlap_nodes, empty_box_nodes)
        
        # Save block analysis
        analysis_file = hierarchy_file.replace("model_hierarchy", "block_analysis")
        save_block_analysis(analysis, analysis_file)
        
        # Print analysis results
        print("\nBlock Analysis Results:")
        with open(analysis_file, 'r') as f:
            print(f.read())
        
        # Print parameter validation
        total_leaf_params = analysis['validation_info']['leaf_params']
        print("\nParameter Validation:")
        print(f"Total leaf node parameters across all blocks: {total_leaf_params:,}")
        print(f"Actual model parameters: {actual_params:,}")
        print(f"Difference: {abs(total_leaf_params - actual_params):,}")
        
        # Verify that total leaf parameters match actual parameters
        assert abs(total_leaf_params - actual_params) < 1000, "Significant mismatch between leaf parameters and actual parameters"
        
        # Print empty box nodes without parameters
        if empty_box_nodes:
            print("\nEmpty box nodes (no parameters counted):")
            for node in empty_box_nodes:
                print(f"  - {node}")
        
    finally:
        # Cleanup temporary state dict file
        Path(state_dict_path).unlink()
        # Don't delete the hierarchy file as we're using it 


def test_vit_block_plating():
    """Test block plating function with a MONAI ViT model."""
    try:
        # Create ViT model
        model = ViT(
            in_channels=1,
            img_size=(96, 96, 96),
            patch_size=(16, 16, 16),
            hidden_size=768,
            mlp_dim=3072,
            num_layers=12,
            num_heads=12,
            proj_type="conv",
            classification=True,
            num_classes=2
        )

        # Calculate actual model parameters
        actual_params = sum(p.numel() for p in model.parameters())

        # Save model state dict to temporary file
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            torch.save(model.state_dict(), f.name)
            state_dict_path = f.name

        # Use the specified hierarchy file
        hierarchy_file = "model_hierarchy_ViT_20250221_155726.txt"
        print("\nAnalyzing block assignments in:", hierarchy_file)
        
        # Parse block indexing
        block_assignment, void_nodes, overlap_nodes, empty_box_nodes = parse_user_block_indexing(hierarchy_file)

        # Analyze block parameters
        analysis = analyze_block_parameters(hierarchy_file, block_assignment, void_nodes, overlap_nodes, empty_box_nodes)

        # Save block analysis
        analysis_file = hierarchy_file.replace("model_hierarchy", "block_analysis")
        save_block_analysis(analysis, analysis_file)

        # Print analysis results
        print("\nBlock Analysis Results:")
        with open(analysis_file, 'r') as f:
            print(f.read())

        # Print parameter validation
        total_leaf_params = analysis['validation_info']['leaf_params']
        print("\nParameter Validation:")
        print(f"Total leaf node parameters across all blocks: {total_leaf_params:,}")
        print(f"Actual model parameters: {actual_params:,}")
        print(f"Difference: {abs(total_leaf_params - actual_params):,}")

        # Verify that total leaf parameters match actual parameters
        assert abs(total_leaf_params - actual_params) < 1000, "Significant mismatch between leaf parameters and actual parameters"

        # Print empty box nodes without parameters
        if empty_box_nodes:
            print("\nEmpty box nodes (no parameters counted):")
            for node in empty_box_nodes:
                print(f"  - {node}")

    finally:
        # Cleanup temporary state dict file
        if 'state_dict_path' in locals():
            Path(state_dict_path).unlink()
        # Don't delete the hierarchy file as we're using it


def create_vit_hierarchy():
    """Create a hierarchy file for a MONAI ViT model for manual block annotation."""
    try:
        # Create ViT model
        model = ViT(
            in_channels=1,
            img_size=(96, 96, 96),
            patch_size=(16, 16, 16),
            hidden_size=768,
            mlp_dim=3072,
            num_layers=12,
            num_heads=12,
            pos_embed='conv',
            classification=True,
            num_classes=2
        )

        # Save model state dict to temporary file
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            torch.save(model.state_dict(), f.name)
            state_dict_path = f.name

        # Build and save hierarchy
        hierarchy, param_counts = build_model_hierarchy_from_state_dict(state_dict_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hierarchy_file = f"model_hierarchy_ViT_{timestamp}.txt"
        save_model_hierarchy(hierarchy, param_counts, hierarchy_file)

        print(f"\nViT model hierarchy has been saved to: {hierarchy_file}")
        print("\nPlease annotate the hierarchy file by:")
        print("1. Adding block indices in the brackets [ ]")
        print("2. Use empty brackets [ ] for nodes that should inherit from their parent")
        print("3. Group components logically (e.g., patch embedding, attention layers, MLP layers, etc.)")
        print("\nTotal model parameters:", sum(p.numel() for p in model.parameters()))

    finally:
        # Cleanup temporary state dict file
        Path(state_dict_path).unlink() 


def create_resnet_hierarchy():
    """Create a hierarchy file for the ResNet-18 model for manual block annotation."""
    try:
        # Load the state dict
        state_dict_path = "experiment/MNIST/models/resnet18_state_dict.pt"
        
        # Build and save hierarchy
        hierarchy, param_counts = build_model_hierarchy_from_state_dict(state_dict_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hierarchy_file = f"model_hierarchy_ResNet18_{timestamp}.txt"
        save_model_hierarchy(hierarchy, param_counts, hierarchy_file)

        print(f"\nResNet-18 model hierarchy has been saved to: {hierarchy_file}")
        print("\nPlease annotate the hierarchy file by:")
        print("1. Adding block indices in the brackets [ ]")
        print("2. Use empty brackets [ ] for nodes that should inherit from their parent")
        print("3. Group components logically (e.g., conv layers, batch norms, etc.)")
        print("\nTotal model parameters:", sum(count for count in param_counts.values()))

    except Exception as e:
        print(f"Error creating hierarchy: {str(e)}") 
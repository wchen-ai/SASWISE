import pytest
import torch
import tempfile
from pathlib import Path
from monai.networks.nets import ViT

from src.models.block_warehouse import BlockWarehouse


@pytest.fixture
def sample_vit_model():
    """Create a sample ViT model for testing."""
    return ViT(
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


@pytest.fixture
def sample_block_analysis(tmp_path):
    """Create a sample block analysis file."""
    analysis_content = """Block Analysis:

Block 1<3>:
Total Parameters: 3,148,032
Nodes:
  - patch_embedding
  - patch_embedding.patch_embeddings
  - norm

Block 2<2>:
Total Parameters: 1,538
Nodes:
  - classification_head
  - classification_head.0

Block 3<4>:
Total Parameters: 21,256,704
Nodes:
  - blocks.0
  - blocks.0.attn
  - blocks.0.attn.out_proj
  - blocks.0.attn.qkv
  - blocks.0.mlp
  - blocks.0.mlp.linear1
  - blocks.0.mlp.linear2
  - blocks.0.norm1
  - blocks.0.norm2

Empty Box Nodes (all descendants are assigned or empty boxes):
  - blocks

Parameter Validation:
Total leaf node parameters across all blocks: 24,406,274
Total number of leaf nodes: 12
Total parameters across all blocks: 24,406,274
"""
    analysis_file = tmp_path / "block_analysis_test.txt"
    analysis_file.write_text(analysis_content)
    return str(analysis_file)


def test_block_warehouse_initialization(sample_vit_model, sample_block_analysis, tmp_path):
    """Test BlockWarehouse initialization and block loading."""
    variants_dir = tmp_path / "variants"
    block_config = BlockWarehouse(sample_vit_model, sample_block_analysis, str(variants_dir))
    
    # Check blocks were loaded correctly
    assert len(block_config.blocks) == 3
    assert block_config.blocks[1]['variants'] == 3
    assert block_config.blocks[2]['variants'] == 2
    assert block_config.blocks[3]['variants'] == 4
    
    # Check variant registry initialization
    assert len(block_config.variant_registry) == 3
    assert block_config.variant_registry[1]['total_variants'] == 3
    assert block_config.variant_registry[2]['total_variants'] == 2
    assert block_config.variant_registry[3]['total_variants'] == 4


def test_create_variant(sample_vit_model, sample_block_analysis, tmp_path):
    """Test creating individual variants."""
    variants_dir = tmp_path / "variants"
    block_config = BlockWarehouse(sample_vit_model, sample_block_analysis, str(variants_dir))
    
    # Create a variant
    block_config.create_variant(1, 1)
    
    # Check variant was created
    variant_path = variants_dir / "block_1_variant_1.pt"
    assert variant_path.exists()
    
    # Check variant registry was updated
    assert 1 in block_config.variant_registry[1]['available_variants']
    
    # Check variant content
    variant_state = torch.load(variant_path)
    assert isinstance(variant_state, dict)
    assert all(key.startswith(('patch_embedding', 'norm')) for key in variant_state.keys())


def test_create_all_variants(sample_vit_model, sample_block_analysis, tmp_path):
    """Test creating all possible variants."""
    variants_dir = tmp_path / "variants"
    block_config = BlockWarehouse(sample_vit_model, sample_block_analysis, str(variants_dir))
    
    # Create all variants
    block_config.create_all_variants()
    
    # Check all variants were created
    expected_variants = {
        1: list(range(1, 4)),  # 3 variants
        2: list(range(1, 3)),  # 2 variants
        3: list(range(1, 5))   # 4 variants
    }
    
    for block, variants in expected_variants.items():
        for variant in variants:
            variant_path = variants_dir / f"block_{block}_variant_{variant}.pt"
            assert variant_path.exists()


def test_build_submodel(sample_vit_model, sample_block_analysis, tmp_path):
    """Test building a submodel from variants."""
    variants_dir = tmp_path / "variants"
    block_config = BlockWarehouse(sample_vit_model, sample_block_analysis, str(variants_dir))
    
    # Create all variants
    block_config.create_all_variants()
    
    # Build a submodel
    submodel_block_config = {
        1: 2,  # Use variant 2 from block 1
        2: 1,  # Use variant 1 from block 2
        3: 3   # Use variant 3 from block 3
    }
    
    submodel_state = block_config.build_submodel(submodel_block_config)
    
    # Check submodel content
    assert isinstance(submodel_state, dict)
    assert any(key.startswith('patch_embedding') for key in submodel_state.keys())
    assert any(key.startswith('classification_head') for key in submodel_state.keys())
    assert any(key.startswith('blocks.0') for key in submodel_state.keys())


def test_save_submodel(sample_vit_model, sample_block_analysis, tmp_path):
    """Test saving a complete submodel."""
    variants_dir = tmp_path / "variants"
    block_config = BlockWarehouse(sample_vit_model, sample_block_analysis, str(variants_dir))
    
    # Create all variants
    block_config.create_all_variants()
    
    # Define and save a submodel
    submodel_block_config = {1: 1, 2: 1, 3: 1}
    submodel_path = tmp_path / "test_submodel.pt"
    block_config.save_submodel(submodel_block_config, str(submodel_path))
    
    # Check submodel was saved
    assert submodel_path.exists()
    
    # Load and verify submodel
    submodel_state = torch.load(submodel_path)
    assert isinstance(submodel_state, dict)
    assert len(submodel_state) > 0


def test_error_handling(sample_vit_model, sample_block_analysis, tmp_path):
    """Test error handling in BlockWarehouse."""
    variants_dir = tmp_path / "variants"
    block_config = BlockWarehouse(sample_vit_model, sample_block_analysis, str(variants_dir))
    
    # Test invalid block number
    with pytest.raises(ValueError, match="Block 99 not found"):
        block_config.create_variant(99, 1)
    
    # Test invalid variant number
    with pytest.raises(ValueError, match="Variant 5 exceeds maximum variants"):
        block_config.create_variant(1, 5)
    
    # Test building submodel with unavailable variant
    with pytest.raises(ValueError, match="Variant 1 not available"):
        block_config.build_submodel({1: 1})  # Haven't created variant 1 yet


def test_cleanup(sample_vit_model, sample_block_analysis, tmp_path):
    """Test cleanup functionality."""
    variants_dir = tmp_path / "variants"
    block_config = BlockWarehouse(sample_vit_model, sample_block_analysis, str(variants_dir))
    
    # Create some variants
    block_config.create_all_variants()
    
    # Verify variants exist
    assert variants_dir.exists()
    assert len(list(variants_dir.glob("*.pt"))) > 0
    
    # Clean up
    block_config.cleanup()
    
    # Verify cleanup
    assert not variants_dir.exists()
    assert len(block_config.variant_registry[1]['available_variants']) == 0


def test_save_division_info(sample_vit_model, sample_block_analysis, tmp_path):
    """Test saving division information to a file."""
    variants_dir = tmp_path / "variants"
    block_config = BlockWarehouse(sample_vit_model, sample_block_analysis, str(variants_dir))
    
    # Create some variants
    block_config.create_variant(1, 1)
    block_config.create_variant(2, 1)
    block_config.create_variant(3, 1)
    
    # Save division info
    division_file = tmp_path / "division_info.txt"
    block_config.save_division_info(str(division_file))
    
    # Verify file was created
    assert division_file.exists()
    
    # Check content
    content = division_file.read_text()
    assert "Block Division and Variant Information" in content
    assert "Total Model Parameters:" in content
    assert "Block 1:" in content
    assert "Block 2:" in content
    assert "Block 3:" in content
    assert "Available Variants: 1" in content
    assert "Example BlockConfig Format:" in content 
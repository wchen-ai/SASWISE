import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import yaml
from datetime import datetime
from assemble_model import assemble_model
from torchvision.models import resnet18
import numpy as np
from tqdm import tqdm
import random
import itertools
import warnings


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_device(config):
    if config['training']['device'] == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def load_test_data(config):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST(
        config['paths']['data_dir'],
        train=False,
        transform=transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['val_batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    return test_loader


def calculate_max_possible_block_configs(variant_info):
    """Calculate the maximum number of possible block_config combinations."""
    num_variants_per_block = []
    for block_id in range(1, len(variant_info['blocks']) + 1):
        block = variant_info['blocks'][str(block_id)]
        num_variants_per_block.append(block['num_variants'])
    
    # Total possible combinations is the product of number of variants per block
    return np.prod(num_variants_per_block)


def generate_all_possible_block_configs(variant_info):
    """Generate all possible block_config combinations."""
    variant_options = []
    for block_id in range(1, len(variant_info['blocks']) + 1):
        block = variant_info['blocks'][str(block_id)]
        variant_options.append(list(range(1, block['num_variants'] + 1)))
    
    # Generate all combinations
    return list(itertools.product(*variant_options))


def calculate_block_config_similarity(block_config1, block_config2):
    """Calculate similarity between two block_configs."""
    return sum(1 for a, b in zip(block_config1, block_config2) if a == b) / len(block_config1)


def generate_diverse_block_config(variant_info, previous_block_configs=None, max_attempts=100):
    """Generate a block_config that is sufficiently different from previous block_configs."""
    if previous_block_configs is None:
        previous_block_configs = []
    
    num_blocks = len(variant_info['blocks'])
    best_block_config = None
    lowest_similarity = float('inf')
    
    for _ in range(max_attempts):
        block_config = []
        for block_id in range(1, num_blocks + 1):
            block = variant_info['blocks'][str(block_id)]
            num_variants = block['num_variants']
            block_config.append(random.randint(1, num_variants))
        
        if not previous_block_configs:
            return block_config
        
        max_similarity = max(calculate_block_config_similarity(block_config, prev_block_config) 
                           for prev_block_config in previous_block_configs)
        
        if max_similarity < lowest_similarity:
            lowest_similarity = max_similarity
            best_block_config = block_config.copy()
        
        if lowest_similarity < 0.6:
            return best_block_config
    
    return best_block_config


def generate_block_configs(variant_info, config_spec):
    """Generate block_configs based on the specification.
    
    Args:
        variant_info: Dictionary containing variant information
        config_spec: Can be "full", a percentage like "10%", or a specific number
        
    Returns:
        List of block_configs to test
    """
    max_possible = calculate_max_possible_block_configs(variant_info)
    print(f"Maximum possible block_config combinations: {max_possible}")
    
    # Determine how many block_configs to generate
    if config_spec == "full":
        num_block_configs = max_possible
        print(f"Generating all {num_block_configs} possible block_config combinations")
    elif isinstance(config_spec, str) and config_spec.endswith("%"):
        percentage = float(config_spec.rstrip("%")) / 100
        num_block_configs = int(max_possible * percentage)
        print(f"Generating {num_block_configs} block_configs ({config_spec} of all possible combinations)")
    else:
        try:
            num_block_configs = int(config_spec)
            print(f"Generating {num_block_configs} block_configs")
        except ValueError:
            raise ValueError(f"Invalid block_config specification: {config_spec}. Use 'full', a percentage like '10%', or a number.")
    
    # Check if requested number exceeds maximum
    if num_block_configs > max_possible:
        warnings.warn(f"Requested {num_block_configs} block_configs, but only {max_possible} are possible. Using all possible block_configs.")
        num_block_configs = max_possible
    
    # Strategy depends on how many block_configs we need relative to the maximum
    if num_block_configs == max_possible:
        # Generate all possible combinations
        return generate_all_possible_block_configs(variant_info)
    elif num_block_configs > max_possible * 0.5:
        # If we need more than 50% of all possibilities, generate all and sample
        warnings.warn(f"Generating {num_block_configs} block_configs (>50% of all possible). This may take some time.")
        all_block_configs = generate_all_possible_block_configs(variant_info)
        random.shuffle(all_block_configs)
        return all_block_configs[:num_block_configs]
    else:
        # For a smaller number, generate diverse block_configs one by one
        block_configs = []
        for _ in tqdm(range(num_block_configs), desc="Generating diverse block_configs"):
            block_config = generate_diverse_block_config(variant_info, block_configs)
            block_configs.append(block_config)
        return block_configs


def create_model(config, device, block_config):
    """Create and load a model with the specified block_config configuration."""
    model = resnet18()
    
    # Create temporary path for assembled model
    output_path = f"experiment/MNIST/models/temp_test_model.pth"
    assemble_model(config['paths']['model_path'], block_config, output_path)
    state_dict = torch.load(output_path, map_location=device)
    
    # Modify first conv layer for grayscale input
    model.conv1 = nn.Conv2d(
        config['model']['in_channels'],
        64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False
    )
    
    if 'conv1.weight' in state_dict:
        rgb_weights = state_dict['conv1.weight']
        grayscale_weights = rgb_weights.mean(dim=1, keepdim=True)
        state_dict['conv1.weight'] = grayscale_weights
    
    model.fc = nn.Linear(512, config['model']['num_classes'])
    model.load_state_dict(state_dict, strict=False)
    
    # Clean up temporary file
    if os.path.exists(output_path):
        os.remove(output_path)
    
    return model.to(device)


def test_model(model, test_loader, device):
    """Test the model and return predictions and accuracy."""
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc='Testing'):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    return np.array(all_preds), np.array(all_labels), accuracy


def main(config_spec="10"):
    """Test multiple assembled submodels and save their predictions.
    
    Args:
        config_spec: Can be "full", a percentage like "10%", or a specific number
    """
    # Load configuration
    config = load_config('experiment/MNIST/train_config.yaml')
    device = get_device(config)
    print(f"Using device: {device}")
    
    # Load test data
    test_loader = load_test_data(config)
    print("Test data loaded successfully")
    
    # Load variant info
    variants_dir = Path(config['paths']['model_path']).parent.parent / "variants"
    with open(variants_dir / "variant_info.json", 'r') as f:
        variant_info = json.load(f)
    
    # Create results directory
    results_dir = Path(config['paths']['log_dir']) / "test_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate block_configs based on specification
    block_configs = generate_block_configs(variant_info, config_spec)
    
    # Test each block_config
    all_results = []
    
    for submodel_idx, block_config in enumerate(block_configs):
        print(f"\nTesting submodel {submodel_idx + 1}/{len(block_configs)}")
        print(f"BlockConfig: {block_config}")
        
        # Create and test model
        model = create_model(config, device, block_config)
        predictions, labels, accuracy = test_model(model, test_loader, device)
        print(f"Test Accuracy: {accuracy:.2f}%")
        
        # Save results
        result = {
            'submodel_id': submodel_idx + 1,
            'block_config': block_config,
            'accuracy': accuracy,
            'predictions': predictions.tolist(),
            'true_labels': labels.tolist(),
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        all_results.append(result)
        
        # Save individual submodel results
        submodel_file = results_dir / f"submodel_{submodel_idx + 1}_results.json"
        with open(submodel_file, 'w') as f:
            json.dump(result, f, indent=2)
    
    # Save summary of all submodels
    summary_file = results_dir / "test_summary.json"
    summary = {
        'num_submodels': len(block_configs),
        'config_spec': config_spec,
        'average_accuracy': np.mean([r['accuracy'] for r in all_results]),
        'submodels': all_results
    }
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nTesting completed. Results saved in {results_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Test multiple assembled submodels')
    parser.add_argument('--config_spec', type=str, default="10",
                      help='BlockConfig specification: "full", percentage like "10%", or a number (default: 10)')
    args = parser.parse_args()
    main(args.config_spec) 
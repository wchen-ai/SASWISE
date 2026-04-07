import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F
import yaml
from datetime import datetime
from collections import defaultdict
import random
import numpy as np
import argparse
import sys
import importlib

# Add the project root to the path so we can import modules properly
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models.assemble_model import assemble_model
from src.utils.training_logger import TrainingLogger


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_device(config):
    if config['training']['device'] == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def load_data(config):
    """Load dataset from disk based on config settings."""
    data_dir = config['paths']['data_dir']
    
    # Check if we should use MONAI for data loading
    use_monai = config['data'].get('use_monai', False)
    
    if use_monai:
        # Import MONAI dataset loader
        from src.data.monai_cifar10_dataset import load_monai_cifar10_dataset, get_monai_dataloaders
        
        # Load datasets with MONAI caching
        train_dataset, val_dataset = load_monai_cifar10_dataset(
            data_dir=data_dir,
            use_augmentation=config['data'].get('dataset_args', {}).get('use_augmentation', True),
            normalize=config['data'].get('dataset_args', {}).get('normalize', True),
            cache_rate=config['data'].get('cache_rate', 1.0),
            num_workers=config['data'].get('num_workers', 4)
        )
        
        # Create MONAI DataLoaders
        train_loader, val_loader = get_monai_dataloaders(
            train_ds=train_dataset,
            val_ds=val_dataset,
            train_batch_size=config['data']['train_batch_size'],
            val_batch_size=config['data']['val_batch_size'],
            num_workers=config['data']['num_workers'],
            pin_memory=config['data']['pin_memory']
        )
    else:
        # Use the original dataset loading method
        dataset_module = importlib.import_module(config['data']['dataset_module'])
        dataset_loader = getattr(dataset_module, config['data']['dataset_loader'])
        
        # Load train and validation datasets
        train_dataset, val_dataset = dataset_loader(data_dir, **config['data'].get('dataset_args', {}))
        
        # Create subsets if specified (optional)
        if 'train_subset_fraction' in config['data'] and config['data']['train_subset_fraction'] < 1.0:
            train_size = len(train_dataset)
            val_size = len(val_dataset)
            
            train_subset_size = int(train_size * config['data']['train_subset_fraction'])
            val_subset_size = int(val_size * config['data'].get('val_subset_fraction', 1.0))
            
            train_indices = torch.randperm(train_size)[:train_subset_size]
            val_indices = torch.randperm(val_size)[:val_subset_size]
            
            train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
            val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['data']['train_batch_size'],
            shuffle=True,
            num_workers=config['data']['num_workers'],
            pin_memory=config['data']['pin_memory']
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['data']['val_batch_size'],
            shuffle=False,
            num_workers=config['data']['num_workers'],
            pin_memory=config['data']['pin_memory']
        )
    
    print(f"Using {len(train_dataset)} training samples and {len(val_dataset)} validation samples")
    return train_loader, val_loader


def create_model(config, device, block_config, model_num):
    """Create a model with specified block_config configuration."""
    # Load model architecture using the method specified in config
    model_module = importlib.import_module(config['model']['model_module'])
    model_class = getattr(model_module, config['model']['model_class'])
    
    # Create model instance with default parameters
    model_args = config['model'].get('model_args', {}).copy()
    if 'num_classes' in model_args:
        num_classes = model_args.pop('num_classes')  # Save and remove num_classes
    else:
        num_classes = 10  # Default to 10 classes (e.g., for CIFAR-10)
    
    model = model_class(**model_args)
    
    # Load assembled model state dict
    model_path = config['paths']['model_path']
    output_path = f"{os.path.dirname(model_path)}/current_submodel_model_{model_num}.pth"
    
    # Use assemble_model to create a model with the specified block_config
    assemble_model(model_path, block_config, output_path)
    state_dict = torch.load(output_path, map_location=device)
    
    # Load the state dict with the original architecture
    model.load_state_dict(state_dict, strict=False)
    
    # Modify the final layer based on model type
    if hasattr(model, 'fc'):  # ResNet, DenseNet, etc.
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Linear):  # Some models
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):  # VGG, etc.
        if isinstance(model.classifier[-1], nn.Linear):
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)
    else:
        print(f"Warning: Could not automatically modify final layer for {model_class.__name__}")
    
    return model.to(device)


def train_epoch(model1, model2, train_loader, criterion, optimizer, device, alpha):
    model1.train()
    model2.eval()  # model2 is only used for consistency loss, no updates
    
    running_acc_loss = 0.0
    running_cons_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass through both models
        outputs1 = model1(inputs)
        with torch.no_grad():
            outputs2 = model2(inputs)
        
        # Calculate accuracy loss (cross entropy with ground truth)
        accuracy_loss = criterion(outputs1, targets)
        
        # Calculate consistency loss (MSE between model outputs)
        consistency_loss = F.mse_loss(F.softmax(outputs1, dim=1), 
                                    F.softmax(outputs2, dim=1))
        
        # Combined loss with alpha weight
        loss = accuracy_loss + alpha * consistency_loss
        
        # Backward pass (only for model1)
        loss.backward()
        optimizer.step()
        
        # Statistics - use batch size for proper scaling
        batch_size = targets.size(0)
        running_acc_loss += accuracy_loss.item() * batch_size
        running_cons_loss += consistency_loss.item() * batch_size
        _, predicted = outputs1.max(1)
        total += batch_size
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar with per-sample losses for better interpretation
        pbar.set_postfix({
            'acc_loss': running_acc_loss/total,
            'cons_loss': running_cons_loss/total,
            'acc': 100.*correct/total
        })
    
    # Return per-sample losses and accuracy
    return (running_acc_loss/total,  # Per-sample accuracy loss
            running_cons_loss/total, # Per-sample consistency loss
            100.*correct/total)      # Accuracy percentage


def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return running_loss/len(val_loader), 100.*correct/total


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
        
        # If no previous block_configs, accept first generated block_config
        if not previous_block_configs:
            return block_config
        
        # Calculate maximum similarity to any previous block_config
        max_similarity = max(calculate_block_config_similarity(block_config, prev_block_config) 
                           for prev_block_config in previous_block_configs)
        
        # Update best block_config if this one is more diverse
        if max_similarity < lowest_similarity:
            lowest_similarity = max_similarity
            best_block_config = block_config.copy()
        
        # Accept block_config if it's diverse enough
        if lowest_similarity < 0.6:
            return best_block_config
    
    # Return best found block_config if we couldn't find a perfectly diverse one
    return best_block_config


def update_variant_weights(model, block_config, variant_info, device):
    """Update weights for each variant used in the block_config."""
    state_dict = model.state_dict()
    
    # For each block in the block_config
    for block_id, variant_id in enumerate(block_config, start=1):
        block = variant_info['blocks'][str(block_id)]
        variant = block['variants'][variant_id - 1]
        
        # Get the nodes for this block
        block_nodes = block['nodes']
        
        # Create a state dict for this variant containing only its nodes
        variant_state_dict = {
            key: value for key, value in state_dict.items()
            if any(key.startswith(node) or key == node for node in block_nodes)
        }
        
        # Save the updated weights
        torch.save(variant_state_dict, variant['state_dict'])


def update_variant_history(variant_info, block_config, epochs, log_dir):
    """Update training history for each variant."""
    history_file = log_dir / "variant_training_history.json"
    
    # Load existing history or create new
    if history_file.exists():
        with open(history_file, 'r') as f:
            history = json.load(f)
    else:
        history = defaultdict(lambda: defaultdict(int))
        history = dict(history)  # Convert to regular dict for JSON serialization
    
    # Update epoch counts for each variant in the block_config
    for block_id, variant_id in enumerate(block_config, start=1):
        block_key = f"block_{block_id}"
        variant_key = f"variant_{variant_id}"
        
        if block_key not in history:
            history[block_key] = {}
        if variant_key not in history[block_key]:
            history[block_key][variant_key] = 0
            
        history[block_key][variant_key] += epochs
    
    # Save updated history
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train model with diversification')
    parser.add_argument('--config', type=str, required=True, 
                        help='Path to the configuration YAML file')
    args = parser.parse_args()
    
    # Load configuration from the provided path
    config = load_config(args.config)
    
    # Set up device
    device = get_device(config)
    print(f"Using device: {device}")
    
    # Create log directory if it doesn't exist
    log_dir = Path(config['paths']['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize logger
    logger = TrainingLogger(log_dir)
    
    # Load data
    train_loader, val_loader = load_data(config)
    print("Data loaded successfully")
    
    # Load variant info for block_config generation
    variants_dir = Path(config['paths']['model_path']).parent.parent / "variants"
    with open(variants_dir / "variant_info.json", 'r') as f:
        variant_info = json.load(f)
    
    # Training loop
    print("\nStarting training...")
    
    # Initialize block_config history
    block_config_history = []
    
    for round_idx in range(1, config['training']['num_rounds'] + 1):
        print(f"\nRound {round_idx}/{config['training']['num_rounds']}")
        
        # Generate diverse block_configs for this round
        block_config1 = generate_diverse_block_config(variant_info, block_config_history)
        block_config_history.append(block_config1)
        
        # Generate second block_config diverse from both block_config history and block_config1
        block_config2 = generate_diverse_block_config(variant_info, block_config_history)
        block_config_history.append(block_config2)
        
        # Keep block_config history manageable (keep last 5 rounds = 10 block_configs)
        if len(block_config_history) > 10:
            block_config_history = block_config_history[-10:]
        
        similarity = calculate_block_config_similarity(block_config1, block_config2)
        print(f"BlockConfig 1: {block_config1}")
        print(f"BlockConfig 2: {block_config2}")
        print(f"BlockConfig similarity: {similarity:.2f}")
        
        # Create models with different block_configs
        model1 = create_model(config, device, block_config1, 1)
        model2 = create_model(config, device, block_config2, 2)
        print("Models created and loaded successfully")
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            model1.parameters(),
            lr=config['training']['learning_rate'],
            momentum=config['training']['momentum'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Train for specified number of epochs
        for epoch in range(1, config['training']['epochs_per_round'] + 1):
            print(f"\nEpoch {epoch}/{config['training']['epochs_per_round']}")
            
            # Train
            train_acc_loss, train_cons_loss, train_acc = train_epoch(
                model1, model2, train_loader, criterion, optimizer, device,
                config['training']['alpha']
            )
            
            # Evaluate
            val_loss, val_acc = validate(model1, val_loader, criterion, device)
            
            # Log results
            print(f"Epoch {epoch} - Train Acc Loss: {train_acc_loss:.4f}, "
                  f"Cons Loss: {train_cons_loss:.4f}, Accuracy: {train_acc:.2f}%")
            print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
            
            # Log to our new logger
            logger.log_epoch(
                round_idx=round_idx,
                epoch=epoch,
                block_config1=block_config1,
                block_config2=block_config2,
                block_config_similarity=similarity,
                train_acc_loss=train_acc_loss,
                train_cons_loss=train_cons_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc
            )
        
        # Update variant weights and history
        update_variant_weights(model1, block_config1, variant_info, device)
        update_variant_history(variant_info, block_config1, config['training']['epochs_per_round'], log_dir)
        print(f"Updated weights and history for block_config 1 variants")
        
        # End round in logger (generates plots)
        logger.end_round(round_idx)
        print(f"Round {round_idx} completed. Metrics logged and plots generated.")
    
    print("\nTraining completed. Check the logs directory for detailed metrics and visualizations.")


if __name__ == "__main__":
    main() 
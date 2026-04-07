import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F
from torchvision.models import resnet18


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_device(config):
    if config['training']['device'] == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def load_data(config):
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet expects 224x224 images
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    # Load MNIST datasets
    train_dataset = datasets.MNIST(
        config['paths']['data_dir'], 
        train=True, 
        download=True,
        transform=transform
    )
    
    # Create a subset of training data (1%)
    train_size = len(train_dataset)
    subset_size = int(train_size * config['data']['train_subset_fraction'])
    indices = torch.randperm(train_size)[:subset_size]
    train_dataset = torch.utils.data.Subset(train_dataset, indices)
    
    val_dataset = datasets.MNIST(
        config['paths']['data_dir'],
        train=False,
        transform=transform
    )
    
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
    
    print(f"Using {subset_size} training samples ({config['data']['train_subset_fraction']*100}% of full dataset)")
    return train_loader, val_loader


def create_model(config, device):
    # Create ResNet18 model
    model = resnet18()
    
    # Load assembled model state dict
    state_dict = torch.load(config['paths']['model_path'], map_location=device)
    
    # Modify first conv layer for grayscale input
    model.conv1 = nn.Conv2d(
        config['model']['in_channels'],
        64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False
    )
    
    # Handle the channel mismatch in conv1.weight
    if 'conv1.weight' in state_dict:
        # Average the RGB channels to create grayscale weights
        rgb_weights = state_dict['conv1.weight']
        grayscale_weights = rgb_weights.mean(dim=1, keepdim=True)
        state_dict['conv1.weight'] = grayscale_weights
    
    # Modify final fc layer for MNIST classes
    model.fc = nn.Linear(512, config['model']['num_classes'])
    
    # Load the modified state dict
    model.load_state_dict(state_dict, strict=False)
    
    return model.to(device)


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        pbar.set_postfix({
            'loss': running_loss/total,
            'acc': 100.*correct/total
        })
    
    return running_loss/len(train_loader), 100.*correct/total


def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({
                'loss': running_loss/total,
                'acc': 100.*correct/total
            })
    
    return running_loss/len(val_loader), 100.*correct/total


def main():
    # Load configuration
    config = load_config('experiment/MNIST/train_config.yaml')
    
    # Set up device
    device = get_device(config)
    print(f"Using device: {device}")
    
    # Load data
    train_loader, val_loader = load_data(config)
    print("Data loaded successfully")
    
    # Create model
    model = create_model(config, device)
    print("Model created and loaded successfully")
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config['training']['learning_rate'],
        momentum=config['training']['momentum'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(config['training']['epochs']):
        print(f"\nEpoch {epoch+1}/{config['training']['epochs']}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        print(f"Training - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")


if __name__ == "__main__":
    main() 
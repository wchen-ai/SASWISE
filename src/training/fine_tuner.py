from typing import Dict, Any, Optional
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import Module
from tqdm import tqdm
import logging
from pathlib import Path

from ..utils.logger import TrainingLogger


class FineTuner:
    """Class for handling the fine-tuning process."""
    
    def __init__(
        self,
        model: Module,
        config: Dict[str, Any],
        warehouse: Dict[str, Any],
        logger: Optional[TrainingLogger] = None
    ):
        self.model = model
        self.config = config
        self.warehouse = warehouse
        self.logger = logger or TrainingLogger()
        
        self.device = next(model.parameters()).device
        self.optimizer = self._setup_optimizer()
        
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup the optimizer based on configuration."""
        optimizer_config = self.config.get('optimizer', {})
        return AdamW(
            self.model.parameters(),
            lr=optimizer_config.get('learning_rate', 1e-4),
            weight_decay=optimizer_config.get('weight_decay', 0.01)
        )
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: Optional[int] = None
    ):
        """
        Execute the fine-tuning training loop.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            epochs: Number of epochs to train for (defaults to config value)
        """
        epochs = epochs or self.config.get('epochs', 10)
        gradient_clip_val = self.config.get('gradient_clip_val', None)
        
        self.logger.start_training(epochs)
        
        for epoch in range(epochs):
            self.logger.start_epoch(epoch)
            
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch}')):
                loss = self._training_step(batch, batch_idx, gradient_clip_val)
                train_loss += loss
                
                if batch_idx % self.config.get('log_every_n_steps', 100) == 0:
                    self.logger.log_step(batch_idx, loss)
            
            avg_train_loss = train_loss / len(train_loader)
            self.logger.log_epoch_train(avg_train_loss)
            
            # Validation phase
            if val_loader is not None:
                val_loss = self._validate(val_loader)
                self.logger.log_epoch_validation(val_loss)
            
            # Save checkpoint
            if self.config.get('save_checkpoints', True):
                self._save_checkpoint(epoch)
            
            self.logger.end_epoch()
        
        self.logger.end_training()
        self._update_warehouse()
    
    def _training_step(
        self,
        batch: Any,
        batch_idx: int,
        gradient_clip_val: Optional[float]
    ) -> float:
        """Execute a single training step."""
        self.optimizer.zero_grad()
        
        # Forward pass
        loss = self.model(*batch) if isinstance(batch, tuple) else self.model(batch)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if gradient_clip_val is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                gradient_clip_val
            )
        
        # Optimizer step
        self.optimizer.step()
        
        return loss.item()
    
    def _validate(self, val_loader: DataLoader) -> float:
        """Run validation and return average validation loss."""
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                loss = self.model(*batch) if isinstance(batch, tuple) else self.model(batch)
                val_loss += loss.item()
        
        return val_loss / len(val_loader)
    
    def _save_checkpoint(self, epoch: int):
        """Save a training checkpoint."""
        checkpoint_dir = Path(self.config.get('save_dir', 'checkpoints'))
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, checkpoint_path)
        
        self.logger.log_info(f'Saved checkpoint to {checkpoint_path}')
    
    def _update_warehouse(self):
        """Update the warehouse with the fine-tuned weights."""
        self.warehouse['model'].load_state_dict(self.model.state_dict())
        self.logger.log_info('Updated warehouse with fine-tuned weights') 
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

class TrainingLogger:
    """
    Logger for training metrics with visualization capabilities.
    """
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.json_dir = self.log_dir / "json"
        self.plot_dir = self.log_dir / "plots"
        self.json_dir.mkdir(exist_ok=True)
        self.plot_dir.mkdir(exist_ok=True)
        
        # Initialize log data
        self.current_round_data = []
        self.all_rounds_data = []
        
        # Load existing data if available
        self.summary_file = self.log_dir / "training_summary.json"
        if self.summary_file.exists():
            with open(self.summary_file, 'r') as f:
                self.all_rounds_data = json.load(f)
    
    def log_epoch(self, round_idx, epoch, block_config1, block_config2, block_config_similarity, 
                 train_acc_loss, train_cons_loss, train_acc, val_loss=None, val_acc=None):
        """Log metrics for a single epoch."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create log entry
        log_entry = {
            'timestamp': timestamp,
            'round': round_idx,
            'epoch': epoch,
            'block_config1': block_config1,
            'block_config2': block_config2,
            'block_config_similarity': block_config_similarity,
            'train_accuracy_loss': train_acc_loss,
            'train_consistency_loss': train_cons_loss,
            'train_accuracy': train_acc
        }
        
        # Add validation metrics if available
        if val_loss is not None:
            log_entry['val_loss'] = val_loss
        if val_acc is not None:
            log_entry['val_accuracy'] = val_acc
        
        # Save to current round data
        self.current_round_data.append(log_entry)
        
        # Save individual epoch log
        epoch_file = self.json_dir / f"round_{round_idx}_epoch_{epoch}.json"
        with open(epoch_file, 'w') as f:
            json.dump(log_entry, f, indent=2)
        
        return log_entry
    
    def end_round(self, round_idx):
        """Finalize logging for a round and generate plots."""
        # Save round summary
        round_file = self.json_dir / f"round_{round_idx}_summary.json"
        with open(round_file, 'w') as f:
            json.dump(self.current_round_data, f, indent=2)
        
        # Add to all rounds data
        self.all_rounds_data.extend(self.current_round_data)
        
        # Save overall summary
        with open(self.summary_file, 'w') as f:
            json.dump(self.all_rounds_data, f, indent=2)
        
        # Generate plots
        self._plot_round_metrics(round_idx)
        self._plot_all_rounds_metrics()
        
        # Reset current round data
        self.current_round_data = []
    
    def _plot_round_metrics(self, round_idx):
        """Generate plots for a single round."""
        if not self.current_round_data:
            return
        
        # Extract data
        epochs = [entry['epoch'] for entry in self.current_round_data]
        train_acc_loss = [entry['train_accuracy_loss'] for entry in self.current_round_data]
        train_cons_loss = [entry['train_consistency_loss'] for entry in self.current_round_data]
        train_acc = [entry['train_accuracy'] for entry in self.current_round_data]
        
        # Check if validation data exists
        has_val = 'val_accuracy' in self.current_round_data[0]
        if has_val:
            val_loss = [entry.get('val_loss', 0) for entry in self.current_round_data]
            val_acc = [entry.get('val_accuracy', 0) for entry in self.current_round_data]
        
        # Create figure with subplots
        fig, axs = plt.subplots(3 if has_val else 2, 1, figsize=(10, 12 if has_val else 8), sharex=True)
        
        # Plot training losses
        axs[0].plot(epochs, train_acc_loss, 'b-', label='Accuracy Loss')
        axs[0].plot(epochs, train_cons_loss, 'r-', label='Consistency Loss')
        axs[0].set_ylabel('Loss')
        axs[0].set_title(f'Round {round_idx} Training Losses')
        axs[0].legend()
        axs[0].grid(True)
        
        # Plot training accuracy
        axs[1].plot(epochs, train_acc, 'g-', label='Training Accuracy')
        if has_val:
            axs[1].plot(epochs, val_acc, 'm-', label='Validation Accuracy')
        axs[1].set_ylabel('Accuracy (%)')
        axs[1].set_title(f'Round {round_idx} Accuracy')
        axs[1].legend()
        axs[1].grid(True)
        
        # Plot validation loss if available
        if has_val:
            axs[2].plot(epochs, val_loss, 'm-', label='Validation Loss')
            axs[2].set_ylabel('Loss')
            axs[2].set_title(f'Round {round_idx} Validation Loss')
            axs[2].legend()
            axs[2].grid(True)
            axs[2].set_xlabel('Epoch')
        else:
            axs[1].set_xlabel('Epoch')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(self.plot_dir / f"round_{round_idx}_metrics.png")
        plt.close()
    
    def _plot_all_rounds_metrics(self):
        """Generate plots for all rounds."""
        if not self.all_rounds_data:
            return
        
        # Group data by round
        rounds_data = {}
        for entry in self.all_rounds_data:
            round_idx = entry['round']
            if round_idx not in rounds_data:
                rounds_data[round_idx] = []
            rounds_data[round_idx].append(entry)
        
        # Extract data for plotting
        rounds = sorted(rounds_data.keys())
        
        # Calculate average metrics per round
        avg_train_acc = []
        avg_train_acc_loss = []
        avg_train_cons_loss = []
        avg_val_acc = []
        
        for r in rounds:
            round_entries = rounds_data[r]
            avg_train_acc.append(np.mean([entry['train_accuracy'] for entry in round_entries]))
            avg_train_acc_loss.append(np.mean([entry['train_accuracy_loss'] for entry in round_entries]))
            avg_train_cons_loss.append(np.mean([entry['train_consistency_loss'] for entry in round_entries]))
            
            if 'val_accuracy' in round_entries[0]:
                avg_val_acc.append(np.mean([entry.get('val_accuracy', 0) for entry in round_entries]))
        
        # Create figure with subplots
        fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Plot average losses
        axs[0].plot(rounds, avg_train_acc_loss, 'b-', label='Avg Accuracy Loss')
        axs[0].plot(rounds, avg_train_cons_loss, 'r-', label='Avg Consistency Loss')
        axs[0].set_ylabel('Loss')
        axs[0].set_title('Average Losses per Round')
        axs[0].legend()
        axs[0].grid(True)
        
        # Plot average accuracy
        axs[1].plot(rounds, avg_train_acc, 'g-', label='Avg Training Accuracy')
        if avg_val_acc:
            axs[1].plot(rounds, avg_val_acc, 'm-', label='Avg Validation Accuracy')
        axs[1].set_ylabel('Accuracy (%)')
        axs[1].set_title('Average Accuracy per Round')
        axs[1].set_xlabel('Round')
        axs[1].legend()
        axs[1].grid(True)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(self.plot_dir / "all_rounds_metrics.png")
        plt.close()
        
        # Also create a plot showing the progression of final accuracy per round
        final_train_acc = []
        final_val_acc = []
        
        for r in rounds:
            round_entries = sorted(rounds_data[r], key=lambda x: x['epoch'])
            final_entry = round_entries[-1]  # Last epoch in the round
            final_train_acc.append(final_entry['train_accuracy'])
            if 'val_accuracy' in final_entry:
                final_val_acc.append(final_entry['val_accuracy'])
        
        plt.figure(figsize=(10, 6))
        plt.plot(rounds, final_train_acc, 'g-', label='Final Training Accuracy')
        if final_val_acc:
            plt.plot(rounds, final_val_acc, 'm-', label='Final Validation Accuracy')
        plt.ylabel('Accuracy (%)')
        plt.xlabel('Round')
        plt.title('Final Accuracy per Round')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.plot_dir / "final_accuracy_per_round.png")
        plt.close() 
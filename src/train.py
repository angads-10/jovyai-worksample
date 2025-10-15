"""
Training Script for Decision Transformer

This script implements the training loop for offline RL Decision Transformer
on both MIMIC-III hospital data and D4RL simulation data.

Author: Jovy AI Research Team
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import argparse
import yaml
from pathlib import Path
import logging
from tqdm import tqdm
import wandb
from typing import Dict, Optional

from model import DecisionTransformer, DecisionTransformerConfig, create_model
from dataset import create_dataloaders, TrajectoryDataModule

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DecisionTransformerTrainer:
    """Trainer class for Decision Transformer."""
    
    def __init__(self,
                 config: DecisionTransformerConfig,
                 train_loader,
                 val_loader=None,
                 test_loader=None,
                 device: str = 'cuda',
                 use_wandb: bool = False):
        """
        Initialize trainer.
        
        Args:
            config: Model and training configuration
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            test_loader: Test data loader (optional)
            device: Device to use for training
            use_wandb: Whether to use Weights & Biases logging
        """
        self.config = config
        self.device = device
        self.use_wandb = use_wandb
        
        # Create model
        self.model = create_model(config).to(device)
        
        # Create optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=1e-5
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs
        )
        
        # Data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir='runs/decision_transformer')
        
        # Initialize Weights & Biases
        if use_wandb:
            wandb.init(
                project="decision-transformer-jovy",
                config=config.to_dict(),
                name=f"dt_{config.d_model}d_{config.n_layers}l"
            )
        
        logger.info(f"Trainer initialized with {self._count_parameters():,} parameters")
    
    def _count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            states = batch['states'].to(self.device)
            actions = batch['actions'].to(self.device)
            rtgs = batch['rtgs'].unsqueeze(-1).to(self.device)  # Add dimension for RTG
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Predict actions
            predicted_actions = self.model(states, actions, rtgs)
            
            # Compute loss (only on valid timesteps)
            valid_mask = ~attention_mask
            loss = self.model.compute_loss(states, actions, rtgs, actions, valid_mask)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Log to TensorBoard
            if batch_idx % 100 == 0:
                self.writer.add_scalar('Loss/Train_Batch', loss.item(), 
                                     len(self.train_losses) * len(self.train_loader) + batch_idx)
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self) -> float:
        """Validate model."""
        if self.val_loader is None:
            return 0.0
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                states = batch['states'].to(self.device)
                actions = batch['actions'].to(self.device)
                rtgs = batch['rtgs'].unsqueeze(-1).to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Forward pass
                predicted_actions = self.model(states, actions, rtgs)
                
                # Compute loss
                valid_mask = ~attention_mask
                loss = self.model.compute_loss(states, actions, rtgs, actions, valid_mask)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        self.val_losses.append(avg_loss)
        
        return avg_loss
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_dir = Path('checkpoints')
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config.to_dict()
        }
        
        # Save regular checkpoint
        checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model at epoch {epoch}")
        
        logger.info(f"Saved checkpoint at epoch {epoch}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint['epoch']
    
    def train(self, num_epochs: Optional[int] = None):
        """Train the model."""
        if num_epochs is None:
            num_epochs = self.config.num_epochs
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Update learning rate
            self.scheduler.step()
            
            # Log metrics
            self.writer.add_scalar('Loss/Train_Epoch', train_loss, epoch)
            self.writer.add_scalar('Loss/Val_Epoch', val_loss, epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            if epoch % 10 == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # Print progress
            logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                       f"Train Loss: {train_loss:.4f}, "
                       f"Val Loss: {val_loss:.4f}")
        
        logger.info("Training completed!")
        
        # Save final model
        self.save_checkpoint(num_epochs - 1)
        
        # Close writers
        self.writer.close()
        if self.use_wandb:
            wandb.finish()

def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description='Train Decision Transformer')
    parser.add_argument('--config', type=str, default='configs/dt_base.yaml',
                       help='Path to config file')
    parser.add_argument('--dataset', type=str, choices=['mimic', 'd4rl'], default='d4rl',
                       help='Dataset type')
    parser.add_argument('--env', type=str, default='hopper-medium-v2',
                       help='D4RL environment name')
    parser.add_argument('--data_dir', type=str, default='data/',
                       help='Data directory')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases logging')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Load configuration
    if Path(args.config).exists():
        config = DecisionTransformerConfig.from_yaml(args.config)
    else:
        config = DecisionTransformerConfig()
    
    # Override config with command line arguments
    if args.dataset == 'd4rl':
        config.state_dim = 11  # Hopper state dimension
        config.action_dim = 3  # Hopper action dimension
    
    logger.info(f"Using configuration: {config.to_dict()}")
    
    # Create data loaders
    try:
        if args.dataset == 'mimic':
            train_loader, test_loader = create_dataloaders(
                dataset_type='mimic',
                data_dir=args.data_dir,
                batch_size=config.batch_size,
                max_length=config.max_length
            )
            val_loader = test_loader  # Use test as validation for now
        else:  # d4rl
            train_loader, test_loader = create_dataloaders(
                dataset_type='d4rl',
                data_dir=args.data_dir,
                batch_size=config.batch_size,
                max_length=config.max_length
            )
            val_loader = test_loader
        
        logger.info(f"Created data loaders: train={len(train_loader.dataset)}, "
                   f"test={len(test_loader.dataset) if test_loader else 0}")
        
    except Exception as e:
        logger.error(f"Failed to create data loaders: {e}")
        return
    
    # Create trainer
    trainer = DecisionTransformerTrainer(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=args.device,
        use_wandb=args.use_wandb
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train(num_epochs=config.num_epochs)

if __name__ == "__main__":
    main()

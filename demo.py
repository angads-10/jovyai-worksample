#!/usr/bin/env python3
"""
Demo Script for Decision Transformer Repository

This script demonstrates the key functionality of the Decision Transformer
implementation for offline RL on hospital trajectories.

Usage:
    python demo.py [--dataset mimic|d4rl] [--quick]

Author: Jovy AI Research Team
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demo_model_creation():
    """Demo model creation and basic functionality."""
    logger.info("=== Model Creation Demo ===")
    
    from src.model import DecisionTransformer, DecisionTransformerConfig
    
    # Create configuration
    config = DecisionTransformerConfig(
        state_dim=17,
        action_dim=9,
        d_model=64,  # Smaller for demo
        n_layers=2,
        max_length=20
    )
    
    # Create model
    model = DecisionTransformer(
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        d_model=config.d_model,
        n_heads=4,
        n_layers=config.n_layers,
        max_length=config.max_length
    )
    
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test forward pass
    batch_size, seq_len = 2, 10
    states = torch.randn(batch_size, config.max_length, config.state_dim)
    actions = torch.randn(batch_size, config.max_length, config.action_dim)
    rtgs = torch.randn(batch_size, config.max_length, 1)
    
    with torch.no_grad():
        predicted_actions = model(states, actions, rtgs)
    
    logger.info(f"Forward pass successful: {predicted_actions.shape}")
    
    # Test loss computation
    target_actions = torch.randn_like(actions)
    loss = model.compute_loss(states, actions, rtgs, target_actions)
    logger.info(f"Loss computation successful: {loss.item():.4f}")
    
    return model, config

def demo_data_loading(dataset_type='mimic'):
    """Demo data loading functionality."""
    logger.info(f"=== Data Loading Demo ({dataset_type}) ===")
    
    try:
        if dataset_type == 'mimic':
            from src.dataset import TrajectoryDataset
            
            data_path = Path('data/hospital_trajectories.csv')
            if data_path.exists():
                dataset = TrajectoryDataset(str(data_path), max_length=50)
                logger.info(f"MIMIC dataset loaded: {len(dataset)} trajectories")
                
                # Test data loading
                if len(dataset) > 0:
                    sample = dataset[0]
                    logger.info(f"Sample trajectory shapes:")
                    logger.info(f"  States: {sample['states'].shape}")
                    logger.info(f"  Actions: {sample['actions'].shape}")
                    logger.info(f"  RTGs: {sample['rtgs'].shape}")
                    logger.info(f"  Length: {sample['length']}")
                
                return dataset
            else:
                logger.warning("MIMIC data not found, creating synthetic data")
                return None
                
        elif dataset_type == 'd4rl':
            from data.load_d4rl import D4RLLoader
            
            # Check if D4RL is available
            try:
                train_loader, test_loader = D4RLLoader.get_dataloader(
                    env_name='hopper-medium-v2',
                    batch_size=4,
                    max_length=50
                )
                
                logger.info("D4RL dataset loaded successfully")
                
                # Test a batch
                batch = next(iter(train_loader))
                logger.info(f"Batch shapes:")
                logger.info(f"  States: {batch['states'].shape}")
                logger.info(f"  Actions: {batch['actions'].shape}")
                logger.info(f"  RTGs: {batch['rtgs'].shape}")
                
                return train_loader, test_loader
                
            except ImportError:
                logger.warning("D4RL not installed, skipping D4RL demo")
                return None
                
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        return None

def demo_training_step(model, config, dataset=None):
    """Demo a single training step."""
    logger.info("=== Training Step Demo ===")
    
    if dataset is None:
        logger.warning("No dataset available, using synthetic data")
        # Create synthetic batch
        batch_size = 2
        states = torch.randn(batch_size, config.max_length, config.state_dim)
        actions = torch.randn(batch_size, config.max_length, config.action_dim)
        rtgs = torch.randn(batch_size, config.max_length, 1)
        attention_mask = torch.zeros(batch_size, config.max_length, dtype=torch.bool)
    else:
        # Use real data
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(dataloader))
        
        states = batch['states']
        actions = batch['actions']
        rtgs = batch['rtgs'].unsqueeze(-1)
        attention_mask = batch['attention_mask']
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # Training step
    model.train()
    optimizer.zero_grad()
    
    predicted_actions = model(states, actions, rtgs)
    valid_mask = ~attention_mask
    loss = model.compute_loss(states, actions, rtgs, actions, valid_mask)
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    logger.info(f"Training step completed, loss: {loss.item():.4f}")

def demo_evaluation(model, config, dataset=None):
    """Demo evaluation functionality."""
    logger.info("=== Evaluation Demo ===")
    
    if dataset is None:
        logger.warning("No dataset available, using synthetic evaluation")
        
        # Create synthetic test data
        batch_size = 4
        states = torch.randn(batch_size, config.max_length, config.state_dim)
        actions = torch.randn(batch_size, config.max_length, config.action_dim)
        rtgs = torch.randn(batch_size, config.max_length, 1)
        
        # Simple evaluation metrics
        model.eval()
        with torch.no_grad():
            predicted_actions = model(states, actions, rtgs)
            mse = torch.mean((predicted_actions - actions) ** 2)
            mae = torch.mean(torch.abs(predicted_actions - actions))
        
        logger.info(f"Evaluation metrics:")
        logger.info(f"  MSE: {mse.item():.4f}")
        logger.info(f"  MAE: {mae.item():.4f}")
        
    else:
        # Use real evaluation
        try:
            from src.eval import DecisionTransformerEvaluator
            from torch.utils.data import DataLoader
            
            test_loader = DataLoader(dataset, batch_size=4, shuffle=False)
            evaluator = DecisionTransformerEvaluator(model, test_loader)
            
            # Run basic evaluation
            results = evaluator.evaluate_all()
            
            logger.info("Evaluation results:")
            logger.info(f"  BC Accuracy: {results['behavior_cloning']['accuracy']:.4f}")
            logger.info(f"  RTG Correlation: {results['rtg_correlation']['correlation']:.4f}")
            logger.info(f"  Safety Violation Rate: {results['safety_violations']['violation_rate']:.4f}")
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")

def demo_api_conversion(dataset=None):
    """Demo API fine-tuning data conversion."""
    logger.info("=== API Conversion Demo ===")
    
    try:
        from src.api_finetune import APIFineTuneConverter
        
        converter = APIFineTuneConverter(token_format='text')
        
        if dataset is not None and len(dataset) > 0:
            # Convert first trajectory to SFT format
            sample = dataset[0]
            trajectory = {
                'states': sample['states'].numpy(),
                'actions': sample['actions'].numpy(),
                'rtgs': sample['rtgs'].numpy(),
                'rewards': torch.zeros_like(sample['rtgs']).numpy(),
                'trajectory_id': 0
            }
            
            sft_example = converter.trajectory_to_sft_format(trajectory)
            
            logger.info("SFT conversion successful:")
            logger.info(f"  Messages: {len(sft_example['messages'])}")
            logger.info(f"  System message length: {len(sft_example['messages'][0]['content'])}")
            logger.info(f"  User message length: {len(sft_example['messages'][1]['content'])}")
            
        else:
            logger.warning("No dataset available for API conversion demo")
            
    except Exception as e:
        logger.error(f"API conversion failed: {e}")

def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description='Demo Decision Transformer functionality')
    parser.add_argument('--dataset', type=str, choices=['mimic', 'd4rl'], default='mimic',
                       help='Dataset type to demo')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick demo with minimal functionality')
    
    args = parser.parse_args()
    
    logger.info("Starting Decision Transformer Demo")
    logger.info("=" * 50)
    
    try:
        # Demo 1: Model creation
        model, config = demo_model_creation()
        
        # Demo 2: Data loading
        dataset = demo_data_loading(args.dataset)
        
        # Demo 3: Training step
        demo_training_step(model, config, dataset)
        
        # Demo 4: Evaluation
        demo_evaluation(model, config, dataset)
        
        # Demo 5: API conversion (if not quick mode)
        if not args.quick:
            demo_api_conversion(dataset)
        
        logger.info("=" * 50)
        logger.info("Demo completed successfully!")
        logger.info("\nNext steps:")
        logger.info("1. Install dependencies: pip install -r requirements.txt")
        logger.info("2. Train model: python src/train.py --dataset d4rl")
        logger.info("3. Evaluate model: python src/eval.py --model_path checkpoints/best_model.pt")
        logger.info("4. Explore notebook: jupyter notebook notebooks/visualize_rollouts.ipynb")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        logger.info("This is expected if dependencies are not installed.")
        logger.info("Please install requirements.txt and try again.")

if __name__ == "__main__":
    main()

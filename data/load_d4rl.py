"""
D4RL Dataset Loading for Offline RL Decision Transformer

This script loads D4RL benchmark datasets for reproducible offline RL experiments.
Provides a standardized interface for training Decision Transformers on simulation data.

Author: Jovy AI Research Team
"""

import d4rl
import gym
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class D4RLTrajectoryDataset(Dataset):
    """
    PyTorch Dataset for D4RL trajectories.
    
    Converts D4RL data into the format expected by Decision Transformer:
    - States: Environment observations
    - Actions: Agent actions
    - Rewards: Environment rewards
    - RTG: Return-to-Go (cumulative future reward)
    """
    
    def __init__(self, 
                 env_name: str,
                 max_length: int = 100,
                 normalize_rewards: bool = True):
        """
        Initialize D4RL dataset.
        
        Args:
            env_name: D4RL environment name (e.g., 'hopper-medium-v2')
            max_length: Maximum trajectory length for training
            normalize_rewards: Whether to normalize rewards
        """
        self.env_name = env_name
        self.max_length = max_length
        self.normalize_rewards = normalize_rewards
        
        # Load D4RL dataset
        logger.info(f"Loading D4RL dataset: {env_name}")
        self.env = gym.make(env_name)
        self.dataset = self.env.get_dataset()
        
        # Extract trajectories
        self.trajectories = self._extract_trajectories()
        
        # Normalize states and actions
        self.state_mean, self.state_std = self._compute_normalization_stats()
        
        logger.info(f"Loaded {len(self.trajectories)} trajectories")
        logger.info(f"State dim: {self.dataset['observations'].shape[1]}")
        logger.info(f"Action dim: {self.dataset['actions'].shape[1]}")
    
    def _extract_trajectories(self) -> List[Dict]:
        """Extract individual trajectories from D4RL dataset."""
        observations = self.dataset['observations']
        actions = self.dataset['actions']
        rewards = self.dataset['rewards']
        terminals = self.dataset['terminals']
        timeouts = self.dataset.get('timeouts', np.zeros_like(terminals))
        
        trajectories = []
        current_traj = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'rtgs': []
        }
        
        for i in range(len(observations)):
            current_traj['observations'].append(observations[i])
            current_traj['actions'].append(actions[i])
            current_traj['rewards'].append(rewards[i])
            
            # Check if trajectory ends
            if terminals[i] or timeouts[i] or len(current_traj['observations']) >= self.max_length:
                if len(current_traj['observations']) > 5:  # Minimum trajectory length
                    # Calculate RTGs
                    current_traj['rtgs'] = self._calculate_rtgs(current_traj['rewards'])
                    trajectories.append(current_traj)
                
                # Start new trajectory
                current_traj = {
                    'observations': [],
                    'actions': [],
                    'rewards': [],
                    'rtgs': []
                }
        
        return trajectories
    
    def _calculate_rtgs(self, rewards: List[float]) -> List[float]:
        """Calculate Return-to-Go sequence."""
        rtgs = []
        rtg = 0.0
        
        # Calculate RTG from end to beginning
        for reward in reversed(rewards):
            rtg = reward + 0.99 * rtg  # Discount factor 0.99
            rtgs.append(rtg)
        
        # Reverse to get correct order
        return list(reversed(rtgs))
    
    def _compute_normalization_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute normalization statistics for states and actions."""
        all_states = np.concatenate([traj['observations'] for traj in self.trajectories])
        all_actions = np.concatenate([traj['actions'] for traj in self.trajectories])
        
        state_mean = np.mean(all_states, axis=0)
        state_std = np.std(all_states, axis=0) + 1e-8  # Avoid division by zero
        
        action_mean = np.mean(all_actions, axis=0)
        action_std = np.std(all_actions, axis=0) + 1e-8
        
        return (state_mean, state_std), (action_mean, action_std)
    
    def __len__(self) -> int:
        return len(self.trajectories)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single trajectory."""
        traj = self.trajectories[idx]
        
        # Normalize states and actions
        states = (np.array(traj['observations']) - self.state_mean[0]) / self.state_mean[1]
        actions = (np.array(traj['actions']) - self.state_std[0]) / self.state_std[1]
        
        # Pad or truncate to max_length
        traj_length = len(states)
        if traj_length > self.max_length:
            # Randomly sample a window of max_length
            start_idx = np.random.randint(0, traj_length - self.max_length + 1)
            states = states[start_idx:start_idx + self.max_length]
            actions = actions[start_idx:start_idx + self.max_length]
            rewards = np.array(traj['rewards'])[start_idx:start_idx + self.max_length]
            rtgs = np.array(traj['rtgs'])[start_idx:start_idx + self.max_length]
        else:
            # Pad with zeros
            pad_length = self.max_length - traj_length
            states = np.pad(states, ((0, pad_length), (0, 0)), mode='constant')
            actions = np.pad(actions, ((0, pad_length), (0, 0)), mode='constant')
            rewards = np.pad(traj['rewards'], (0, pad_length), mode='constant')
            rtgs = np.pad(traj['rtgs'], (0, pad_length), mode='constant')
        
        return {
            'states': torch.FloatTensor(states),
            'actions': torch.FloatTensor(actions),
            'rewards': torch.FloatTensor(rewards),
            'rtgs': torch.FloatTensor(rtgs),
            'length': traj_length
        }

class D4RLLoader:
    """Utility class for loading D4RL datasets."""
    
    @staticmethod
    def get_dataloader(env_name: str,
                      batch_size: int = 32,
                      max_length: int = 100,
                      shuffle: bool = True,
                      num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
        """
        Create train and test dataloaders for D4RL environment.
        
        Args:
            env_name: D4RL environment name
            batch_size: Batch size for training
            max_length: Maximum trajectory length
            shuffle: Whether to shuffle training data
            num_workers: Number of data loading workers
            
        Returns:
            Tuple of (train_loader, test_loader)
        """
        # Create dataset
        dataset = D4RLTrajectoryDataset(env_name, max_length=max_length)
        
        # Split into train/test (80/20)
        total_size = len(dataset)
        train_size = int(0.8 * total_size)
        
        train_dataset = torch.utils.data.Subset(dataset, range(train_size))
        test_dataset = torch.utils.data.Subset(dataset, range(train_size, total_size))
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        logger.info(f"Created dataloaders:")
        logger.info(f"  Train: {len(train_dataset)} trajectories")
        logger.info(f"  Test: {len(test_dataset)} trajectories")
        logger.info(f"  Batch size: {batch_size}")
        
        return train_loader, test_loader
    
    @staticmethod
    def get_available_environments() -> List[str]:
        """Get list of available D4RL environments."""
        return [
            # Hopper
            'hopper-random-v2',
            'hopper-medium-v2', 
            'hopper-expert-v2',
            'hopper-medium-replay-v2',
            
            # Walker2d
            'walker2d-random-v2',
            'walker2d-medium-v2',
            'walker2d-expert-v2',
            'walker2d-medium-replay-v2',
            
            # HalfCheetah
            'halfcheetah-random-v2',
            'halfcheetah-medium-v2',
            'halfcheetah-expert-v2',
            'halfcheetah-medium-replay-v2',
            
            # Ant
            'ant-random-v2',
            'ant-medium-v2',
            'ant-expert-v2',
            'ant-medium-replay-v2',
        ]

def main():
    """Demo script for D4RL data loading."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Load D4RL dataset for Decision Transformer')
    parser.add_argument('--env', type=str, default='hopper-medium-v2',
                       help='D4RL environment name')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for dataloader')
    parser.add_argument('--max_length', type=int, default=100,
                       help='Maximum trajectory length')
    
    args = parser.parse_args()
    
    # Test data loading
    try:
        train_loader, test_loader = D4RLLoader.get_dataloader(
            env_name=args.env,
            batch_size=args.batch_size,
            max_length=args.max_length
        )
        
        # Test a batch
        batch = next(iter(train_loader))
        logger.info(f"Batch shapes:")
        logger.info(f"  States: {batch['states'].shape}")
        logger.info(f"  Actions: {batch['actions'].shape}")
        logger.info(f"  RTGs: {batch['rtgs'].shape}")
        
        logger.info("D4RL data loading successful!")
        
    except Exception as e:
        logger.error(f"Error loading D4RL data: {e}")
        logger.info("Available environments:")
        for env in D4RLLoader.get_available_environments():
            logger.info(f"  {env}")

if __name__ == "__main__":
    main()

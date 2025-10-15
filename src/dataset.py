"""
Dataset Classes for Offline RL Decision Transformer

This module provides dataset classes for loading and preprocessing
trajectory data for Decision Transformer training.

Author: Jovy AI Research Team
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class TrajectoryDataset(Dataset):
    """
    PyTorch Dataset for offline RL trajectories.
    
    Handles both MIMIC-III hospital data and D4RL simulation data
    in a unified format for Decision Transformer training.
    """
    
    def __init__(self,
                 data_path: str,
                 max_length: int = 100,
                 normalize: bool = True,
                 dataset_type: str = 'mimic'):
        """
        Initialize trajectory dataset.
        
        Args:
            data_path: Path to trajectory data (CSV or JSON)
            max_length: Maximum trajectory length for training
            normalize: Whether to normalize states and actions
            dataset_type: Type of dataset ('mimic' or 'd4rl')
        """
        self.data_path = Path(data_path)
        self.max_length = max_length
        self.normalize = normalize
        self.dataset_type = dataset_type
        
        # Load data
        self.trajectories = self._load_trajectories()
        
        # Compute normalization statistics if needed
        if self.normalize:
            self._compute_normalization_stats()
        
        logger.info(f"Loaded {len(self.trajectories)} trajectories from {data_path}")
    
    def _load_trajectories(self) -> List[Dict]:
        """Load trajectories from file."""
        if self.data_path.suffix == '.csv':
            return self._load_from_csv()
        elif self.data_path.suffix == '.json':
            return self._load_from_json()
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
    
    def _load_from_csv(self) -> List[Dict]:
        """Load trajectories from CSV file."""
        df = pd.read_csv(self.data_path)
        
        # Group by trajectory identifier
        if 'icustay_id' in df.columns:
            group_col = 'icustay_id'
        elif 'trajectory_id' in df.columns:
            group_col = 'trajectory_id'
        else:
            # Assume each row is a separate trajectory
            group_col = None
        
        trajectories = []
        
        if group_col is not None:
            # Group by trajectory
            for traj_id, group in df.groupby(group_col):
                trajectory = self._create_trajectory_from_group(group)
                if trajectory and len(trajectory['states']) > 5:  # Minimum length
                    trajectories.append(trajectory)
        else:
            # Each row is a trajectory
            for _, row in df.iterrows():
                trajectory = self._create_trajectory_from_row(row)
                if trajectory and len(trajectory['states']) > 5:
                    trajectories.append(trajectory)
        
        return trajectories
    
    def _load_from_json(self) -> List[Dict]:
        """Load trajectories from JSON file."""
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and 'trajectories' in data:
            return data['trajectories']
        else:
            raise ValueError("Invalid JSON format")
    
    def _create_trajectory_from_group(self, group: pd.DataFrame) -> Optional[Dict]:
        """Create trajectory from grouped DataFrame."""
        try:
            # Sort by timestep
            group = group.sort_values('timestep')
            
            # Parse state and action vectors
            states = []
            actions = []
            rewards = []
            rtgs = []
            
            for _, row in group.iterrows():
                # Parse state vector
                if isinstance(row['state'], str):
                    state = json.loads(row['state'].replace("'", '"'))
                else:
                    state = row['state']
                states.append(np.array(state, dtype=np.float32))
                
                # Parse action vector
                if isinstance(row['action'], str):
                    action = json.loads(row['action'].replace("'", '"'))
                else:
                    action = row['action']
                actions.append(np.array(action, dtype=np.float32))
                
                rewards.append(float(row['reward']))
                rtgs.append(float(row['rtg']))
            
            return {
                'states': np.array(states),
                'actions': np.array(actions),
                'rewards': np.array(rewards),
                'rtgs': np.array(rtgs),
                'length': len(states),
                'trajectory_id': group.iloc[0]['icustay_id'] if 'icustay_id' in group.columns else len(self.trajectories)
            }
        except Exception as e:
            logger.warning(f"Failed to create trajectory from group: {e}")
            return None
    
    def _create_trajectory_from_row(self, row: pd.Series) -> Optional[Dict]:
        """Create trajectory from single row."""
        try:
            # This assumes the row contains a full trajectory
            # Implementation depends on specific data format
            return {
                'states': np.array(row['states']),
                'actions': np.array(row['actions']),
                'rewards': np.array(row['rewards']),
                'rtgs': np.array(row['rtgs']),
                'length': len(row['states']),
                'trajectory_id': row.get('trajectory_id', 0)
            }
        except Exception as e:
            logger.warning(f"Failed to create trajectory from row: {e}")
            return None
    
    def _compute_normalization_stats(self):
        """Compute normalization statistics."""
        all_states = np.concatenate([traj['states'] for traj in self.trajectories])
        all_actions = np.concatenate([traj['actions'] for traj in self.trajectories])
        
        self.state_mean = np.mean(all_states, axis=0)
        self.state_std = np.std(all_states, axis=0) + 1e-8
        
        self.action_mean = np.mean(all_actions, axis=0)
        self.action_std = np.std(all_actions, axis=0) + 1e-8
        
        logger.info(f"Computed normalization stats:")
        logger.info(f"  State dim: {len(self.state_mean)}")
        logger.info(f"  Action dim: {len(self.action_mean)}")
    
    def _normalize_trajectory(self, trajectory: Dict) -> Dict:
        """Normalize trajectory states and actions."""
        if not self.normalize:
            return trajectory
        
        normalized_traj = trajectory.copy()
        
        # Normalize states
        normalized_traj['states'] = (trajectory['states'] - self.state_mean) / self.state_std
        
        # Normalize actions
        normalized_traj['actions'] = (trajectory['actions'] - self.action_mean) / self.action_std
        
        return normalized_traj
    
    def __len__(self) -> int:
        return len(self.trajectories)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single trajectory."""
        trajectory = self.trajectories[idx]
        trajectory = self._normalize_trajectory(trajectory)
        
        # Get trajectory data
        states = trajectory['states']
        actions = trajectory['actions']
        rewards = trajectory['rewards']
        rtgs = trajectory['rtgs']
        length = trajectory['length']
        
        # Pad or truncate to max_length
        if length > self.max_length:
            # Randomly sample a window of max_length
            start_idx = np.random.randint(0, length - self.max_length + 1)
            states = states[start_idx:start_idx + self.max_length]
            actions = actions[start_idx:start_idx + self.max_length]
            rewards = rewards[start_idx:start_idx + self.max_length]
            rtgs = rtgs[start_idx:start_idx + self.max_length]
            actual_length = self.max_length
        else:
            # Pad with zeros
            pad_length = self.max_length - length
            states = np.pad(states, ((0, pad_length), (0, 0)), mode='constant')
            actions = np.pad(actions, ((0, pad_length), (0, 0)), mode='constant')
            rewards = np.pad(rewards, (0, pad_length), mode='constant')
            rtgs = np.pad(rtgs, (0, pad_length), mode='constant')
            actual_length = length
        
        # Create attention mask for valid timesteps
        attention_mask = np.zeros(self.max_length, dtype=bool)
        attention_mask[actual_length:] = True  # True means ignore (mask out)
        
        return {
            'states': torch.FloatTensor(states),
            'actions': torch.FloatTensor(actions),
            'rewards': torch.FloatTensor(rewards),
            'rtgs': torch.FloatTensor(rtgs),
            'length': actual_length,
            'attention_mask': torch.BoolTensor(attention_mask),
            'trajectory_id': trajectory['trajectory_id']
        }

class TrajectoryDataModule:
    """Data module for managing trajectory datasets."""
    
    def __init__(self,
                 train_path: str,
                 test_path: Optional[str] = None,
                 val_path: Optional[str] = None,
                 batch_size: int = 32,
                 max_length: int = 100,
                 normalize: bool = True,
                 num_workers: int = 4):
        """
        Initialize data module.
        
        Args:
            train_path: Path to training data
            test_path: Path to test data (optional)
            val_path: Path to validation data (optional)
            batch_size: Batch size for dataloaders
            max_length: Maximum trajectory length
            normalize: Whether to normalize data
            num_workers: Number of data loading workers
        """
        self.train_path = train_path
        self.test_path = test_path
        self.val_path = val_path
        self.batch_size = batch_size
        self.max_length = max_length
        self.normalize = normalize
        self.num_workers = num_workers
        
        # Create datasets
        self.train_dataset = TrajectoryDataset(
            train_path, max_length=max_length, normalize=normalize
        )
        
        if test_path:
            self.test_dataset = TrajectoryDataset(
                test_path, max_length=max_length, normalize=normalize
            )
        else:
            self.test_dataset = None
        
        if val_path:
            self.val_dataset = TrajectoryDataset(
                val_path, max_length=max_length, normalize=normalize
            )
        else:
            self.val_dataset = None
    
    def get_train_dataloader(self, shuffle: bool = True) -> DataLoader:
        """Get training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
    
    def get_test_dataloader(self) -> Optional[DataLoader]:
        """Get test dataloader."""
        if self.test_dataset is None:
            return None
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_val_dataloader(self) -> Optional[DataLoader]:
        """Get validation dataloader."""
        if self.val_dataset is None:
            return None
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_data_info(self) -> Dict:
        """Get dataset information."""
        info = {
            'train_size': len(self.train_dataset),
            'test_size': len(self.test_dataset) if self.test_dataset else 0,
            'val_size': len(self.val_dataset) if self.val_dataset else 0,
            'max_length': self.max_length,
            'batch_size': self.batch_size
        }
        
        # Get state and action dimensions from first sample
        if len(self.train_dataset) > 0:
            sample = self.train_dataset[0]
            info['state_dim'] = sample['states'].shape[1]
            info['action_dim'] = sample['actions'].shape[1]
        
        return info

def create_dataloaders(dataset_type: str = 'mimic',
                      data_dir: str = 'data/',
                      batch_size: int = 32,
                      max_length: int = 100,
                      train_split: float = 0.8) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create train and test dataloaders.
    
    Args:
        dataset_type: Type of dataset ('mimic' or 'd4rl')
        data_dir: Directory containing data files
        batch_size: Batch size for training
        max_length: Maximum trajectory length
        train_split: Fraction of data to use for training
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    if dataset_type == 'mimic':
        # For MIMIC data, assume single CSV file
        data_path = Path(data_dir) / 'hospital_trajectories.csv'
        
        if not data_path.exists():
            raise FileNotFoundError(f"MIMIC data not found at {data_path}")
        
        # Create dataset and split
        full_dataset = TrajectoryDataset(data_path, max_length=max_length)
        
        # Split into train/test
        total_size = len(full_dataset)
        train_size = int(train_split * total_size)
        
        train_dataset = torch.utils.data.Subset(full_dataset, range(train_size))
        test_dataset = torch.utils.data.Subset(full_dataset, range(train_size, total_size))
        
    elif dataset_type == 'd4rl':
        # For D4RL, use the D4RL loader
        from .load_d4rl import D4RLLoader
        train_loader, test_loader = D4RLLoader.get_dataloader(
            env_name='hopper-medium-v2',  # Default environment
            batch_size=batch_size,
            max_length=max_length
        )
        return train_loader, test_loader
    
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, test_loader

if __name__ == "__main__":
    # Demo script
    import argparse
    
    parser = argparse.ArgumentParser(description='Test trajectory dataset')
    parser.add_argument('--data_path', type=str, default='data/hospital_trajectories.csv')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_length', type=int, default=100)
    
    args = parser.parse_args()
    
    try:
        # Test dataset loading
        dataset = TrajectoryDataset(args.data_path, max_length=args.max_length)
        
        # Test dataloader
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        
        # Test a batch
        batch = next(iter(dataloader))
        
        print(f"Dataset loaded successfully!")
        print(f"Number of trajectories: {len(dataset)}")
        print(f"Batch shapes:")
        print(f"  States: {batch['states'].shape}")
        print(f"  Actions: {batch['actions'].shape}")
        print(f"  RTGs: {batch['rtgs'].shape}")
        print(f"  Lengths: {batch['length']}")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")

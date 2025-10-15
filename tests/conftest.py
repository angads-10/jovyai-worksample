"""
Pytest configuration and shared fixtures for Decision Transformer tests.

This module provides common fixtures and configuration for all test modules
in the test suite.
"""

import pytest
import torch
import numpy as np
import tempfile
import os
from pathlib import Path

from src.model import DecisionTransformerConfig, DecisionTransformer
from src.dataset import TrajectoryDataset


@pytest.fixture
def test_config():
    """Create a test configuration for consistent testing."""
    return DecisionTransformerConfig(
        state_dim=10,
        action_dim=4,
        d_model=64,
        n_heads=4,
        n_layers=2,
        max_length=20,
        dropout=0.1,
        learning_rate=1e-4,
        batch_size=4,
        num_epochs=5
    )


@pytest.fixture
def test_model(test_config):
    """Create a test model instance."""
    return DecisionTransformer(
        state_dim=test_config.state_dim,
        action_dim=test_config.action_dim,
        d_model=test_config.d_model,
        n_heads=test_config.n_heads,
        n_layers=test_config.n_layers,
        max_length=test_config.max_length,
        dropout=test_config.dropout
    )


@pytest.fixture
def sample_trajectory_data():
    """Create sample trajectory data for testing."""
    return {
        'states': np.random.randn(50, 10).astype(np.float32),
        'actions': np.random.randn(50, 4).astype(np.float32),
        'rewards': np.random.randn(50).astype(np.float32),
        'rtgs': np.random.randn(50).astype(np.float32),
        'length': 50,
        'trajectory_id': 1
    }


@pytest.fixture
def sample_batch(test_config):
    """Create sample batch data for testing."""
    batch_size = 2
    return {
        'states': torch.randn(batch_size, test_config.max_length, test_config.state_dim),
        'actions': torch.randn(batch_size, test_config.max_length, test_config.action_dim),
        'rtgs': torch.randn(batch_size, test_config.max_length, 1),
        'rewards': torch.randn(batch_size, test_config.max_length),
        'length': torch.tensor([15, 20]),
        'attention_mask': torch.zeros(batch_size, test_config.max_length, dtype=torch.bool),
        'trajectory_id': torch.tensor([1, 2])
    }


@pytest.fixture
def temp_csv_file():
    """Create a temporary CSV file with trajectory data."""
    import pandas as pd
    
    # Create sample trajectory data
    data = []
    for traj_id in range(3):
        for timestep in range(20):
            data.append({
                'trajectory_id': traj_id,
                'timestep': timestep,
                'state': [np.random.randn(5) for _ in range(3)],  # 3 state features
                'action': [np.random.randn(2) for _ in range(2)],  # 2 action features
                'reward': np.random.randn(),
                'rtg': np.random.randn(),
                'length': 20
            })
    
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    df = pd.DataFrame(data)
    df.to_csv(temp_file.name, index=False)
    temp_file.close()
    
    yield temp_file.name
    
    # Cleanup
    os.unlink(temp_file.name)


@pytest.fixture
def temp_directory():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_d4rl_dataset():
    """Create a mock D4RL dataset for testing."""
    class MockD4RLDataset:
        def __init__(self):
            self.observations = np.random.randn(1000, 11)
            self.actions = np.random.randn(1000, 3)
            self.rewards = np.random.randn(1000)
            self.terminals = np.zeros(1000)
            self.terminals[::100] = 1  # Terminate every 100 steps
            self.timeouts = np.zeros_like(self.terminals)
        
        def get_dataset(self):
            return {
                'observations': self.observations,
                'actions': self.actions,
                'rewards': self.rewards,
                'terminals': self.terminals,
                'timeouts': self.timeouts
            }
    
    return MockD4RLDataset()


@pytest.fixture(scope="session")
def device():
    """Get the appropriate device for testing."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def random_seed():
    """Set random seed for reproducible tests."""
    torch.manual_seed(42)
    np.random.seed(42)
    yield 42


@pytest.fixture(autouse=True)
def setup_test_environment(random_seed):
    """Setup test environment for each test."""
    # Set random seeds for reproducibility
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # Set environment variables for testing
    os.environ['PYTHONPATH'] = str(Path(__file__).parent.parent)
    
    yield
    
    # Cleanup after test
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Mark slow tests
        if "slow" in item.name or "integration" in item.name:
            item.add_marker(pytest.mark.slow)
        
        # Mark GPU tests
        if "gpu" in item.name or "cuda" in item.name:
            item.add_marker(pytest.mark.gpu)
        
        # Mark integration tests
        if "integration" in item.name:
            item.add_marker(pytest.mark.integration)


# Test utilities
class TestUtils:
    """Utility functions for testing."""
    
    @staticmethod
    def assert_tensor_equal(tensor1, tensor2, tolerance=1e-6):
        """Assert two tensors are equal within tolerance."""
        assert torch.allclose(tensor1, tensor2, atol=tolerance)
    
    @staticmethod
    def assert_tensor_shape(tensor, expected_shape):
        """Assert tensor has expected shape."""
        assert tensor.shape == expected_shape
    
    @staticmethod
    def assert_model_parameters_finite(model):
        """Assert all model parameters are finite."""
        for param in model.parameters():
            assert torch.isfinite(param).all()
    
    @staticmethod
    def create_mock_trajectory(length, state_dim, action_dim):
        """Create a mock trajectory for testing."""
        return {
            'states': np.random.randn(length, state_dim).astype(np.float32),
            'actions': np.random.randn(length, action_dim).astype(np.float32),
            'rewards': np.random.randn(length).astype(np.float32),
            'rtgs': np.random.randn(length).astype(np.float32),
            'length': length,
            'trajectory_id': np.random.randint(1000)
        }


@pytest.fixture
def test_utils():
    """Provide test utilities."""
    return TestUtils

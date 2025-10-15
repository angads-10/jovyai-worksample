"""
Tests for Decision Transformer model implementation.

This module contains comprehensive tests for the DecisionTransformer class,
including forward pass, loss computation, and edge cases.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock

from src.model import (
    DecisionTransformer,
    DecisionTransformerConfig,
    create_model,
    count_parameters
)


class TestDecisionTransformerConfig:
    """Test DecisionTransformerConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = DecisionTransformerConfig()
        
        assert config.state_dim == 17
        assert config.action_dim == 9
        assert config.d_model == 128
        assert config.n_heads == 8
        assert config.n_layers == 3
        assert config.max_length == 100
        assert config.dropout == 0.1
        assert config.learning_rate == 1e-4
        assert config.batch_size == 32
        assert config.num_epochs == 100
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = DecisionTransformerConfig(
            state_dim=20,
            action_dim=5,
            d_model=256,
            n_heads=16,
            n_layers=6,
            max_length=200,
            dropout=0.2,
            learning_rate=5e-5,
            batch_size=16,
            num_epochs=200
        )
        
        assert config.state_dim == 20
        assert config.action_dim == 5
        assert config.d_model == 256
        assert config.n_heads == 16
        assert config.n_layers == 6
        assert config.max_length == 200
        assert config.dropout == 0.2
        assert config.learning_rate == 5e-5
        assert config.batch_size == 16
        assert config.num_epochs == 200
    
    def test_config_to_dict(self):
        """Test configuration to dictionary conversion."""
        config = DecisionTransformerConfig()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict['state_dim'] == config.state_dim
        assert config_dict['action_dim'] == config.action_dim
        assert config_dict['d_model'] == config.d_model
    
    def test_config_from_yaml(self, tmp_path):
        """Test loading configuration from YAML file."""
        import yaml
        
        # Create temporary YAML file
        config_data = {
            'state_dim': 25,
            'action_dim': 7,
            'd_model': 512,
            'n_heads': 8,
            'n_layers': 4,
            'max_length': 150,
            'dropout': 0.15,
            'learning_rate': 2e-4,
            'batch_size': 64,
            'num_epochs': 150
        }
        
        yaml_file = tmp_path / "test_config.yaml"
        with open(yaml_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Load configuration
        config = DecisionTransformerConfig.from_yaml(str(yaml_file))
        
        assert config.state_dim == 25
        assert config.action_dim == 7
        assert config.d_model == 512
        assert config.n_heads == 8
        assert config.n_layers == 4
        assert config.max_length == 150
        assert config.dropout == 0.15
        assert config.learning_rate == 2e-4
        assert config.batch_size == 64
        assert config.num_epochs == 150


class TestDecisionTransformer:
    """Test DecisionTransformer model class."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return DecisionTransformerConfig(
            state_dim=10,
            action_dim=4,
            d_model=64,
            n_heads=4,
            n_layers=2,
            max_length=20,
            dropout=0.1
        )
    
    @pytest.fixture
    def model(self, config):
        """Create test model."""
        return DecisionTransformer(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            max_length=config.max_length,
            dropout=config.dropout
        )
    
    @pytest.fixture
    def sample_batch(self, config):
        """Create sample batch data."""
        batch_size = 2
        seq_len = 10
        
        return {
            'states': torch.randn(batch_size, config.max_length, config.state_dim),
            'actions': torch.randn(batch_size, config.max_length, config.action_dim),
            'rtgs': torch.randn(batch_size, config.max_length, 1)
        }
    
    def test_model_creation(self, config):
        """Test model can be created with valid configuration."""
        model = DecisionTransformer(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            max_length=config.max_length,
            dropout=config.dropout
        )
        
        assert isinstance(model, DecisionTransformer)
        assert model.state_dim == config.state_dim
        assert model.action_dim == config.action_dim
        assert model.d_model == config.d_model
        assert model.max_length == config.max_length
    
    def test_forward_pass(self, model, sample_batch):
        """Test forward pass produces correct output shape."""
        states = sample_batch['states']
        actions = sample_batch['actions']
        rtgs = sample_batch['rtgs']
        
        output = model(states, actions, rtgs)
        
        expected_shape = (states.shape[0], model.max_length, model.action_dim)
        assert output.shape == expected_shape
        assert isinstance(output, torch.Tensor)
    
    def test_forward_pass_with_attention_mask(self, model, sample_batch):
        """Test forward pass with custom attention mask."""
        states = sample_batch['states']
        actions = sample_batch['actions']
        rtgs = sample_batch['rtgs']
        
        # Create attention mask
        attention_mask = torch.zeros(model.max_length, model.max_length, dtype=torch.bool)
        
        output = model(states, actions, rtgs, attention_mask)
        
        expected_shape = (states.shape[0], model.max_length, model.action_dim)
        assert output.shape == expected_shape
    
    def test_compute_loss(self, model, sample_batch):
        """Test loss computation works correctly."""
        states = sample_batch['states']
        actions = sample_batch['actions']
        rtgs = sample_batch['rtgs']
        target_actions = torch.randn_like(actions)
        
        loss = model.compute_loss(states, actions, rtgs, target_actions)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
    
    def test_compute_loss_with_mask(self, model, sample_batch):
        """Test loss computation with attention mask."""
        states = sample_batch['states']
        actions = sample_batch['actions']
        rtgs = sample_batch['rtgs']
        target_actions = torch.randn_like(actions)
        
        # Create mask (only first 5 timesteps are valid)
        mask = torch.zeros(states.shape[0], model.max_length, dtype=torch.bool)
        mask[:, 5:] = True  # True means ignore
        
        loss = model.compute_loss(states, actions, rtgs, target_actions, mask)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
    
    def test_get_action(self, model, sample_batch):
        """Test get_action method for inference."""
        states = sample_batch['states']
        actions = sample_batch['actions']
        rtgs = sample_batch['rtgs']
        timestep = 5
        
        model.eval()
        with torch.no_grad():
            predicted_action = model.get_action(states, actions, rtgs, timestep)
        
        expected_shape = (states.shape[0], model.action_dim)
        assert predicted_action.shape == expected_shape
        assert isinstance(predicted_action, torch.Tensor)
    
    def test_different_batch_sizes(self, config):
        """Test model works with different batch sizes."""
        model = DecisionTransformer(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            max_length=config.max_length,
            dropout=config.dropout
        )
        
        for batch_size in [1, 4, 8]:
            states = torch.randn(batch_size, config.max_length, config.state_dim)
            actions = torch.randn(batch_size, config.max_length, config.action_dim)
            rtgs = torch.randn(batch_size, config.max_length, 1)
            
            output = model(states, actions, rtgs)
            assert output.shape[0] == batch_size
    
    def test_different_sequence_lengths(self, config):
        """Test model handles different sequence lengths."""
        model = DecisionTransformer(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            max_length=config.max_length,
            dropout=config.dropout
        )
        
        batch_size = 2
        states = torch.randn(batch_size, config.max_length, config.state_dim)
        actions = torch.randn(batch_size, config.max_length, config.action_dim)
        rtgs = torch.randn(batch_size, config.max_length, 1)
        
        # Test with different sequence lengths
        for seq_len in [5, 10, 15, config.max_length]:
            states_subset = states[:, :seq_len, :]
            actions_subset = actions[:, :seq_len, :]
            rtgs_subset = rtgs[:, :seq_len, :]
            
            output = model(states_subset, actions_subset, rtgs_subset)
            assert output.shape[0] == batch_size
            assert output.shape[1] == config.max_length  # Always pad to max_length
    
    def test_gradient_flow(self, model, sample_batch):
        """Test that gradients flow properly during training."""
        states = sample_batch['states']
        actions = sample_batch['actions']
        rtgs = sample_batch['rtgs']
        target_actions = torch.randn_like(actions)
        
        # Enable gradient computation
        states.requires_grad_(True)
        actions.requires_grad_(True)
        rtgs.requires_grad_(True)
        
        loss = model.compute_loss(states, actions, rtgs, target_actions)
        loss.backward()
        
        # Check that gradients exist
        assert states.grad is not None
        assert actions.grad is not None
        assert rtgs.grad is not None
    
    def test_model_parameters(self, model):
        """Test model parameter initialization."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert total_params > 0
        assert trainable_params > 0
        assert trainable_params == total_params  # All parameters should be trainable
    
    def test_model_device_compatibility(self, model, sample_batch):
        """Test model works on different devices."""
        # Test on CPU
        model_cpu = model.cpu()
        states_cpu = sample_batch['states'].cpu()
        actions_cpu = sample_batch['actions'].cpu()
        rtgs_cpu = sample_batch['rtgs'].cpu()
        
        output_cpu = model_cpu(states_cpu, actions_cpu, rtgs_cpu)
        assert output_cpu.device.type == 'cpu'
        
        # Test on CUDA if available
        if torch.cuda.is_available():
            model_cuda = model.cuda()
            states_cuda = sample_batch['states'].cuda()
            actions_cuda = sample_batch['actions'].cuda()
            rtgs_cuda = sample_batch['rtgs'].cuda()
            
            output_cuda = model_cuda(states_cuda, actions_cuda, rtgs_cuda)
            assert output_cuda.device.type == 'cuda'


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_create_model(self):
        """Test create_model utility function."""
        config = DecisionTransformerConfig()
        model = create_model(config)
        
        assert isinstance(model, DecisionTransformer)
        assert model.state_dim == config.state_dim
        assert model.action_dim == config.action_dim
    
    def test_count_parameters(self):
        """Test count_parameters utility function."""
        config = DecisionTransformerConfig()
        model = create_model(config)
        
        param_count = count_parameters(model)
        assert isinstance(param_count, int)
        assert param_count > 0
        
        # Verify count matches manual calculation
        manual_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert param_count == manual_count


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_invalid_config(self):
        """Test model creation with invalid configuration."""
        with pytest.raises((ValueError, RuntimeError)):
            DecisionTransformer(
                state_dim=-1,  # Invalid dimension
                action_dim=4,
                d_model=64,
                n_heads=4,
                n_layers=2,
                max_length=20,
                dropout=0.1
            )
    
    def test_empty_batch(self):
        """Test model with empty batch."""
        config = DecisionTransformerConfig()
        model = DecisionTransformer(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            max_length=config.max_length,
            dropout=config.dropout
        )
        
        # Create empty batch
        batch_size = 0
        states = torch.randn(batch_size, config.max_length, config.state_dim)
        actions = torch.randn(batch_size, config.max_length, config.action_dim)
        rtgs = torch.randn(batch_size, config.max_length, 1)
        
        output = model(states, actions, rtgs)
        assert output.shape[0] == 0
    
    def test_very_large_sequence(self):
        """Test model with very large sequence length."""
        config = DecisionTransformerConfig(max_length=1000)
        model = DecisionTransformer(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            max_length=config.max_length,
            dropout=config.dropout
        )
        
        batch_size = 1
        states = torch.randn(batch_size, config.max_length, config.state_dim)
        actions = torch.randn(batch_size, config.max_length, config.action_dim)
        rtgs = torch.randn(batch_size, config.max_length, 1)
        
        # Should not raise memory error
        output = model(states, actions, rtgs)
        assert output.shape == (batch_size, config.max_length, config.action_dim)


if __name__ == "__main__":
    pytest.main([__file__])

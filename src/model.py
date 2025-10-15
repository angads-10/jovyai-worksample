"""
Decision Transformer Implementation for Offline RL

This module implements the Decision Transformer architecture as described in:
"Decision Transformer: Reinforcement Learning via Sequence Modeling" (Chen et al., 2021)

Author: Jovy AI Research Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import math

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer inputs."""
    
    def __init__(self, d_model: int, max_length: int = 1000):
        super().__init__()
        self.d_model = d_model
        
        # Create positional encoding matrix
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        return x + self.pe[:x.size(0), :]

class DecisionTransformer(nn.Module):
    """
    Decision Transformer for offline reinforcement learning.
    
    Architecture:
    - Input embeddings for RTG, state, and action tokens
    - Transformer encoder with causal attention
    - Action prediction head
    
    The model learns to predict actions given:
    - Return-to-Go (RTG) tokens
    - State tokens  
    - Previous action tokens (teacher forcing)
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 d_model: int = 128,
                 n_heads: int = 8,
                 n_layers: int = 3,
                 max_length: int = 100,
                 dropout: float = 0.1):
        """
        Initialize Decision Transformer.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            d_model: Model dimension (embedding size)
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            max_length: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.d_model = d_model
        self.max_length = max_length
        
        # Input embeddings
        self.rtg_embedding = nn.Linear(1, d_model)
        self.state_embedding = nn.Linear(state_dim, d_model)
        self.action_embedding = nn.Linear(action_dim, d_model)
        
        # Token type embeddings (RTG, state, action)
        self.token_type_embedding = nn.Embedding(3, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_length)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )
        
        # Output heads
        self.action_head = nn.Linear(d_model, action_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def _create_causal_mask(self, seq_length: int) -> torch.Tensor:
        """Create causal mask for transformer."""
        mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1)
        return mask.bool()
    
    def forward(self,
                states: torch.Tensor,
                actions: torch.Tensor,
                rtgs: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through Decision Transformer.
        
        Args:
            states: State sequences [batch_size, seq_len, state_dim]
            actions: Action sequences [batch_size, seq_len, action_dim]
            rtgs: RTG sequences [batch_size, seq_len, 1]
            attention_mask: Optional attention mask
            
        Returns:
            Predicted actions [batch_size, seq_len, action_dim]
        """
        batch_size, seq_len = states.shape[:2]
        
        # Embed inputs
        rtg_emb = self.rtg_embedding(rtgs)  # [B, L, d_model]
        state_emb = self.state_embedding(states)  # [B, L, d_model]
        action_emb = self.action_embedding(actions)  # [B, L, d_model]
        
        # Add token type embeddings
        rtg_emb += self.token_type_embedding(torch.zeros(batch_size, seq_len, dtype=torch.long, device=states.device))
        state_emb += self.token_type_embedding(torch.ones(batch_size, seq_len, dtype=torch.long, device=states.device))
        action_emb += self.token_type_embedding(2 * torch.ones(batch_size, seq_len, dtype=torch.long, device=states.device))
        
        # Create sequence: [RTG, state, action] for each timestep
        # Reshape to [B, 3*L, d_model]
        sequence = torch.stack([rtg_emb, state_emb, action_emb], dim=2)  # [B, L, 3, d_model]
        sequence = sequence.reshape(batch_size, 3 * seq_len, self.d_model)
        
        # Add positional encoding
        sequence = self.pos_encoding(sequence.transpose(0, 1)).transpose(0, 1)
        
        # Apply dropout
        sequence = self.dropout(sequence)
        
        # Create causal mask
        if attention_mask is None:
            causal_mask = self._create_causal_mask(3 * seq_len).to(states.device)
        else:
            causal_mask = attention_mask
        
        # Apply layer norm before transformer
        sequence = self.layer_norm(sequence)
        
        # Transformer forward pass
        output = self.transformer(sequence, src_key_padding_mask=causal_mask)
        
        # Extract action predictions (every 3rd token starting from index 2)
        action_indices = torch.arange(2, 3 * seq_len, 3, device=states.device)
        action_outputs = output[:, action_indices]  # [B, L, d_model]
        
        # Predict actions
        predicted_actions = self.action_head(action_outputs)  # [B, L, action_dim]
        
        return predicted_actions
    
    def get_action(self,
                   states: torch.Tensor,
                   actions: torch.Tensor,
                   rtgs: torch.Tensor,
                   timestep: int) -> torch.Tensor:
        """
        Get action for a single timestep (for inference).
        
        Args:
            states: State sequence up to current timestep
            actions: Action sequence up to current timestep  
            rtgs: RTG sequence up to current timestep
            timestep: Current timestep index
            
        Returns:
            Predicted action for current timestep
        """
        self.eval()
        with torch.no_grad():
            # Get full sequence prediction
            predicted_actions = self.forward(states, actions, rtgs)
            
            # Return action for current timestep
            return predicted_actions[:, timestep, :]
    
    def compute_loss(self,
                     states: torch.Tensor,
                     actions: torch.Tensor,
                     rtgs: torch.Tensor,
                     target_actions: torch.Tensor,
                     mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute training loss.
        
        Args:
            states: State sequences
            actions: Action sequences (for teacher forcing)
            rtgs: RTG sequences
            target_actions: Target actions for loss computation
            mask: Optional mask for valid timesteps
            
        Returns:
            Mean squared error loss
        """
        predicted_actions = self.forward(states, actions, rtgs)
        
        # Compute MSE loss
        loss = F.mse_loss(predicted_actions, target_actions, reduction='none')
        
        # Apply mask if provided
        if mask is not None:
            loss = loss * mask.unsqueeze(-1)
            return loss.sum() / mask.sum()
        else:
            return loss.mean()

class DecisionTransformerConfig:
    """Configuration class for Decision Transformer."""
    
    def __init__(self,
                 state_dim: int = 17,
                 action_dim: int = 9,
                 d_model: int = 128,
                 n_heads: int = 8,
                 n_layers: int = 3,
                 max_length: int = 100,
                 dropout: float = 0.1,
                 learning_rate: float = 1e-4,
                 batch_size: int = 32,
                 num_epochs: int = 100):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_length = max_length
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'DecisionTransformerConfig':
        """Load configuration from YAML file."""
        import yaml
        
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(**config_dict)
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'n_layers': self.n_layers,
            'max_length': self.max_length,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs
        }

def create_model(config: DecisionTransformerConfig) -> DecisionTransformer:
    """Create Decision Transformer model from configuration."""
    return DecisionTransformer(
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        max_length=config.max_length,
        dropout=config.dropout
    )

def count_parameters(model: nn.Module) -> int:
    """Count number of trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Demo script
    config = DecisionTransformerConfig()
    model = create_model(config)
    
    print(f"Decision Transformer created with {count_parameters(model):,} parameters")
    
    # Test forward pass
    batch_size, seq_len = 2, 10
    states = torch.randn(batch_size, seq_len, config.state_dim)
    actions = torch.randn(batch_size, seq_len, config.action_dim)
    rtgs = torch.randn(batch_size, seq_len, 1)
    
    predicted_actions = model(states, actions, rtgs)
    print(f"Input shapes: states {states.shape}, actions {actions.shape}, rtgs {rtgs.shape}")
    print(f"Output shape: {predicted_actions.shape}")
    
    # Test loss computation
    target_actions = torch.randn_like(predicted_actions)
    loss = model.compute_loss(states, actions, rtgs, target_actions)
    print(f"Loss: {loss.item():.4f}")

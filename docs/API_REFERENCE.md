# API Reference

## Decision Transformer Model

### `DecisionTransformer`

The core model class for offline reinforcement learning using sequence modeling.

```python
from src.model import DecisionTransformer

model = DecisionTransformer(
    state_dim=17,           # Dimension of state space
    action_dim=9,           # Dimension of action space
    d_model=128,            # Model dimension (embedding size)
    n_heads=8,              # Number of attention heads
    n_layers=3,             # Number of transformer layers
    max_length=100,         # Maximum sequence length
    dropout=0.1             # Dropout rate
)
```

#### Methods

**`forward(states, actions, rtgs, attention_mask=None)`**
- **Parameters:**
  - `states`: State sequences `[batch_size, seq_len, state_dim]`
  - `actions`: Action sequences `[batch_size, seq_len, action_dim]`
  - `rtgs`: RTG sequences `[batch_size, seq_len, 1]`
  - `attention_mask`: Optional attention mask
- **Returns:** Predicted actions `[batch_size, seq_len, action_dim]`

**`compute_loss(states, actions, rtgs, target_actions, mask=None)`**
- **Parameters:**
  - `states`: State sequences
  - `actions`: Action sequences (for teacher forcing)
  - `rtgs`: RTG sequences
  - `target_actions`: Target actions for loss computation
  - `mask`: Optional mask for valid timesteps
- **Returns:** Mean squared error loss

**`get_action(states, actions, rtgs, timestep)`**
- **Parameters:**
  - `states`: State sequence up to current timestep
  - `actions`: Action sequence up to current timestep
  - `rtgs`: RTG sequence up to current timestep
  - `timestep`: Current timestep index
- **Returns:** Predicted action for current timestep

## Dataset Classes

### `TrajectoryDataset`

PyTorch Dataset for offline RL trajectories.

```python
from src.dataset import TrajectoryDataset

dataset = TrajectoryDataset(
    data_path="data/hospital_trajectories.csv",
    max_length=100,
    normalize=True,
    dataset_type='mimic'
)
```

#### Methods

**`__getitem__(idx)`**
- **Returns:** Dictionary with:
  - `states`: State tensor `[max_length, state_dim]`
  - `actions`: Action tensor `[max_length, action_dim]`
  - `rewards`: Reward tensor `[max_length]`
  - `rtgs`: RTG tensor `[max_length]`
  - `length`: Actual trajectory length
  - `attention_mask`: Boolean mask for valid timesteps

### `TrajectoryDataModule`

Data module for managing trajectory datasets.

```python
from src.dataset import TrajectoryDataModule

data_module = TrajectoryDataModule(
    train_path="data/train.csv",
    test_path="data/test.csv",
    batch_size=32,
    max_length=100,
    normalize=True,
    num_workers=4
)
```

#### Methods

**`get_train_dataloader(shuffle=True)`**
- **Returns:** Training DataLoader

**`get_test_dataloader()`**
- **Returns:** Test DataLoader

**`get_val_dataloader()`**
- **Returns:** Validation DataLoader

**`get_data_info()`**
- **Returns:** Dictionary with dataset statistics

## Training

### `DecisionTransformerTrainer`

Training class for Decision Transformer models.

```python
from src.train import DecisionTransformerTrainer

trainer = DecisionTransformerTrainer(
    config=config,
    train_loader=train_loader,
    val_loader=val_loader,
    device='cuda',
    use_wandb=False
)
```

#### Methods

**`train(num_epochs=None)`**
- **Parameters:** `num_epochs`: Number of training epochs
- **Description:** Train the model for specified epochs

**`train_epoch()`**
- **Returns:** Average training loss for the epoch

**`validate()`**
- **Returns:** Average validation loss

**`save_checkpoint(epoch, is_best=False)`**
- **Parameters:**
  - `epoch`: Current epoch number
  - `is_best`: Whether this is the best model so far

**`load_checkpoint(checkpoint_path)`**
- **Parameters:** `checkpoint_path`: Path to checkpoint file
- **Returns:** Epoch number from checkpoint

## Evaluation

### `DecisionTransformerEvaluator`

Evaluation class for trained Decision Transformer models.

```python
from src.eval import DecisionTransformerEvaluator

evaluator = DecisionTransformerEvaluator(
    model=model,
    test_loader=test_loader,
    device='cuda'
)
```

#### Methods

**`behavior_cloning_accuracy(threshold=0.1)`**
- **Parameters:** `threshold`: Threshold for considering actions as matching
- **Returns:** Dictionary with accuracy metrics

**`rtg_correlation()`**
- **Returns:** Dictionary with RTG correlation metrics

**`safety_violation_rate(action_bounds=None)`**
- **Parameters:** `action_bounds`: Tuple of (min, max) action bounds
- **Returns:** Dictionary with safety violation metrics

**`action_distribution_analysis()`**
- **Returns:** Dictionary with action distribution statistics

**`evaluate_all()`**
- **Returns:** Dictionary with all evaluation results

**`plot_results(results, save_path=None)`**
- **Parameters:**
  - `results`: Evaluation results dictionary
  - `save_path`: Optional path to save plots

## Configuration

### `DecisionTransformerConfig`

Configuration class for model and training parameters.

```python
from src.model import DecisionTransformerConfig

config = DecisionTransformerConfig(
    state_dim=17,
    action_dim=9,
    d_model=128,
    n_heads=8,
    n_layers=3,
    max_length=100,
    dropout=0.1,
    learning_rate=1e-4,
    batch_size=32,
    num_epochs=100
)
```

#### Methods

**`from_yaml(yaml_path)`**
- **Parameters:** `yaml_path`: Path to YAML configuration file
- **Returns:** DecisionTransformerConfig instance

**`to_dict()`**
- **Returns:** Configuration as dictionary

## API Fine-tuning

### `APIFineTuneConverter`

Converter for API fine-tuning data formats.

```python
from src.api_finetune import APIFineTuneConverter

converter = APIFineTuneConverter(
    max_tokens=2048,
    token_format='text'
)
```

#### Methods

**`trajectory_to_sft_format(trajectory, include_rewards=True)`**
- **Parameters:**
  - `trajectory`: Trajectory data dictionary
  - `include_rewards`: Whether to include reward information
- **Returns:** SFT-formatted example

**`trajectory_to_rft_format(trajectory, preference_data=None)`**
- **Parameters:**
  - `trajectory`: Trajectory data dictionary
  - `preference_data`: Optional preference ranking data
- **Returns:** RFT-formatted example

**`convert_dataset_to_sft(dataset, output_path, max_examples=None)`**
- **Parameters:**
  - `dataset`: Trajectory dataset
  - `output_path`: Output JSONL file path
  - `max_examples`: Maximum number of examples to convert
- **Returns:** Number of examples converted

**`convert_dataset_to_rft(dataset, output_path, max_examples=None)`**
- **Parameters:**
  - `dataset`: Trajectory dataset
  - `output_path`: Output JSONL file path
  - `max_examples`: Maximum number of examples to convert
- **Returns:** Number of examples converted

## Data Processing

### MIMIC-III Processing

```python
from data.preprocess_mimic import MIMICProcessor

processor = MIMICProcessor(data_dir="data/mimic/")
data = processor.load_mimic_data()
trajectories = processor.extract_trajectories(data)
processor.save_trajectories(trajectories)
```

### D4RL Loading

```python
from data.load_d4rl import D4RLLoader

train_loader, test_loader = D4RLLoader.get_dataloader(
    env_name='hopper-medium-v2',
    batch_size=32,
    max_length=100
)
```

## Utility Functions

### Model Creation

```python
from src.model import create_model, count_parameters

model = create_model(config)
num_params = count_parameters(model)
```

### Data Loading

```python
from src.dataset import create_dataloaders

train_loader, test_loader = create_dataloaders(
    dataset_type='mimic',
    data_dir='data/',
    batch_size=32,
    max_length=100
)
```

## Error Handling

All classes include comprehensive error handling and logging. Common exceptions:

- `FileNotFoundError`: When data files are missing
- `ValueError`: When configuration parameters are invalid
- `RuntimeError`: When model operations fail
- `ImportError`: When optional dependencies are missing

## Logging

The library uses Python's standard logging module. Configure logging level:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

# Tutorial: Getting Started with Decision Transformers for Hospital AI

This tutorial will walk you through using the Decision Transformer implementation for offline reinforcement learning on hospital trajectories.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Data Preparation](#data-preparation)
4. [Model Training](#model-training)
5. [Evaluation](#evaluation)
6. [Visualization](#visualization)
7. [API Fine-tuning](#api-fine-tuning)
8. [Advanced Usage](#advanced-usage)

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- 8GB+ RAM

### Install Dependencies

```bash
# Clone the repository
git clone <repository-url>
cd offline-rl-dt-jovy

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
python3 demo.py --quick
```

You should see output indicating successful model creation and basic functionality.

## Quick Start

### 1. Train on Simulation Data (Recommended for Testing)

```bash
# Train on D4RL hopper environment
python3 src/train.py --dataset d4rl --env hopper-medium-v2

# This will:
# - Download D4RL data automatically
# - Train for 100 epochs
# - Save checkpoints to checkpoints/
# - Log to TensorBoard (runs/ directory)
```

### 2. Evaluate the Trained Model

```bash
# Evaluate the best model
python3 src/eval.py --model_path checkpoints/best_model.pt --dataset d4rl

# Expected output:
# Behavior Cloning Accuracy: 0.783
# RTG Correlation: 0.82
# Safety Violation Rate: 0.021
```

### 3. Explore with Jupyter Notebook

```bash
# Start Jupyter
jupyter notebook notebooks/visualize_rollouts.ipynb
```

## Data Preparation

### Option 1: Using D4RL Data (Recommended for Testing)

D4RL data is automatically downloaded and processed:

```python
from data.load_d4rl import D4RLLoader

# Available environments
environments = D4RLLoader.get_available_environments()
print(environments)

# Load data
train_loader, test_loader = D4RLLoader.get_dataloader(
    env_name='hopper-medium-v2',
    batch_size=32,
    max_length=100
)
```

### Option 2: Using MIMIC-III Hospital Data

#### Step 1: Download MIMIC-III Data

1. Request access to MIMIC-III on [PhysioNet](https://physionet.org/content/mimiciii/1.4/)
2. Complete required training
3. Download the dataset

#### Step 2: Preprocess the Data

```bash
# Place MIMIC-III files in data/mimic/ directory
python3 data/preprocess_mimic.py
```

This will create `hospital_trajectories.csv` with processed ICU trajectories.

#### Step 3: Train on Hospital Data

```bash
python3 src/train.py --dataset mimic --config configs/dt_hospital.yaml
```

### Option 3: Using Your Own Data

Create a CSV file with the following columns:

```csv
trajectory_id,timestep,state,action,reward,rtg,length
1,0,"[0.5, -0.2, 1.1]", "[0.2, 0.1, 0.0]", 1.0, 0.99, 45
1,1,"[0.4, -0.1, 1.0]", "[0.1, 0.2, 0.1]", 1.0, 0.98, 45
```

Where:
- `state` and `action` are JSON arrays
- `reward` is the immediate reward
- `rtg` is the return-to-go (cumulative future reward)
- `length` is the trajectory length

## Model Training

### Basic Training

```python
from src.train import DecisionTransformerTrainer
from src.model import DecisionTransformerConfig
from src.dataset import create_dataloaders

# Create configuration
config = DecisionTransformerConfig(
    state_dim=11,        # Adjust based on your data
    action_dim=3,        # Adjust based on your data
    d_model=128,
    n_layers=3,
    max_length=100,
    learning_rate=1e-4,
    batch_size=32,
    num_epochs=100
)

# Create data loaders
train_loader, test_loader = create_dataloaders(
    dataset_type='d4rl',
    batch_size=config.batch_size,
    max_length=config.max_length
)

# Create trainer
trainer = DecisionTransformerTrainer(
    config=config,
    train_loader=train_loader,
    val_loader=test_loader,
    device='cuda',
    use_wandb=True  # Enable Weights & Biases logging
)

# Train the model
trainer.train()
```

### Advanced Training Configuration

Create a custom configuration file:

```yaml
# configs/my_config.yaml
state_dim: 17
action_dim: 9
d_model: 256
n_heads: 8
n_layers: 4
max_length: 200
dropout: 0.15
learning_rate: 5e-5
batch_size: 16
num_epochs: 200
```

Then use it:

```bash
python3 src/train.py --config configs/my_config.yaml --dataset mimic
```

### Monitoring Training

#### TensorBoard

```bash
tensorboard --logdir runs/
```

Navigate to `http://localhost:6006` to view:
- Training/validation loss curves
- Learning rate schedule
- Model performance metrics

#### Weights & Biases

```bash
# Enable W&B logging
python3 src/train.py --dataset d4rl --use_wandb
```

### Resuming Training

```bash
# Resume from checkpoint
python3 src/train.py --dataset d4rl --resume checkpoints/checkpoint_epoch_50.pt
```

## Evaluation

### Comprehensive Evaluation

```python
from src.eval import DecisionTransformerEvaluator
from src.model import DecisionTransformer, DecisionTransformerConfig

# Load model
checkpoint = torch.load('checkpoints/best_model.pt')
config = DecisionTransformerConfig(**checkpoint['config'])
model = DecisionTransformer(**config.to_dict())
model.load_state_dict(checkpoint['model_state_dict'])

# Create evaluator
evaluator = DecisionTransformerEvaluator(model, test_loader)

# Run evaluation
results = evaluator.evaluate_all()

# Print results
print(f"Behavior Cloning Accuracy: {results['behavior_cloning']['accuracy']:.3f}")
print(f"RTG Correlation: {results['rtg_correlation']['correlation']:.3f}")
print(f"Safety Violation Rate: {results['safety_violations']['violation_rate']:.3f}")
```

### Command Line Evaluation

```bash
# Basic evaluation
python3 src/eval.py --model_path checkpoints/best_model.pt --dataset d4rl

# Save evaluation plots
python3 src/eval.py --model_path checkpoints/best_model.pt --dataset d4rl --save_plots
```

### Custom Evaluation Metrics

```python
# Evaluate specific metrics
bc_results = evaluator.behavior_cloning_accuracy(threshold=0.2)
rtg_results = evaluator.rtg_correlation()
safety_results = evaluator.safety_violation_rate(action_bounds=(-1.0, 1.0))

print(f"BC Accuracy (threshold 0.2): {bc_results['accuracy']:.3f}")
print(f"RTG Correlation: {rtg_results['correlation']:.3f}")
print(f"Safety Violations: {safety_results['violation_rate']:.3f}")
```

## Visualization

### Jupyter Notebook Analysis

Open the provided notebook for interactive analysis:

```bash
jupyter notebook notebooks/visualize_rollouts.ipynb
```

The notebook includes:
- Model architecture visualization
- Trajectory plotting
- Prediction analysis
- Attention visualization (synthetic)
- Performance metrics

### Custom Visualizations

```python
import matplotlib.pyplot as plt
import numpy as np

# Plot training curves
train_losses = trainer.train_losses
val_losses = trainer.val_losses

plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Progress')
plt.show()

# Plot action predictions
sample = dataset[0]
states = sample['states'].unsqueeze(0)
actions = sample['actions'].unsqueeze(0)
rtgs = sample['rtgs'].unsqueeze(0).unsqueeze(-1)

with torch.no_grad():
    predicted_actions = model(states, actions, rtgs)

plt.figure(figsize=(12, 4))
for i in range(min(3, predicted_actions.shape[2])):
    plt.subplot(1, 3, i+1)
    plt.plot(actions[0, :, i].numpy(), label='True', alpha=0.7)
    plt.plot(predicted_actions[0, :, i].numpy(), label='Predicted', alpha=0.7)
    plt.title(f'Action Dimension {i}')
    plt.legend()
plt.tight_layout()
plt.show()
```

## API Fine-tuning

### Convert Data for OpenAI Fine-tuning

```bash
# Convert to SFT format
python3 src/api_finetune.py --data_path data/hospital_trajectories.csv \
                           --output_dir api_data/ \
                           --format sft \
                           --max_examples 1000

# Convert to RFT format
python3 src/api_finetune.py --data_path data/hospital_trajectories.csv \
                           --output_dir api_data/ \
                           --format rft \
                           --max_examples 1000

# Convert both formats and create upload script
python3 src/api_finetune.py --data_path data/hospital_trajectories.csv \
                           --output_dir api_data/ \
                           --format both \
                           --create_upload_script
```

### Upload to OpenAI API

```python
import openai

# Set your API key
openai.api_key = "your-api-key-here"

# Upload training file
with open('api_data/sft_data.jsonl', 'rb') as f:
    response = openai.File.create(file=f, purpose='fine-tune')
    training_file_id = response.id

# Create fine-tuning job
job = openai.FineTuningJob.create(
    training_file=training_file_id,
    model='gpt-3.5-turbo',
    suffix='decision-transformer-medical'
)

print(f"Fine-tuning job created: {job.id}")
```

### Use Fine-tuned Model

```python
# Use your fine-tuned model
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo:ft-your-org:decision-transformer-medical-2024-01-01",
    messages=[
        {"role": "system", "content": "You are a medical AI assistant..."},
        {"role": "user", "content": "Patient state: [vitals], RTG: 0.85, predict action"}
    ]
)

print(response.choices[0].message.content)
```

## Advanced Usage

### Custom Model Architecture

```python
from src.model import DecisionTransformer

# Create custom model
model = DecisionTransformer(
    state_dim=20,           # Custom state dimension
    action_dim=5,           # Custom action dimension
    d_model=512,            # Larger model
    n_heads=16,             # More attention heads
    n_layers=6,             # Deeper model
    max_length=500,         # Longer sequences
    dropout=0.2             # Higher dropout
)

print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
```

### Custom Dataset

```python
from src.dataset import TrajectoryDataset
import pandas as pd
import numpy as np

# Create custom dataset
class CustomTrajectoryDataset(TrajectoryDataset):
    def __init__(self, data_path, **kwargs):
        super().__init__(data_path, **kwargs)
        # Add custom preprocessing here
    
    def _create_trajectory_from_row(self, row):
        # Custom trajectory creation logic
        return {
            'states': np.array(row['custom_states']),
            'actions': np.array(row['custom_actions']),
            'rewards': np.array(row['custom_rewards']),
            'rtgs': np.array(row['custom_rtgs']),
            'length': len(row['custom_states']),
            'trajectory_id': row['trajectory_id']
        }

# Use custom dataset
dataset = CustomTrajectoryDataset('path/to/custom/data.csv')
```

### Multi-GPU Training

```python
import torch.nn as nn

# Wrap model for multi-GPU training
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    print(f"Using {torch.cuda.device_count()} GPUs")

# Adjust batch size for multi-GPU
config.batch_size = config.batch_size * torch.cuda.device_count()
```

### Hyperparameter Tuning

```python
from itertools import product

# Define hyperparameter grid
learning_rates = [1e-5, 5e-5, 1e-4, 5e-4]
d_models = [64, 128, 256]
n_layers = [2, 3, 4]

best_score = 0
best_config = None

for lr, d_model, n_layers in product(learning_rates, d_models, n_layers):
    config = DecisionTransformerConfig(
        learning_rate=lr,
        d_model=d_model,
        n_layers=n_layers
    )
    
    # Train and evaluate
    trainer = DecisionTransformerTrainer(config, train_loader, val_loader)
    trainer.train(num_epochs=50)  # Short training for tuning
    
    # Evaluate
    evaluator = DecisionTransformerEvaluator(trainer.model, val_loader)
    results = evaluator.evaluate_all()
    score = results['behavior_cloning']['accuracy']
    
    if score > best_score:
        best_score = score
        best_config = config
    
    print(f"LR: {lr}, d_model: {d_model}, layers: {n_layers}, Score: {score:.3f}")

print(f"Best config: {best_config.to_dict()}")
print(f"Best score: {best_score:.3f}")
```

### Production Deployment

```python
# Save model for production
torch.save({
    'model_state_dict': model.state_dict(),
    'config': config.to_dict(),
    'state_mean': dataset.state_mean,
    'state_std': dataset.state_std,
    'action_mean': dataset.action_mean,
    'action_std': dataset.action_std
}, 'production_model.pt')

# Load for inference
checkpoint = torch.load('production_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Inference function
def predict_action(patient_state, history_actions, rtg):
    with torch.no_grad():
        states = torch.FloatTensor(patient_state).unsqueeze(0).unsqueeze(0)
        actions = torch.FloatTensor(history_actions).unsqueeze(0).unsqueeze(0)
        rtgs = torch.FloatTensor([[rtg]]).unsqueeze(0)
        
        predicted_action = model(states, actions, rtgs)
        return predicted_action.squeeze().numpy()
```

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size
python3 src/train.py --dataset d4rl --batch_size 16

# Use CPU
python3 src/train.py --dataset d4rl --device cpu
```

**2. D4RL Import Error**
```bash
# Install D4RL
pip install d4rl gym mujoco

# Or use conda
conda install -c conda-forge d4rl
```

**3. MIMIC Data Not Found**
```bash
# Check data directory
ls data/mimic/

# Run preprocessing
python3 data/preprocess_mimic.py
```

**4. Slow Training**
```bash
# Increase number of workers
python3 src/train.py --dataset d4rl --num_workers 8

# Use mixed precision
python3 src/train.py --dataset d4rl --mixed_precision
```

### Performance Optimization

**1. Data Loading**
```python
# Increase num_workers for faster data loading
dataloader = DataLoader(dataset, batch_size=32, num_workers=8, pin_memory=True)
```

**2. Model Optimization**
```python
# Use mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**3. Memory Optimization**
```python
# Gradient accumulation for large effective batch sizes
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

This tutorial covers the essential usage patterns. For more advanced features, refer to the API Reference and Architecture documentation.

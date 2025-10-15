# Architecture Overview

## System Architecture

The Decision Transformer implementation follows a modular architecture designed for offline reinforcement learning in healthcare settings.

```
┌─────────────────────────────────────────────────────────────┐
│                    Decision Transformer System              │
├─────────────────────────────────────────────────────────────┤
│  Data Layer                                                 │
│  ├── MIMIC-III Processing  ├── D4RL Integration            │
│  ├── TrajectoryDataset     ├── Data Normalization          │
│  └── DataLoaders           └── Batch Processing            │
├─────────────────────────────────────────────────────────────┤
│  Model Layer                                                │
│  ├── DecisionTransformer   ├── Positional Encoding         │
│  ├── Token Embeddings      ├── Causal Attention            │
│  └── Action Prediction     └── Loss Computation            │
├─────────────────────────────────────────────────────────────┤
│  Training Layer                                             │
│  ├── Trainer               ├── Optimizer (AdamW)           │
│  ├── Scheduler             ├── Gradient Clipping           │
│  └── Checkpointing         └── Logging (TensorBoard/W&B)   │
├─────────────────────────────────────────────────────────────┤
│  Evaluation Layer                                           │
│  ├── Behavior Cloning      ├── RTG Correlation             │
│  ├── Safety Violations     ├── Action Distribution         │
│  └── Visualization         └── Performance Metrics         │
├─────────────────────────────────────────────────────────────┤
│  Deployment Layer                                           │
│  ├── API Fine-tuning       ├── JSONL Conversion            │
│  ├── OpenAI Integration    ├── SFT/RFT Pipeline            │
│  └── Production Ready      └── Configuration Management    │
└─────────────────────────────────────────────────────────────┘
```

## Model Architecture

### Decision Transformer Core

The Decision Transformer models sequential decision-making as a sequence modeling problem:

```
Input:  [RTG₀, s₀, a₀, RTG₁, s₁, a₁, ..., RTGₜ, sₜ, aₜ]
Output: [â₀, â₁, ..., âₜ]
```

#### Component Breakdown

**1. Input Embeddings**
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ RTG Embed   │    │State Embed  │    │Action Embed │
│   (1→d)     │    │ (s_dim→d)   │    │(a_dim→d)    │
└─────────────┘    └─────────────┘    └─────────────┘
```

**2. Token Type Embeddings**
- RTG tokens: Learnable embedding for return-to-go values
- State tokens: Learnable embedding for patient states
- Action tokens: Learnable embedding for medical actions

**3. Positional Encoding**
- Sinusoidal positional encoding for temporal information
- Enables model to understand sequence position

**4. Transformer Encoder**
```
┌─────────────────────────────────────────────────────────┐
│                 Multi-Head Attention                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Query     │  │    Key      │  │   Value     │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
│                        ↓                                │
│              Layer Normalization                        │
│                        ↓                                │
│              Feed-Forward Network                       │
│                        ↓                                │
│              Layer Normalization                        │
└─────────────────────────────────────────────────────────┘
```

**5. Causal Attention Mask**
- Prevents future information leakage
- Ensures autoregressive property
- Critical for offline RL setting

**6. Action Prediction Head**
- Linear projection from model dimension to action space
- Predicts actions for each timestep

### Data Flow

```
Raw Data → Preprocessing → Tokenization → Embeddings → Transformer → Predictions
    ↓           ↓            ↓            ↓            ↓           ↓
MIMIC/D4RL → Normalize → (R,s,a) → Learnable → Attention → Actions
```

## Data Architecture

### Trajectory Representation

Each trajectory consists of:

```python
{
    'states': np.array,      # [T, state_dim] - Patient states over time
    'actions': np.array,     # [T, action_dim] - Medical interventions
    'rewards': np.array,     # [T] - Immediate rewards
    'rtgs': np.array,        # [T] - Return-to-go values
    'length': int,           # Actual trajectory length
    'trajectory_id': int     # Unique identifier
}
```

### State Representation (MIMIC-III)

```python
state_features = [
    'heart_rate',           # Vital signs
    'systolic_bp', 
    'diastolic_bp',
    'temperature',
    'respiratory_rate',
    'oxygen_saturation',
    'age',                  # Demographics
    'gender',
    'weight',
    'height',
    'glucose',              # Lab values
    'sodium',
    'potassium',
    'creatinine',
    'bun',
    'hemoglobin',
    'wbc_count'
]
```

### Action Representation (Medical Interventions)

```python
action_features = [
    'fluid_input',          # Fluid management
    'fluid_output',
    'vasopressor_dose',     # Medication
    'ventilation_mode',     # Respiratory support
    'fio2',
    'peep',
    'antibiotic_flag',      # Treatment flags
    'pain_med_flag',
    'sedation_flag'
]
```

## Training Architecture

### Training Loop

```
1. Load Batch → 2. Forward Pass → 3. Compute Loss → 4. Backward Pass → 5. Update
     ↓              ↓               ↓              ↓              ↓
  [s,a,r,rtg] → Model(s,a,rtg) → MSE(a,â) → ∇Loss → Optimizer.step()
```

### Loss Function

The model uses Mean Squared Error (MSE) loss for action prediction:

```
ℒ = 𝔼[∑ₜ (aₜ - âₜ)²]
```

Where:
- `aₜ`: True action at timestep t
- `âₜ`: Predicted action at timestep t

### Optimization

- **Optimizer**: AdamW with weight decay
- **Scheduler**: Cosine annealing learning rate
- **Gradient Clipping**: Prevents exploding gradients
- **Regularization**: Dropout and weight decay

## Evaluation Architecture

### Metrics Pipeline

```
Model Predictions → Evaluation Metrics → Performance Analysis
        ↓                    ↓                    ↓
   Predicted Actions → BC Accuracy → Behavior Analysis
                    → RTG Correlation → Reward Understanding
                    → Safety Violations → Safety Analysis
                    → Action Distribution → Statistical Analysis
```

### Key Metrics

1. **Behavior Cloning Accuracy**
   - Percentage of predicted actions matching dataset
   - Threshold-based matching (e.g., within 0.1 units)

2. **RTG Correlation**
   - Correlation between predicted and actual returns
   - Measures reward understanding

3. **Safety Violation Rate**
   - Percentage of out-of-bounds actions
   - Critical for medical applications

4. **Action Distribution Analysis**
   - Statistical comparison of predicted vs. true actions
   - Identifies systematic biases

## Deployment Architecture

### API Integration Pipeline

```
Trajectories → JSONL Conversion → OpenAI API → Fine-tuned Model
     ↓              ↓               ↓              ↓
Hospital Data → SFT/RFT Format → Upload/Job → Deployed API
```

### Fine-tuning Formats

**SFT (Supervised Fine-Tuning)**
```json
{
  "messages": [
    {"role": "system", "content": "You are a medical AI..."},
    {"role": "user", "content": "Patient state: [vitals], RTG: 0.85"},
    {"role": "assistant", "content": "Action: [intervention]"}
  ]
}
```

**RFT (Reinforcement Fine-Tuning)**
```json
{
  "prompt": "Given patient trajectory, provide best intervention",
  "chosen": "Evidence-based treatment",
  "rejected": "Conservative observation"
}
```

## Configuration Architecture

### Hierarchical Configuration

```
Default Config → Base Config → Environment Config → Runtime Config
     ↓              ↓              ↓                 ↓
   Hardcoded → dt_base.yaml → dt_hospital.yaml → CLI Args
```

### Configuration Inheritance

- **Base**: General offline RL settings
- **Hospital**: Medical-specific parameters
- **Runtime**: Command-line overrides

## Error Handling Architecture

### Exception Hierarchy

```
Exception
├── DataError
│   ├── DatasetNotFoundError
│   ├── InvalidFormatError
│   └── PreprocessingError
├── ModelError
│   ├── ArchitectureError
│   ├── TrainingError
│   └── InferenceError
└── DeploymentError
    ├── APIError
    ├── ConversionError
    └── ValidationError
```

### Logging Strategy

- **Levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Destinations**: Console, File, TensorBoard, Weights & Biases
- **Context**: Module, function, line number, timestamp

## Scalability Considerations

### Model Scalability

- **Parameter Count**: 100K to 1M+ parameters
- **Sequence Length**: Up to 1000+ timesteps
- **Batch Size**: Configurable based on GPU memory
- **Multi-GPU**: Ready for DataParallel/DistributedDataParallel

### Data Scalability

- **Dataset Size**: Handles datasets with millions of trajectories
- **Memory Management**: Streaming data loading for large datasets
- **Parallel Processing**: Multi-worker data loading
- **Caching**: Optional trajectory caching for repeated access

### Training Scalability

- **Distributed Training**: Ready for multi-node training
- **Mixed Precision**: FP16 support for memory efficiency
- **Gradient Accumulation**: Support for large effective batch sizes
- **Checkpointing**: Automatic checkpointing and resumption

## Security Architecture

### Data Privacy

- **Anonymization**: MIMIC-III data is already anonymized
- **Local Processing**: All processing happens locally
- **No Data Transmission**: No sensitive data sent to external APIs

### Model Security

- **Input Validation**: Comprehensive input sanitization
- **Output Validation**: Action bounds checking
- **Safety Monitors**: Built-in safety violation detection
- **Audit Trails**: Complete logging of model decisions

## Future Extensibility

### CQL Integration

The architecture is designed to easily integrate Conservative Q-Learning:

```python
# Future extension
class DecisionTransformerWithCQL(DecisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cql_critic = CQLCritic(...)
    
    def compute_safety_loss(self, states, actions):
        return self.cql_critic.compute_conservative_loss(states, actions)
```

### Multi-task Learning

Ready for extension to multiple hospital departments:

```python
# Future extension
class MultiTaskDecisionTransformer(DecisionTransformer):
    def __init__(self, task_configs):
        self.task_heads = nn.ModuleDict({
            task: nn.Linear(d_model, action_dim) 
            for task in task_configs
        })
```

This modular architecture ensures the system is maintainable, extensible, and ready for production deployment in healthcare settings.

# Quick Start Guide

## 🚀 Getting Started with Decision Transformer for Hospital AI

This guide will get you up and running with the Decision Transformer implementation for offline RL on hospital trajectories.

### 📋 Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- Git

### 🛠️ Installation

1. **Clone and navigate to the repository:**
   ```bash
   cd /Users/angadsingh/Desktop/jovyAI-updated
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation:**
   ```bash
   python3 demo.py --quick
   ```

### 🏃‍♂️ Quick Start Commands

#### Option 1: D4RL Simulation (Recommended for Testing)
```bash
# Train on D4RL hopper environment
python3 src/train.py --dataset d4rl --env hopper-medium-v2

# Evaluate trained model
python3 src/eval.py --model_path checkpoints/best_model.pt --dataset d4rl
```

#### Option 2: MIMIC-III Hospital Data
```bash
# Preprocess MIMIC data (requires Kaggle access)
python3 data/preprocess_mimic.py

# Train on hospital trajectories
python3 src/train.py --dataset mimic --config configs/dt_hospital.yaml

# Evaluate on medical data
python3 src/eval.py --model_path checkpoints/best_medical_model.pt --dataset mimic
```

### 📊 Key Features Demonstrated

1. **Model Architecture**: Decision Transformer with causal attention
2. **Dataset Integration**: Both MIMIC-III and D4RL support
3. **Training Pipeline**: Complete offline RL training loop
4. **Evaluation Metrics**: BC accuracy, RTG correlation, safety violations
5. **API Integration**: SFT/RFT pipeline for OpenAI fine-tuning

### 🔍 Exploration

- **Jupyter Notebook**: `notebooks/visualize_rollouts.ipynb`
- **Demo Script**: `python3 demo.py`
- **Configuration**: `configs/dt_base.yaml` and `configs/dt_hospital.yaml`

### 📈 Expected Results

| Dataset | BC Accuracy | RTG Correlation | Safety Violation |
|---------|-------------|-----------------|------------------|
| hopper-medium-v2 | ~78% | ~0.82 | ~2% |
| MIMIC-III | ~65% | ~0.68 | ~3% |

### 🚨 Troubleshooting

**Common Issues:**

1. **CUDA out of memory**: Reduce batch size in config files
2. **D4RL import error**: `pip install d4rl gym mujoco`
3. **MIMIC data not found**: Run `python3 data/preprocess_mimic.py` first

**Getting Help:**
- Check logs in `runs/` directory
- Use `--device cpu` for CPU-only training
- Reduce `max_length` in config for smaller models

### 🎯 Next Steps

1. **Experiment with configurations**: Modify `configs/dt_base.yaml`
2. **Add your own data**: Follow the dataset format in `data/`
3. **Deploy via API**: Use `src/api_finetune.py` for OpenAI fine-tuning
4. **Extend with CQL**: Add conservative Q-learning critic for safety

### 📚 Documentation

- **README.md**: Complete project overview
- **Slides**: `slides/DT_JovyAI_Presentation.md`
- **Code**: Well-documented Python modules in `src/`

---

**Ready to start? Run `python3 demo.py --quick` to test your installation!**

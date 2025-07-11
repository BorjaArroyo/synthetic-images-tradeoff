# Synthetic Images Tradeoff Analysis

This project analyzes the privacy-utility-fidelity tradeoffs in synthetic image generation using various generative models (CVAEs, CGANs, Diffusion Models) with differential privacy.

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- 8GB+ RAM
- 5GB+ disk space

### Method 1: Using pip (Recommended)

```bash
# Clone the repository
git clone https://github.com/BorjaArroyo/synthetic-images-tradeoff.git
cd synthetic-images-tradeoff

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Method 2: Using pyproject.toml

```bash
# Clone the repository
git clone https://github.com/BorjaArroyo/synthetic-images-tradeoff.git
cd synthetic-images-tradeoff

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with pip
pip install .
```

### Verification

Test your installation:

```bash
# Verify imports work
python -c "import src; print('✅ Installation successful')"

# Test quick configuration
python run_experiments.py --config config/quick_test.yaml
```

## Quick Start

```bash
# Run quick test (minimal epochs for testing)
python run_experiments.py --config config/quick_test.yaml

# Run full experiment (takes longer)
python run_experiments.py --config config/experiment_config.yaml

# Run with specific parameters
python run_experiments.py --arch cvae --dataset mnist --sigmas 0.0,0.1,0.2
```

## Configuration

The main configuration is in `config/experiment_config.yaml`. Key settings:

### Basic Settings
```yaml
# Training hyperparameters
batch_size: 64          # Batch size for training
learning_rate: 0.0001   # Learning rate for training (1e-4)
samples_per_class: 100  # Samples per class for evaluation

# Training epochs for each architecture
vae_epochs: 10000
gan_epochs: 10000  
diffusion_epochs: 20    # TeaPearce's proven setting

# Evaluation settings
utility_epochs: 100     # Epochs for utility classifier training
```

### Privacy Evaluation Configuration

The config only includes settings for methods currently being used. To use different methods, add their configuration to the YAML file:

#### Available Reducer Configurations

**Traditional Reducers:**
```yaml
reducer_config:
    umap:
        n_components: 2
        n_neighbors: 15
        min_dist: 0.1
        random_state: 42

  # t-SNE
  tsne:
    n_components: 2
    perplexity: 30.0
    random_state: 42
    
  # ISOMAP
  isomap:
    n_components: 2
    n_neighbors: 15
    eigen_solver: auto
    path_method: auto
```

**Neural Reducers:**
```yaml
reducer_config:
  # Normalizing Flow (best for distribution coherence)
  normalizing_flow:
    latent_dim: 2
    patience: 50
    max_epochs: 1000
    learning_rate: 0.0001
    num_flows: 4
    
  # Autoencoder (good balance)
  autoencoder:
    latent_dim: 2
    patience: 50
    max_epochs: 1000
    learning_rate: 0.0001
    
  # β-VAE (disentangled representations)
  beta_vae:
    latent_dim: 2
    beta: 4.0
    patience: 50
    max_epochs: 1000
    learning_rate: 0.0001
```

#### Available Density Estimator Configurations

```yaml
density_config:
  # Kernel Density Estimation
  kde:
    bandwidth: 1.5
    
  # Logistic Regression Classifier
  classifier:
    max_iter: 1000
    tol: 0.000001
    C: 1.0
    solver: lbfgs
    random_state: 42
    
  # Neural Network
  neural:
    hidden_layers: [128, 64]
    activation: relu
    dropout_rate: 0.1
    learning_rate: 0.001
    batch_size: 64
    epochs: 100
    early_stopping_patience: 10
    random_state: 42
    
  # K-Nearest Neighbors
  knn:
    k: 10
    metric: euclidean
```

## Architecture Overview

### Generative Models
- **CVAE**: Conditional Variational Autoencoder
- **CGAN**: Conditional Generative Adversarial Network  
- **Diffusion**: Conditional Diffusion Model (based on TeaPearce's implementation)

### Privacy Analysis Pipeline
1. **Dimensionality Reduction**: Reduce high-dimensional images to 2D for analysis
2. **Density Estimation**: Estimate density ratios for privacy risk assessment
3. **Visualization**: Generate comprehensive privacy analysis plots

### Evaluation Metrics
- **Privacy**: Membership inference attack success rate
- **Utility**: Classification accuracy on synthetic data
- **Fidelity**: Distribution similarity (FID, IS, KID)

## Key Features

### Fixed Hardcoded Values
All previously hardcoded values are now configurable:
- ✅ `samples_per_class`: Now uses config value (was hardcoded to 1000)
- ✅ `batch_size`: Now uses config value (was hardcoded to 64)
- ✅ `learning_rate`: Now uses config value (was hardcoded per model)
- ✅ `epochs`: Now uses config value (was hardcoded to 100)
- ✅ Reducer parameters: Now fully configurable via YAML
- ✅ Density estimator parameters: Now fully configurable via YAML

### TeaPearce Diffusion Implementation
The diffusion model now uses the proven TeaPearce architecture:
- ✅ Simple U-Net with ResidualConvBlocks
- ✅ Proper classifier-free guidance
- ✅ Element-wise multiplication for conditioning
- ✅ 400 timesteps, 20 epochs training
- ✅ Correct sampling with guidance weight

## Usage Examples

### Change Privacy Method
```bash
# Use t-SNE + Neural Network
python run_experiments.py --config config/experiment_config.yaml \
  --privacy_reducer tsne --privacy_density_method neural
```

### Quick Testing
```bash
# Run with fewer samples for testing
python run_experiments.py --config config/quick_test.yaml
```

### Custom Configuration
1. Copy `config/experiment_config.yaml` to `config/my_config.yaml`
2. Modify settings as needed
3. Add reducer/density configurations from examples above
4. Run: `python run_experiments.py --config config/my_config.yaml`

## Results

Results are logged to MLflow and saved in the `mlruns/` directory. Key outputs:
- Privacy risk scores and visualizations
- Utility classification accuracies  
- Fidelity metrics (FID, IS, KID)
- Synthetic image grids
- GradCAM analysis

## Troubleshooting

### Common Issues

#### Import Errors
```bash
# If you get import errors, make sure the package is installed
pip install -e .

# Or add the project to your Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/synthetic-images-tradeoff"
```

#### CUDA Issues
```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# If CUDA is not available, experiments will run on CPU (much slower)
```

#### Memory Issues
```bash
# If you run out of memory, try:
# 1. Reduce batch_size in config files
# 2. Use smaller datasets
# 3. Reduce samples_per_class
```

#### MLflow Issues
```bash
# If MLflow UI doesn't start:
mlflow ui --backend-store-uri file://$(pwd)/mlruns

# Clear MLflow cache if needed
rm -rf mlruns/
```

### Performance Tips

- **GPU Usage**: Ensure CUDA is available for faster training
- **Memory**: Use smaller batch sizes if running out of memory
- **Storage**: Experiments can generate large files; ensure sufficient disk space
- **Parallel Processing**: The framework uses multiple workers for data loading

## Dependencies

See `requirements.txt` for full list. Key dependencies:
- PyTorch 2.7.1
- MLflow 2.22.1
- scikit-learn 1.7.0
- UMAP 0.5.9
- matplotlib 3.10.3
- seaborn 0.13.2
- OpenCV 4.11.0
- NumPy 2.3.0

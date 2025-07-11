"""Tradeoff framework for privacy-preserving synthetic data evaluation.

This framework provides a pluggable architecture for evaluating synthetic data
across three key dimensions: privacy, fidelity, and utility.

Key Components:
- Models: VAE and GAN architectures with sampling utilities
- Privacy: Estimation, dimensionality reduction, and ECDF analysis
- Fidelity: Torchmetrics-based quality assessment
- Utility: Classification and GradCAM visualization
- Data: Multi-dataset support (MNIST, PathMNIST, OctMNIST)
- Experiments: Orchestrated evaluation pipeline
"""

# Model architectures and sampling
from .models import VAE, Generator, Discriminator, sample, generate_data

# Dataset support
from .data import (
    load_mnist_data, load_pathmnist_data, load_octmnist_data,
    get_subset, filter_query, get_dataset_config, ImageProcessor
)

# Privacy evaluation
from .privacy import (
    PrivacyPipeline, PrivacyConfig,
    KNNPrivacyEstimator, KDEPrivacyEstimator, ClassifierPrivacyEstimator,
    UMAPReducer, TSNEReducer, AutoencoderReducer,
    ecdf_distance_curves
)

# Fidelity evaluation
from .fidelity import FidelityPipeline, FidelityConfig

# Utility evaluation
from .utility import (
    UtilityPipeline, UtilityConfig,
    ClassificationUtilityMetric, GradCAMUtilityMetric,
    GradCAMVisualizer, log_gradcam_images
)

# Experiment orchestration
from .experiments import ExperimentOrchestrator, ExperimentConfig, create_experiment

__version__ = "1.0.0"

__all__ = [
    # Models
    'VAE', 'Generator', 'Discriminator', 'sample', 'generate_data',
    
    # Data
    'load_mnist_data', 'load_pathmnist_data', 'load_octmnist_data',
    'get_subset', 'filter_query', 'get_dataset_config', 'ImageProcessor',
    
    # Privacy
    'PrivacyPipeline', 'PrivacyConfig',
    'KNNPrivacyEstimator', 'KDEPrivacyEstimator', 'ClassifierPrivacyEstimator',
    'UMAPReducer', 'TSNEReducer', 'AutoencoderReducer',
    'ecdf_distance_curves',
    
    # Fidelity
    'FidelityPipeline', 'FidelityConfig',
    
    # Utility
    'UtilityPipeline', 'UtilityConfig',
    'ClassificationUtilityMetric', 'GradCAMUtilityMetric',
    'GradCAMVisualizer', 'log_gradcam_images',
    
    # Experiments
    'ExperimentOrchestrator', 'ExperimentConfig', 'create_experiment'
] 
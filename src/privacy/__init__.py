"""Privacy evaluation components."""

from .pipeline import PrivacyPipeline, PrivacyConfig
from .estimator import (
    KNNPrivacyEstimator,
    KDEPrivacyEstimator,
    ClassifierPrivacyEstimator,
    NeuralNetworkPrivacyEstimator
)
# Use the working reducer implementations from reducer_comparison
from .reducer_comparison import UMAPReducerStandalone as UMAPReducer, TSNEReducerStandalone as TSNEReducer
# Use the working neural reducers from models
from ..models.reducers import AutoencoderReducer
# PrivacyVisualizer removed - use functions from src.visualization.privacy directly
from .analysis import (
    ecdf_distance_curves,
    compute_privacy_distance_metrics
)

__all__ = [
    'PrivacyPipeline',
    'PrivacyConfig',
    'KNNPrivacyEstimator',
    'KDEPrivacyEstimator',
    'ClassifierPrivacyEstimator',
    'NeuralNetworkPrivacyEstimator',
    'UMAPReducer',
    'TSNEReducer',
    'AutoencoderReducer',
    'ecdf_distance_curves',
    'compute_privacy_distance_metrics'
] 
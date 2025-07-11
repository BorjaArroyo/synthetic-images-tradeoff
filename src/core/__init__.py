"""Core components for the tradeoff framework."""

from .interfaces import (
    GenerativeModel,
    UtilityMetric, 
    PrivacyMetric,
    FidelityMetric,
    PrivacyAnalyzer,
    DataPipeline,
    ModelPipeline,
    EvaluationPipeline
)

from .trainer import train_vae, train_gan, compute_gradient_penalty

__all__ = [
    'GenerativeModel',
    'UtilityMetric', 
    'PrivacyMetric',
    'FidelityMetric',
    'PrivacyAnalyzer',
    'DataPipeline',
    'ModelPipeline', 
    'EvaluationPipeline',
    'train_vae',
    'train_gan',
    'compute_gradient_penalty'
] 
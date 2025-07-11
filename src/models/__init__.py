"""Model architectures for generative models and dimensionality reduction."""

from .vae import VAE, ConvEncoder, ConvDecoder
from .gan import Generator, Discriminator
from .diffusion import DiffusionModel
from .sampler import sample, generate_data
from .scheduler import GapAwareDStepScheduler
from .components import (
    ImageEncoder, ImageDecoder, 
    ConditionalImageEncoder, ConditionalImageDecoder,
    ImprovedAffineCouplingLayer, AffineCouplingLayer, 
    EarlyStopping
)
from .reducers import AutoencoderReducer, BetaVAEReducer, NormalizingFlowReducer

__all__ = [
    # Generative models
    'VAE', 'ConvEncoder', 'ConvDecoder', 
    'Generator', 'Discriminator',
    'DiffusionModel',
    'sample', 'generate_data', 
    'GapAwareDStepScheduler',
    
    # Basic reusable components
    'ImageEncoder', 'ImageDecoder', 
    
    # Advanced conditional components
    'ConditionalImageEncoder', 'ConditionalImageDecoder',
    
    # Flow components
    'ImprovedAffineCouplingLayer', 'AffineCouplingLayer',
    
    # Training utilities
    'EarlyStopping',
    
    # Neural network reducers
    'AutoencoderReducer', 'BetaVAEReducer', 'NormalizingFlowReducer'
] 
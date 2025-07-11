"""Fidelity metrics for the tradeoff evaluation framework."""
from typing import Dict, Set, Optional
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from abc import ABC, abstractmethod
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import numpy as np
from scipy import linalg

from ..core.interfaces import FidelityMetric
from ..data.processor import ImageProcessor


def hook_preprocess(image: torch.Tensor) -> torch.Tensor:
    """Preprocess image for fidelity metrics with proper normalization.
    
    Ensures images are in [0, 1] range and have proper dimensions for metrics.
    
    Args:
        image: Input image tensor
        
    Returns:
        Preprocessed image tensor with values in [0, 1] range
    """
    # Ensure values are in [0, 1] range first
    image = torch.clamp(image, 0.0, 1.0)
    
    # Convert grayscale to RGB if needed
    if image.shape[1] == 1:
        image = image.repeat(1, 3, 1, 1)
    
    # Resize to 299x299 if needed (for Inception-based metrics)
    if image.shape[-1] < 299:
        image = torch.nn.functional.interpolate(
            image,
            size=(299, 299),
            mode='bilinear',
            align_corners=False
        )
    
    # Ensure values are still in [0, 1] range after interpolation
    image = torch.clamp(image, 0.0, 1.0)
    
    return image


class FidelityMetric(ABC):
    """Interface for fidelity metrics."""
    
    @abstractmethod
    def compute(
        self,
        real_data: torch.Tensor,
        synthetic_data: torch.Tensor
    ) -> float:
        """Compute fidelity metric.
        
        Args:
            real_data: Real data samples
            synthetic_data: Synthetic data samples
            
        Returns:
            Metric value
        """
        pass


class FIDMetric(FidelityMetric):
    """Fréchet Inception Distance (FID) metric using torchmetrics."""
    
    def __init__(
        self,
        device: Optional[torch.device] = None
    ):
        """Initialize FID metric.
        
        Args:
            device: Device to use for computation
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fid = FrechetInceptionDistance(normalize=True).to(self.device)
    
    def compute(
        self,
        real_data: torch.Tensor,
        synthetic_data: torch.Tensor
    ) -> float:
        """Compute FID between real and synthetic data.
        
        Args:
            real_data: Real data samples
            synthetic_data: Synthetic data samples
            
        Returns:
            FID value
        """
        # Preprocess data
        real_data = hook_preprocess(real_data).to(self.device)
        synthetic_data = hook_preprocess(synthetic_data).to(self.device)
        
        # Reset metric state
        self.fid.reset()
        
        # Update with real and fake images
        self.fid.update(real_data, real=True)
        self.fid.update(synthetic_data, real=False)
        
        # Compute FID
        fid_value = self.fid.compute()
        
        return float(fid_value)


class LPIPSMetric(FidelityMetric):
    """Learned Perceptual Image Patch Similarity (LPIPS) metric using torchmetrics."""
    
    def __init__(
        self,
        device: Optional[torch.device] = None
    ):
        """Initialize LPIPS metric.
        
        Args:
            device: Device to use for computation
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(self.device)
    
    def compute(
        self,
        real_data: torch.Tensor,
        synthetic_data: torch.Tensor
    ) -> float:
        """Compute LPIPS between real and synthetic data.
        
        Args:
            real_data: Real data samples
            synthetic_data: Synthetic data samples
            
        Returns:
            LPIPS value
        """
        # Preprocess data
        real_data = hook_preprocess(real_data).to(self.device)
        synthetic_data = hook_preprocess(synthetic_data).to(self.device)
        
        # Reset metric state
        self.lpips.reset()
        
        # Ensure same number of samples
        min_samples = min(len(real_data), len(synthetic_data))
        real_data = real_data[:min_samples]
        synthetic_data = synthetic_data[:min_samples]
        
        # Update LPIPS with paired samples
        self.lpips.update(real_data, synthetic_data)
        
        # Compute LPIPS
        lpips_value = self.lpips.compute()
        
        return float(lpips_value)


class PSNRMetric(FidelityMetric):
    """Peak Signal-to-Noise Ratio (PSNR) metric using torchmetrics."""
    
    def __init__(
        self,
        max_value: float = 1.0,
        device: Optional[torch.device] = None
    ):
        """Initialize PSNR metric.
        
        Args:
            max_value: Maximum value in the images
            device: Device to use for computation
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.psnr = PeakSignalNoiseRatio(data_range=max_value).to(self.device)
    
    def compute(
        self,
        real_data: torch.Tensor,
        synthetic_data: torch.Tensor
    ) -> float:
        """Compute PSNR between real and synthetic data.
        
        Args:
            real_data: Real data samples
            synthetic_data: Synthetic data samples
            
        Returns:
            PSNR value
        """
        # Move to device
        real_data = real_data.to(self.device)
        synthetic_data = synthetic_data.to(self.device)
        
        # Reset metric state
        self.psnr.reset()
        
        # Ensure same number of samples
        min_samples = min(len(real_data), len(synthetic_data))
        real_data = real_data[:min_samples]
        synthetic_data = synthetic_data[:min_samples]
        
        # Update PSNR with paired samples
        self.psnr.update(real_data, synthetic_data)
        
        # Compute PSNR
        psnr_value = self.psnr.compute()
        
        return float(psnr_value)


class ISMetric(FidelityMetric):
    """Inception Score (IS) metric using torchmetrics."""
    
    def __init__(
        self,
        device: Optional[torch.device] = None
    ):
        """Initialize IS metric.
        
        Args:
            device: Device to use for computation
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.inception_score = InceptionScore(normalize=True).to(self.device)
    
    def compute(
        self,
        real_data: torch.Tensor,
        synthetic_data: torch.Tensor
    ) -> float:
        """Compute IS for synthetic data.
        
        Args:
            real_data: Real data samples (not used for IS)
            synthetic_data: Synthetic data samples
            
        Returns:
            IS value
        """
        # Preprocess synthetic data
        synthetic_data = hook_preprocess(synthetic_data).to(self.device)
        
        # Reset metric state
        self.inception_score.reset()
        
        # Update with synthetic images
        self.inception_score.update(synthetic_data)
        
        # Compute IS (returns mean and std, we take the mean)
        is_mean, _ = self.inception_score.compute()
        
        return float(is_mean)


class SSIMMetric(FidelityMetric):
    """Structural Similarity Index Measure (SSIM) metric using torchmetrics."""
    
    def __init__(
        self,
        device: Optional[torch.device] = None
    ):
        """Initialize SSIM metric.
        
        Args:
            device: Device to use for computation
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
    
    def compute(
        self,
        real_data: torch.Tensor,
        synthetic_data: torch.Tensor
    ) -> float:
        """Compute SSIM between real and synthetic data.
        
        Args:
            real_data: Real data samples
            synthetic_data: Synthetic data samples
            
        Returns:
            SSIM value
        """
        # Move to device (no preprocessing needed for SSIM)
        real_data = real_data.to(self.device)
        synthetic_data = synthetic_data.to(self.device)
        
        # Reset metric state
        self.ssim.reset()
        
        # Ensure same number of samples
        min_samples = min(len(real_data), len(synthetic_data))
        real_data = real_data[:min_samples]
        synthetic_data = synthetic_data[:min_samples]
        
        # Update SSIM with paired samples
        self.ssim.update(real_data, synthetic_data)
        
        # Compute SSIM
        ssim_value = self.ssim.compute()
        
        return float(ssim_value)


class FidelityPipeline:
    """Pipeline for computing fidelity metrics."""
    
    def __init__(self):
        """Initialize fidelity pipeline."""
        self.metrics = {
            'fid': FIDMetric(),
            'lpips': LPIPSMetric(),
            'psnr': PSNRMetric(),
            'is': ISMetric(),
            'ssim': SSIMMetric()
        }
    
    def compute_metrics(
        self,
        real_data: torch.Tensor,
        synthetic_data: torch.Tensor,
        selected_metrics: Set[str]
    ) -> Dict[str, float]:
        """Compute selected fidelity metrics.
        
        Args:
            real_data: Real data samples
            synthetic_data: Synthetic data samples
            selected_metrics: Set of metric names to compute
            
        Returns:
            Dictionary of metric values
        """
        # Preprocess data
        real_data = hook_preprocess(real_data)
        synthetic_data = hook_preprocess(synthetic_data)
        
        # Compute selected metrics
        results = {}
        for metric_name in selected_metrics:
            if metric_name in self.metrics:
                results[metric_name] = self.metrics[metric_name].compute(
                    real_data,
                    synthetic_data
                )
        
        return results 
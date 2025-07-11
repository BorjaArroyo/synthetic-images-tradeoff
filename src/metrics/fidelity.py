"""Module for computing fidelity metrics for image generation."""
from typing import Dict
import torch
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.inception import InceptionScore

def hook_preprocess(image: torch.Tensor) -> torch.Tensor:
    """Preprocess image for fidelity metrics with proper normalization.
    
    Ensures images are in [0, 1] range and have proper dimensions for metrics.
    """
    # Ensure values are in [0, 1] range first
    image = torch.clamp(image, 0.0, 1.0)
    
    # Convert grayscale to RGB if needed
    if image.shape[1] == 1:
        image = image.repeat(1, 3, 1, 1)
    
    # Increase image size if needed for Inception-based metrics
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

def compute_fidelity_metrics(
    real_loader: DataLoader,
    fake_loader: DataLoader,
    metrics: set[str],
    device: torch.device
) -> Dict[str, float]:
    """
    Compute fidelity metrics between real and generated images.
    
    Args:
        real_loader: DataLoader containing real images in [0, 1] range
        fake_loader: DataLoader containing generated images in [0, 1] range
        metrics: Set of metrics to compute ('FID', 'LPIPS')
        device: Device to perform computations on
    
    Returns:
        Dictionary with metric names as keys and computed values as values
    
    Note:
        All images are expected to be in [0, 1] range.
        Normalization is handled by the metrics themselves.
    """
    results = {}
    
    # Compute FID if requested
    if 'FID' in metrics:
        fid = FrechetInceptionDistance(normalize=True).to(device)
        # Update with real images
        for batch in real_loader:
            real_images = batch[0] if isinstance(batch, (tuple, list)) else batch
            real_images = hook_preprocess(real_images)
            fid.update(real_images.to(device), real=True)
        # Update with fake images
        for batch in fake_loader:
            fake_images = batch[0] if isinstance(batch, (tuple, list)) else batch
            fake_images = hook_preprocess(fake_images)
            fid.update(fake_images.to(device), real=False)
        results['FID'] = fid.compute().item()

    # Compute LPIPS if requested
    if 'LPIPS' in metrics:
        lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)
        # Process paired samples
        for real_batch, fake_batch in zip(real_loader, fake_loader):
            real_images = real_batch[0] if isinstance(real_batch, (tuple, list)) else real_batch
            real_images = hook_preprocess(real_images)
            fake_images = fake_batch[0] if isinstance(fake_batch, (tuple, list)) else fake_batch
            fake_images = hook_preprocess(fake_images)
            lpips.update(real_images.to(device), fake_images.to(device))
        results['LPIPS'] = lpips.compute().item()

    if "PSNR" in metrics:
        psnr = PeakSignalNoiseRatio().to(device)
        # Process paired samples
        for real_batch, fake_batch in zip(real_loader, fake_loader):
            real_images = real_batch[0] if isinstance(real_batch, (tuple, list)) else real_batch
            real_images = hook_preprocess(real_images)
            fake_images = fake_batch[0] if isinstance(fake_batch, (tuple, list)) else fake_batch
            fake_images = hook_preprocess(fake_images)
            psnr.update(real_images.to(device), fake_images.to(device))
        results['PSNR'] = psnr.compute().item()

    if "IS" in metrics:
        # Inception score
        inception_score = InceptionScore(normalize=True).to(device)
        for batch in fake_loader:
            fake_images = batch[0] if isinstance(batch, (tuple, list)) else batch
            fake_images = hook_preprocess(fake_images)
            inception_score.update(fake_images.to(device))
        results['IS'] = inception_score.compute()[0].item()

    if "SSIM" in metrics:
        # Mean Structural Similarity Index
        from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
        mssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        for real_batch, fake_batch in zip(real_loader, fake_loader):
            real_images = real_batch[0] if isinstance(real_batch, (tuple, list)) else real_batch
            real_images = hook_preprocess(real_images)
            fake_images = fake_batch[0] if isinstance(fake_batch, (tuple, list)) else fake_batch
            fake_images = hook_preprocess(fake_images)
            mssim.update(real_images.to(device), fake_images.to(device))
        results['SSIM'] = mssim.compute().item()
    
    return results 
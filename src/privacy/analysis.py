"""Privacy analysis utilities including ECDF curves."""
import torch
import numpy as np
from scipy.stats import ecdf
from typing import Tuple
import src.visualization as viz


def ecdf_distance_curves(
    train_data: torch.Tensor,
    ref_data: torch.Tensor,
    synthetic_data: torch.Tensor,
) -> Tuple[ecdf, ecdf]:
    """Compute ECDF curves for privacy analysis.
    
    This function computes empirical cumulative distribution functions (ECDFs) 
    of minimum distances between reference/synthetic data and training data.
    This serves as a proxy for privacy analysis.
    
    For privacy analysis context:
    - ref_data: Real holdout data never seen during training (should have higher distances) 
    - synthetic_data: Synthetic data from full/main model being evaluated (may have lower distances if overfitting)
    - Lower distances to training data suggest potential privacy leakage/memorization
    
    Args:
        train_data: Training data tensor (4D: batch, channels, height, width)
        ref_data: Real holdout data tensor (never seen during training)
        synthetic_data: Synthetic data tensor from full/main model being evaluated
        
    Returns:
        Tuple of (ecdf_ref, ecdf_synthetic) representing the ECDF functions
    """
    assert train_data.dim() == ref_data.dim() == synthetic_data.dim() == 4, \
        f"All datasets must be 4D tensors, received {train_data.dim()} {ref_data.dim()} {synthetic_data.dim()}"
    assert train_data.shape[1:] == ref_data.shape[1:] == synthetic_data.shape[1:], \
        f"All datasets must have the same shape, received {train_data.shape[1:]} {ref_data.shape[1:]} {synthetic_data.shape[1:]}"
    
    # Flatten the data for distance computation
    ref_data_flat = ref_data.flatten(1)
    synthetic_data_flat = synthetic_data.flatten(1)
    train_data_flat = train_data.flatten(1)
    
    # Compute minimum distances to training data
    min_distance_ref_train = torch.cdist(ref_data_flat, train_data_flat).min(dim=1).values
    min_distance_synthetic_train = torch.cdist(synthetic_data_flat, train_data_flat).min(dim=1).values
    
    # Compute the ECDF for both sets of distances
    ecdf_ref = ecdf(min_distance_ref_train.numpy())
    ecdf_synthetic = ecdf(min_distance_synthetic_train.numpy())
    
    return ecdf_ref, ecdf_synthetic


# plot_ecdf_comparison removed - use viz.plot_ecdf_comparison directly


def compute_privacy_distance_metrics(
    train_data: torch.Tensor,
    ref_data: torch.Tensor,
    synthetic_data: torch.Tensor,
) -> dict:
    """Compute privacy metrics based on distance analysis.
    
    Args:
        train_data: Training data tensor
        ref_data: Reference/baseline data tensor (e.g., ablated model output)
        synthetic_data: Target synthetic data tensor (e.g., victim model output)
        
    Returns:
        Dictionary containing privacy metrics
    """
    # Compute ECDF curves
    ecdf_ref, ecdf_synthetic = ecdf_distance_curves(train_data, ref_data, synthetic_data)
    
    # Get the actual distance data
    ref_distances = ecdf_ref.cdf.quantiles
    synthetic_distances = ecdf_synthetic.cdf.quantiles
    
    metrics = {
        'ref_mean_distance': float(np.mean(ref_distances)),
        'synthetic_mean_distance': float(np.mean(synthetic_distances)),
        'ref_median_distance': float(np.median(ref_distances)),
        'synthetic_median_distance': float(np.median(synthetic_distances)),
        'ref_min_distance': float(np.min(ref_distances)),
        'synthetic_min_distance': float(np.min(synthetic_distances)),
        'distance_ratio': float(np.mean(synthetic_distances) / np.mean(ref_distances)),
        'ks_statistic': compute_ks_distance(ecdf_ref, ecdf_synthetic),
        'directional_ks_statistic': compute_directional_ks_distance(ecdf_ref, ecdf_synthetic),
        'area_between_curves': compute_area_between_curves(ecdf_ref, ecdf_synthetic)
    }
    
    return metrics


def compute_ks_distance(ecdf1: ecdf, ecdf2: ecdf) -> float:
    """Compute Kolmogorov-Smirnov distance between two ECDFs.
    
    Args:
        ecdf1: First ECDF
        ecdf2: Second ECDF
        
    Returns:
        KS distance
    """
    # Get the quantiles from both ECDFs
    data1 = ecdf1.cdf.quantiles
    data2 = ecdf2.cdf.quantiles
    
    # Get common x values
    x_min = max(data1.min(), data2.min())
    x_max = min(data1.max(), data2.max())
    x_common = np.linspace(x_min, x_max, 1000)
    
    # Evaluate ECDFs at common points
    y1 = ecdf1.cdf.evaluate(x_common)
    y2 = ecdf2.cdf.evaluate(x_common)
    
    # Compute KS distance
    ks_distance = np.max(np.abs(y1 - y2))
    
    return float(ks_distance)


def compute_directional_ks_distance(ecdf_holdout: ecdf, ecdf_synthetic: ecdf) -> float:
    """Compute directional KS distance for privacy analysis.
    
    This function computes a directional variant of the KS statistic that only
    penalizes when the synthetic ECDF is above the holdout ECDF (indicating
    synthetic data has smaller minimum distances to training data, suggesting
    potential privacy leakage).
    
    Args:
        ecdf_holdout: ECDF for holdout/reference data distances to training
        ecdf_synthetic: ECDF for synthetic data distances to training
        
    Returns:
        Directional KS distance [0, 1]
        - 0: No privacy concern (synthetic always has larger/equal distances)
        - Higher values: Increasing privacy leakage concern
    """
    # Get the quantiles from both ECDFs
    holdout_data = ecdf_holdout.cdf.quantiles
    synthetic_data = ecdf_synthetic.cdf.quantiles
    
    # Get common x values for evaluation
    x_min = max(holdout_data.min(), synthetic_data.min())
    x_max = min(holdout_data.max(), synthetic_data.max())
    
    # Handle edge case where ranges don't overlap
    if x_max <= x_min:
        return 1.0  # Complete separation indicates severe privacy issue
    
    x_common = np.linspace(x_min, x_max, 1000)
    
    # Evaluate ECDFs at common points
    y_holdout = ecdf_holdout.cdf.evaluate(x_common)
    y_synthetic = ecdf_synthetic.cdf.evaluate(x_common)
    
    # Compute directional difference
    # Positive when synthetic ECDF > holdout ECDF (privacy concern)
    # Negative when synthetic ECDF < holdout ECDF (good for privacy)
    directional_diff = y_synthetic - y_holdout
    
    # Only consider positive differences (privacy violations)
    privacy_violations = np.maximum(directional_diff, 0)
    
    # Return maximum positive deviation
    directional_ks = np.max(privacy_violations)
    
    return float(directional_ks)


def compute_area_between_curves(ecdf1: ecdf, ecdf2: ecdf) -> float:
    """Compute normalized area between two ECDF curves.
    
    This provides a holistic measure of difference between distributions,
    capturing the integrated difference across the entire range.
    Normalized to [0, 1] for easy comparison with KS statistic.
    
    Args:
        ecdf1: First ECDF (reference - holdout data)
        ecdf2: Second ECDF (synthetic - full model data)
        
    Returns:
        Normalized area between curves [0, 1]
        - 0: Identical distributions (perfect privacy)
        - 1: Maximum possible difference (severe privacy leakage)
    """
    # Get the quantiles from both ECDFs
    data1 = ecdf1.cdf.quantiles
    data2 = ecdf2.cdf.quantiles
    
    # Use same x-range approach as KS distance for consistency
    x_min = max(data1.min(), data2.min())
    x_max = min(data1.max(), data2.max())
    
    # Handle edge case where ranges don't overlap
    if x_max <= x_min:
        return 1.0  # Complete separation = maximum privacy leakage
    
    x_common = np.linspace(x_min, x_max, 1000)
    
    # Evaluate ECDFs at common points (both in [0, 1])
    y1 = ecdf1.cdf.evaluate(x_common)
    y2 = ecdf2.cdf.evaluate(x_common)
    
    # Compute area between curves using trapezoidal integration
    area_between = np.trapz(np.abs(y1 - y2), x_common)
    
    # Normalize by maximum possible area
    # Max area = (x_max - x_min) * 1.0 (width × max_height_difference)
    max_possible_area = x_max - x_min
    normalized_area = area_between / max_possible_area if max_possible_area > 0 else 0.0
    
    # Ensure result is in [0, 1]
    return float(min(max(normalized_area, 0.0), 1.0))


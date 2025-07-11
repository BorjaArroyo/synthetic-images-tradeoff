"""
Privacy Visualization Functions

Provides plotting functions for privacy analysis including ECDF curves,
projection comparisons, and privacy evaluation visualizations.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ecdf
from typing import Tuple, Optional, Dict, Any
import os
import mlflow
from .config import get_color_palette
from .common import save_figure


def plot_ecdf_comparison(
    ecdf_ref: ecdf,
    ecdf_synthetic: ecdf,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    colors: Optional[Dict[str, str]] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    Plot ECDF curves for privacy analysis comparison.
    
    Args:
        ecdf_ref: ECDF for reference data distances
        ecdf_synthetic: ECDF for synthetic data distances
        save_path: Optional path to save the plot
        title: Optional title for the plot
        colors: Optional color scheme override
        figsize: Figure size tuple
    """
    if colors is None:
        colors = get_color_palette('privacy')
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Get data from ECDF objects
    ref_data = ecdf_ref.cdf.quantiles
    synthetic_data = ecdf_synthetic.cdf.quantiles
    
    # Create deterministic x-range based on reference data
    # Use 1.5x the max distance from reference data for consistency
    x_min = 0.0  # Start from zero for cleaner plots
    
    # Use 1.5x max of reference data
    x_max = ref_data.max() * 1.5
    
    x_common = np.linspace(x_min, x_max, 1000)
    
    print(f"📏 ECDF plot range: [0.0, {x_max:.4f}]")
    
    # Plot ECDFs with consistent colors using common x-range
    ax.plot(x_common, ecdf_ref.cdf.evaluate(x_common), 
            label='Real Holdout Data', 
            color=colors['reference'],
            linewidth=plt.rcParams['lines.linewidth'])
    ax.plot(x_common, ecdf_synthetic.cdf.evaluate(x_common), 
            label='Synthetic Data',
            color=colors['synthetic'], 
            linewidth=plt.rcParams['lines.linewidth'])
    
    ax.set_xlabel('Minimum Distance to Training Data')
    ax.set_ylabel('Cumulative Probability')
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=plt.rcParams['grid.alpha'])
    
    save_figure(fig, save_path)


def plot_hexbin_difference(
    numerator: np.ndarray, 
    denominator: np.ndarray, 
    projected_query: Optional[np.ndarray] = None, 
    bins: int = 25,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    colors: Optional[Dict[str, str]] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    Plot hexbin difference between numerator and denominator distributions.
    
    Args:
        numerator: Numerator distribution points (victim model data)
        denominator: Denominator distribution points (ablated model data)  
        projected_query: Query points to highlight
        bins: Number of bins for histogram
        save_path: Optional path to save the plot
        title: Optional title for the plot
        colors: Optional color scheme override
        figsize: Figure size tuple
    """
    if colors is None:
        colors = get_color_palette('privacy')
    
    # Ensure we have 2D data for plotting
    if numerator.shape[1] > 2:
        print(f"Warning: Data has {numerator.shape[1]} dimensions, using first 2 for plotting")
        numerator = numerator[:, :2]
        denominator = denominator[:, :2]
        if projected_query is not None:
            projected_query = projected_query[:, :2]
    
    # Combine both for a shared bin grid
    all_points = np.vstack([numerator, denominator])
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()

    # Compute 2D histograms
    numerator_hist, xedges, yedges = np.histogram2d(
        numerator[:, 0], numerator[:, 1],
        bins=bins, range=[[x_min, x_max], [y_min, y_max]]
    )
    denominator_hist, _, _ = np.histogram2d(
        denominator[:, 0], denominator[:, 1],
        bins=[xedges, yedges]
    )

    # Difference
    diff = numerator_hist - denominator_hist

    # Plot difference
    fig, ax = plt.subplots(figsize=figsize)
    vmax = np.abs(diff).max()
    vmin = -vmax

    mesh = ax.pcolormesh(
        xedges, yedges, diff.T,
        cmap='coolwarm', shading='auto',
        vmin=vmin, vmax=vmax
    )
    cbar = plt.colorbar(mesh, ax=ax)
    cbar.set_label("Victim - Ablated", rotation=270, labelpad=25)
    # cbar.set_label("Victim - Ablated", loc="right")

    if projected_query is not None and len(projected_query) > 0:
        ax.scatter(projected_query[:, 0], projected_query[:, 1],
                   c=colors['query'], marker='x', label='Query', 
                   s=120, linewidth=3)
        ax.legend()

    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    if title:

        ax.set_title(title)
    
    save_figure(fig, save_path)


def plot_projections_comparison(
    projected_numerator: np.ndarray, 
    projected_denominator: np.ndarray, 
    projected_query: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    colors: Optional[Dict[str, str]] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    Plot comparison of projections between numerator and denominator.
    
    Args:
        projected_numerator: Projected numerator samples (victim)
        projected_denominator: Projected denominator samples (ablated)
        projected_query: Projected query samples
        save_path: Optional path to save the plot
        title: Optional title for the plot
        colors: Optional color scheme override
        figsize: Figure size tuple
    """
    if colors is None:
        colors = get_color_palette('privacy')
    
    # Ensure we have 2D data for plotting
    if projected_numerator.shape[1] > 2:
        print(f"Warning: Data has {projected_numerator.shape[1]} dimensions, using first 2 for plotting")
        projected_numerator = projected_numerator[:, :2]
        projected_denominator = projected_denominator[:, :2]
        if projected_query is not None:
            projected_query = projected_query[:, :2]
    
    # Subsample for better visualization if too many points
    def subsample(data: np.ndarray, max_points: int = 2000) -> np.ndarray:
        if len(data) > max_points:
            indices = np.random.choice(len(data), max_points, replace=False)
            return data[indices]
        return data
    
    projected_numerator = subsample(projected_numerator)
    projected_denominator = subsample(projected_denominator)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(projected_numerator[:, 0], projected_numerator[:, 1],
               c=colors['victim'], alpha=0.6, label='Victim', 
               edgecolors='none', s=30)
    ax.scatter(projected_denominator[:, 0], projected_denominator[:, 1],
               c=colors['ablated'], alpha=0.6, label='Ablated', 
               edgecolors='none', s=30)

    if projected_query is not None and len(projected_query) > 0:
        ax.scatter(projected_query[:, 0], projected_query[:, 1],
                   c=colors['query'], marker='x', label='Query', 
                   s=150, linewidth=4)

    ax.legend()
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    if title:

        ax.set_title(title)
    ax.grid(True, alpha=plt.rcParams['grid.alpha'])
    
    save_figure(fig, save_path)


def plot_projections_by_class(
    data: np.ndarray, 
    labels: np.ndarray, 
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    Plot projections colored by class labels.
    
    Args:
        data: 2D projection data
        labels: Class labels
        save_path: Optional path to save the plot
        title: Plot title
        figsize: Figure size tuple
    """
    # Ensure we have 2D data for plotting
    if data.shape[1] > 2:
        print(f"Warning: Data has {data.shape[1]} dimensions, using first 2 for plotting")
        data = data[:, :2]
    
    fig, ax = plt.subplots(figsize=figsize)
    scatter = ax.scatter(data[:, 0], data[:, 1],
                         c=labels, cmap='tab10', alpha=0.7, 
                         edgecolors='none', s=30)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Class Label')
    
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    if title:

        ax.set_title(title)
    ax.grid(True, alpha=plt.rcParams['grid.alpha'])
    
    save_figure(fig, save_path)


def create_privacy_visualizations(
    query_data: torch.Tensor,
    synthetic_data: torch.Tensor,
    real_data: torch.Tensor,
    save_dir: str = "figs",
    log_to_mlflow: bool = True,
    prefix: str = "privacy"
) -> None:
    """
    Create comprehensive privacy visualization plots.
    
    Args:
        query_data: Query samples (projected if using reducer)
        synthetic_data: Synthetic data samples (projected if using reducer)
        real_data: Real data samples (projected if using reducer)
        save_dir: Directory to save plots
        log_to_mlflow: Whether to log plots to MLflow
        prefix: Filename prefix
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert to numpy if needed
    if isinstance(query_data, torch.Tensor):
        query_data = query_data.cpu().numpy()
    if isinstance(synthetic_data, torch.Tensor):
        synthetic_data = synthetic_data.cpu().numpy()
    if isinstance(real_data, torch.Tensor):
        real_data = real_data.cpu().numpy()
    
    colors = get_color_palette('privacy')
    
    # Create hexbin difference plot
    hexbin_path = os.path.join(save_dir, f"{prefix}_hexbin_difference.png")
    plot_hexbin_difference(
        synthetic_data, real_data, query_data,
        save_path=hexbin_path,
# No title for academic article format
    )
    
    # Create projection comparison plot
    comparison_path = os.path.join(save_dir, f"{prefix}_projection_comparison.png")
    plot_projections_comparison(
        synthetic_data, real_data, query_data,
        save_path=comparison_path,
# No title for academic article format
    )
    
    # Log to MLflow if requested
    if log_to_mlflow:
        try:
            mlflow.log_artifact(hexbin_path)
            mlflow.log_artifact(comparison_path)
        except Exception as e:
            print(f"Warning: MLflow logging failed: {e}")


def plot_privacy_simple(
    query_data: torch.Tensor,
    synthetic_data: torch.Tensor,
    real_data: torch.Tensor,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    colors: Optional[Dict[str, str]] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    Simple privacy visualization with scatter plot.
    
    Args:
        query_data: Query data samples
        synthetic_data: Synthetic data samples
        real_data: Real data samples
        save_path: Optional path to save the plot
        title: Optional title for the plot
        colors: Optional color scheme override
        figsize: Figure size tuple
    """
    if colors is None:
        colors = get_color_palette('privacy')
    
    # Convert tensors to numpy arrays
    query_np = query_data.cpu().numpy()
    synthetic_np = synthetic_data.cpu().numpy()
    real_np = real_data.cpu().numpy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot data points
    ax.scatter(
        query_np[:, 0], query_np[:, 1],
        c=colors['query'], label='Query',
        alpha=0.6, s=30
    )
    ax.scatter(
        synthetic_np[:, 0], synthetic_np[:, 1],
        c=colors['synthetic'], label='Synthetic',
        alpha=0.6, s=30
    )
    ax.scatter(
        real_np[:, 0], real_np[:, 1],
        c=colors['reference'], label='Real',
        alpha=0.6, s=30
    )
    
    # Add labels and formatting
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    if title:

        ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=plt.rcParams['grid.alpha'])
    
    save_figure(fig, save_path)


def create_victim_model_query_grid(
    victim_model: torch.nn.Module,
    query_class: int,
    ablated_model: Optional[torch.nn.Module] = None,
    samples_per_row: int = 8,
    num_rows: int = 3,
    device: torch.device = torch.device('cpu'),
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None
) -> torch.Tensor:
    """
    Create a grid visualization of synthetic images generated by the victim model
    for the query class, showing the impact of a single sample on the sampling process.
    
    Args:
        victim_model: Trained victim model (includes query sample)
        query_class: The class index for the query sample
        ablated_model: Optional ablated model (without query sample) for comparison
        samples_per_row: Number of samples per row in the grid
        num_rows: Number of rows to generate
        device: Device to run inference on
        save_path: Optional path to save the plot
        title: Optional title for the plot
        figsize: Figure size tuple
        
    Returns:
        Generated image grid tensor
    """
    from ..models.sampler import sample
    from torchvision.utils import make_grid
    
    colors = get_color_palette('privacy')
    
    victim_model.eval()
    
    total_samples = samples_per_row * num_rows
    
    # Generate labels for the query class
    query_labels = torch.full((total_samples,), query_class, dtype=torch.long, device=device)
    
    with torch.no_grad():
        # Generate samples from victim model
        victim_samples = sample(victim_model, query_labels, device)
        
        # If ablated model is provided, generate comparison samples
        comparison_samples = None
        if ablated_model is not None:
            ablated_model.eval()
            comparison_samples = sample(ablated_model, query_labels, device)
    
    if figsize is None:
        # Adjust figure size based on whether we have comparison
        height = 8 if ablated_model is None else 12
        figsize = (12, height)
    
    # Create the visualization
    if ablated_model is not None:
        # Create comparison grid: victim model on top, ablated model on bottom
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # Victim model grid
        victim_grid = make_grid(
            victim_samples,
            nrow=samples_per_row,
            normalize=True,
            scale_each=True,
            pad_value=1.0
        )
        
        ax1.imshow(victim_grid.permute(1, 2, 0).cpu().numpy())
        ax1.axis('off')
        ax1.text(0.02, 0.98, f'Victim Model (Class {query_class})', 
                transform=ax1.transAxes, fontsize=plt.rcParams['font.size'],
                fontweight='bold', ha='left', va='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['victim'], alpha=0.7))
        
        # Ablated model grid
        ablated_grid = make_grid(
            comparison_samples,
            nrow=samples_per_row,
            normalize=True,
            scale_each=True,
            pad_value=1.0
        )
        
        ax2.imshow(ablated_grid.permute(1, 2, 0).cpu().numpy())
        ax2.axis('off')
        ax2.text(0.02, 0.98, f'Ablated Model (Class {query_class})', 
                transform=ax2.transAxes, fontsize=plt.rcParams['font.size'],
                fontweight='bold', ha='left', va='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['ablated'], alpha=0.7))
        
        # Add subtitle explaining the purpose
        fig.text(0.5, 0.02, 'Impact of Query Sample on Synthetic Generation', 
                ha='center', va='bottom', fontsize=plt.rcParams['font.size']-2,
                style='italic')
        
        result_grid = victim_grid  # Return victim grid as primary result
        
    else:
        # Single grid for victim model only
        fig, ax = plt.subplots(figsize=figsize)
        
        victim_grid = make_grid(
            victim_samples,
            nrow=samples_per_row,
            normalize=True,
            scale_each=True,
            pad_value=1.0
        )
        
        ax.imshow(victim_grid.permute(1, 2, 0).cpu().numpy())
        ax.axis('off')
        
        # Add label for the query class
        ax.text(0.02, 0.98, f'Victim Model - Query Class {query_class}', 
               transform=ax.transAxes, fontsize=plt.rcParams['font.size'],
               fontweight='bold', ha='left', va='top',
               bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['victim'], alpha=0.7))
        
        result_grid = victim_grid
    
    if title:
        fig.suptitle(title)
    
    plt.tight_layout()
    save_figure(fig, save_path)
    
    return result_grid


def create_privacy_impact_visualization(
    victim_model: torch.nn.Module,
    ablated_model: torch.nn.Module,
    query_class: int,
    all_classes: bool = False,
    samples_per_class: int = 6,
    device: torch.device = torch.device('cpu'),
    save_path: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None
) -> None:
    """
    Create a comprehensive visualization showing the impact of the query sample
    across different classes or focused on the query class.
    
    Args:
        victim_model: Trained victim model (includes query sample)
        ablated_model: Trained ablated model (excludes query sample)
        query_class: The class index for the query sample
        all_classes: If True, show impact across all classes; if False, focus on query class
        samples_per_class: Number of samples to generate per class
        device: Device to run inference on
        save_path: Optional path to save the plot
        figsize: Figure size tuple
    """
    from ..models.sampler import sample
    from torchvision.utils import make_grid
    
    colors = get_color_palette('privacy')
    
    victim_model.eval()
    ablated_model.eval()
    
    if all_classes:
        # Determine number of classes from the models
        # This is a simple heuristic - could be passed as parameter
        num_classes = 10  # Adjust based on your dataset
        classes_to_show = list(range(min(num_classes, 6)))  # Show max 6 classes
    else:
        classes_to_show = [query_class]
    
    if figsize is None:
        width = 4 * samples_per_class
        height = 4 * len(classes_to_show)
        figsize = (width, height)
    
    fig, axes = plt.subplots(len(classes_to_show), 2, figsize=figsize)
    
    # Handle single class case
    if len(classes_to_show) == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for i, class_idx in enumerate(classes_to_show):
            # Generate labels for this class
            labels = torch.full((samples_per_class,), class_idx, dtype=torch.long, device=device)
            
            # Generate samples from both models
            victim_samples = sample(victim_model, labels, device)
            ablated_samples = sample(ablated_model, labels, device)
            
            # Create grids for this class
            victim_grid = make_grid(
                victim_samples, nrow=samples_per_class, normalize=True, 
                scale_each=True, pad_value=1.0
            )
            ablated_grid = make_grid(
                ablated_samples, nrow=samples_per_class, normalize=True, 
                scale_each=True, pad_value=1.0
            )
            
            # Plot victim model samples
            axes[i, 0].imshow(victim_grid.permute(1, 2, 0).cpu().numpy())
            axes[i, 0].axis('off')
            
            # Highlight query class with different border
            border_color = colors['query'] if class_idx == query_class else colors['victim']
            label_text = f'Victim - Class {class_idx}' + (' (Query)' if class_idx == query_class else '')
            
            axes[i, 0].text(0.02, 0.98, label_text,
                           transform=axes[i, 0].transAxes, fontsize=plt.rcParams['font.size']-2,
                           fontweight='bold', ha='left', va='top',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor=border_color, alpha=0.7))
            
            # Plot ablated model samples
            axes[i, 1].imshow(ablated_grid.permute(1, 2, 0).cpu().numpy())
            axes[i, 1].axis('off')
            axes[i, 1].text(0.02, 0.98, f'Ablated - Class {class_idx}',
                           transform=axes[i, 1].transAxes, fontsize=plt.rcParams['font.size']-2,
                           fontweight='bold', ha='left', va='top',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['ablated'], alpha=0.7))
    
    # Add column headers
    fig.text(0.25, 0.95, 'With Query Sample', ha='center', va='top', 
             fontsize=plt.rcParams['font.size'], fontweight='bold')
    fig.text(0.75, 0.95, 'Without Query Sample', ha='center', va='top', 
             fontsize=plt.rcParams['font.size'], fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    save_figure(fig, save_path) 
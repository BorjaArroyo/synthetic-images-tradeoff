"""
Utility Visualization Functions

Provides plotting functions for utility analysis including GradCAM visualizations
and synthetic image grids.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Dict, Any
from torchvision.utils import make_grid
from .config import get_color_palette
from .common import save_figure


def plot_gradcam_analysis(
    original_images: torch.Tensor,
    gradcam_overlays: torch.Tensor,
    labels: torch.Tensor,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    max_images: int = 8,
    figsize: Optional[Tuple[float, float]] = None
) -> None:
    """
    Plot GradCAM analysis with original and overlay images.
    
    Args:
        original_images: Original input images
        gradcam_overlays: GradCAM overlay images
        labels: Image labels
        save_path: Optional path to save the plot
        title: Optional title for the plot
        max_images: Maximum number of images to display
        figsize: Figure size tuple
    """
    n_images = min(len(original_images), max_images)
    
    if figsize is None:
        figsize = (4 * n_images, 8)
    
    fig, axes = plt.subplots(2, n_images, figsize=figsize)
    
    if n_images == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(n_images):
        # Original image
        if original_images[i].dim() == 3 and original_images[i].size(0) == 1:
            # Grayscale
            axes[0, i].imshow(original_images[i].squeeze().cpu().numpy(), cmap='gray')
        else:
            # RGB
            axes[0, i].imshow(original_images[i].permute(1, 2, 0).cpu().numpy())
        axes[0, i].axis('off')
        
        # GradCAM overlay
        if gradcam_overlays[i].dim() == 3:
            axes[1, i].imshow(gradcam_overlays[i].permute(1, 2, 0).cpu().numpy())
        else:
            axes[1, i].imshow(gradcam_overlays[i].cpu().numpy())
        axes[1, i].axis('off')
    
    if title:
        fig.suptitle(title)
    
    plt.tight_layout()
    save_figure(fig, save_path)


def plot_gradcam_comparison(
    synthetic_images: torch.Tensor,
    real_images: torch.Tensor,
    synthetic_gradcam: torch.Tensor,
    real_gradcam: torch.Tensor,
    labels: torch.Tensor,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    max_samples: int = 4,
    figsize: Optional[Tuple[float, float]] = None
) -> None:
    """
    Plot comparison between synthetic and real GradCAM visualizations.
    
    Args:
        synthetic_images: Synthetic input images
        real_images: Real input images
        synthetic_gradcam: GradCAM for synthetic images
        real_gradcam: GradCAM for real images
        labels: Image labels
        save_path: Optional path to save the plot
        title: Optional title for the plot
        max_samples: Maximum number of samples to display
        figsize: Figure size tuple
    """
    n_samples = min(len(synthetic_images), len(real_images), max_samples)
    
    if figsize is None:
        figsize = (4 * n_samples, 8)
    
    fig, axes = plt.subplots(2, n_samples, figsize=figsize)
    
    if n_samples == 1:
        axes = axes.reshape(2, 1)
    
    colors = get_color_palette('utility')
    
    for i in range(n_samples):
        # Synthetic GradCAM
        axes[0, i].imshow(synthetic_images[i].squeeze().cpu().numpy(), cmap='gray')
        axes[0, i].imshow(synthetic_gradcam[i].cpu().numpy(), alpha=0.4, cmap='jet')
        axes[0, i].axis('off')
        
        # Real GradCAM
        axes[1, i].imshow(real_images[i].squeeze().cpu().numpy(), cmap='gray')
        axes[1, i].imshow(real_gradcam[i].cpu().numpy(), alpha=0.4, cmap='jet')
        axes[1, i].axis('off')
    
    if title:
        fig.suptitle(title)
    
    plt.tight_layout()
    save_figure(fig, save_path)


def create_synthetic_image_grid(
    model: nn.Module,
    num_classes: int,
    samples_per_class: int = 8,
    device: torch.device = torch.device('cpu'),
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None
) -> torch.Tensor:
    """
    Create a grid of synthetic images from a generative model.
    
    Args:
        model: Trained generative model (VAE or GAN)
        num_classes: Number of classes
        samples_per_class: Number of samples per class
        device: Device to run inference on
        save_path: Optional path to save the plot
        title: Optional title for the plot
        figsize: Figure size tuple
        
    Returns:
        Generated image grid tensor
    """
    model.eval()
    
    all_samples = []
    
    with torch.no_grad():
        for class_idx in range(num_classes):
            # Create labels for this class
            labels = torch.full((samples_per_class,), class_idx, dtype=torch.long, device=device)
            
            # Generate samples
            if hasattr(model, 'sample'):
                # VAE-style sampling
                samples = model.sample(samples_per_class, labels, device)
            elif hasattr(model, 'generate'):
                # GAN-style generation
                noise = torch.randn(samples_per_class, model.latent_dim, device=device)
                samples = model.generate(noise, labels)
            else:
                raise ValueError("Model must have either 'sample' or 'generate' method")
            
            all_samples.append(samples)
    
    # Concatenate all samples
    all_samples = torch.cat(all_samples, dim=0)
    
    # Create grid
    grid = make_grid(
        all_samples,
        nrow=samples_per_class,
        normalize=True,
        scale_each=True,
        pad_value=1.0
    )
    
    if figsize is None:
        figsize = (12, 8)
    
    # Plot the grid
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(grid.permute(1, 2, 0).cpu().numpy())
    ax.axis('off')
    
    # Add class labels on the left side for each row
    for i in range(num_classes):
        # Calculate y position for each class row
        row_height = grid.shape[1] // num_classes
        y_pos = (i + 0.5) * row_height
        ax.text(-30, y_pos, f'Class {i}', 
               ha='right', va='center', 
               fontweight='bold', rotation=90)
    
    if title:

    
        ax.set_title(title)
    
    plt.tight_layout()
    save_figure(fig, save_path)
    
    return grid


def plot_utility_metrics(
    metrics: Dict[str, float],
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6)
) -> None:
    """
    Plot utility metrics as a bar chart.
    
    Args:
        metrics: Dictionary of metric names to values
        save_path: Optional path to save the plot
        title: Plot title
        figsize: Figure size tuple
    """
    colors = get_color_palette('utility')
    
    fig, ax = plt.subplots(figsize=figsize)
    
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    
    bars = ax.bar(metric_names, metric_values, 
                  color=[colors.get('accuracy', '#2A9D8F')] * len(metric_names))
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    ax.set_ylabel('Score')
    if title:

        ax.set_title(title)
    ax.set_ylim(0, max(metric_values) * 1.1)
    
    # Rotate x-axis labels if there are many metrics
    if len(metric_names) > 5:
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    save_figure(fig, save_path)


def plot_training_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    train_metrics: Optional[Dict[str, List[float]]] = None,
    val_metrics: Optional[Dict[str, List[float]]] = None,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 8)
) -> None:
    """
    Plot training and validation curves.
    
    Args:
        train_losses: Training losses over epochs
        val_losses: Validation losses over epochs
        train_metrics: Dictionary of training metrics over epochs
        val_metrics: Dictionary of validation metrics over epochs
        save_path: Optional path to save the plot
        title: Plot title
        figsize: Figure size tuple
    """
    colors = get_color_palette('utility')
    
    # Determine number of subplots needed
    n_plots = 1  # Always have loss plot
    if train_metrics:
        n_plots += len(train_metrics)
    
    # Calculate grid dimensions
    ncols = min(3, n_plots)
    nrows = (n_plots + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n_plots == 1:
        axes = [axes]
    elif nrows == 1:
        axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
    else:
        axes = axes.flatten()
    
    plot_idx = 0
    
    # Plot losses
    ax = axes[plot_idx]
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label='Training Loss', 
            color=colors['original'], linewidth=plt.rcParams['lines.linewidth'])
    
    if val_losses:
        ax.plot(epochs, val_losses, label='Validation Loss', 
                color=colors['synthetic'], linewidth=plt.rcParams['lines.linewidth'])
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=plt.rcParams['grid.alpha'])
    plot_idx += 1
    
    # Plot metrics
    if train_metrics:
        for metric_name, metric_values in train_metrics.items():
            if plot_idx < len(axes):
                ax = axes[plot_idx]
                ax.plot(epochs, metric_values, label=f'Train {metric_name}',
                        color=colors['original'], linewidth=plt.rcParams['lines.linewidth'])
                
                if val_metrics and metric_name in val_metrics:
                    ax.plot(epochs, val_metrics[metric_name], label=f'Val {metric_name}',
                            color=colors['synthetic'], linewidth=plt.rcParams['lines.linewidth'])
                
                ax.set_xlabel('Epoch')
                ax.set_ylabel(metric_name.title())
                ax.legend()
                ax.grid(True, alpha=plt.rcParams['grid.alpha'])
                plot_idx += 1
    
    # Hide unused subplots
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)
    
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    save_figure(fig, save_path) 
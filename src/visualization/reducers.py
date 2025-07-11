"""
Dimensionality reducer visualization functions.
Provides consistent, publication-ready visualizations for all reducer types.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Optional, List, Tuple, Any
import os

from .common import save_figure


def plot_reducer_projection(
    reduced_data: np.ndarray,
    labels: np.ndarray,
    reducer_name: str,
    dataset_name: str,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    show_legend: bool = True,
    query_class: Optional[int] = None
) -> plt.Figure:
    """Plot dimensionality reduction projection colored by class labels.
    
    Args:
        reduced_data: 2D projection data (n_samples, 2)
        labels: Class labels (n_samples,)
        reducer_name: Name of the reducer (e.g., 'UMAP', 't-SNE')
        dataset_name: Name of the dataset
        title: Custom title (if None, auto-generated)
        save_path: Path to save the plot
        figsize: Figure size
        show_legend: Whether to show legend
        query_class: Optional class to highlight with different marker ('x')
        
    Returns:
        Figure object
    """
    # Ensure we have 2D data
    if reduced_data.shape[1] > 2:
        reduced_data = reduced_data[:, :2]
        print(f"Warning: Using first 2 components of {reduced_data.shape[1]}D reduction")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get number of classes
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    
    # Create color map
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    
    # Plot each class separately for better legend control
    for i, class_label in enumerate(unique_labels):
        mask = labels == class_label
        if mask.sum() > 0:
            # Use different marker for query class
            if query_class is not None and class_label == query_class:
                ax.scatter(
                    reduced_data[mask, 0], 
                    reduced_data[mask, 1],
                    c=[colors[i]], 
                    label=f'Class {int(class_label)} (Query)',
                    alpha=0.8,
                    s=40,
                    marker='x',
                    linewidths=2,
                    edgecolors='black'
                )
            else:
                ax.scatter(
                    reduced_data[mask, 0], 
                    reduced_data[mask, 1],
                    c=[colors[i]], 
                    label=f'Class {int(class_label)}',
                    alpha=0.7,
                    s=20,
                    edgecolors='none'
                )
    
    # Formatting
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    
    if title:
        ax.set_title(title)
    
    if show_legend and n_classes <= 20:  # Only show legend if not too many classes
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def plot_reducer_comparison(
    projections: List[np.ndarray],
    labels: np.ndarray,
    reducer_names: List[str],
    dataset_name: str,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5)
) -> plt.Figure:
    """Compare multiple reducer projections side by side.
    
    Args:
        projections: List of 2D projection arrays
        labels: Class labels (same for all projections)
        reducer_names: Names of the reducers
        dataset_name: Name of the dataset
        title: Custom title
        save_path: Path to save the plot
        figsize: Figure size
        
    Returns:
        Figure object
    """
    n_reducers = len(projections)
    fig, axes = plt.subplots(1, n_reducers, figsize=figsize)
    
    if n_reducers == 1:
        axes = [axes]
    
    # Get color mapping
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    
    for i, (projection, reducer_name, ax) in enumerate(zip(projections, reducer_names, axes)):
        # Ensure 2D
        if projection.shape[1] > 2:
            projection = projection[:, :2]
        
        # Plot each class
        for j, class_label in enumerate(unique_labels):
            mask = labels == class_label
            if mask.sum() > 0:
                ax.scatter(
                    projection[mask, 0], 
                    projection[mask, 1],
                    c=[colors[j]], 
                    label=f'Class {int(class_label)}' if i == 0 else "",
                    alpha=0.7,
                    s=15,
                    edgecolors='none'
                )
        
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        # Removed subplot title for academic article format
        ax.grid(True, alpha=0.3)
    
    # Add legend to first subplot only
    if n_classes <= 10:
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    if title:
        fig.suptitle(title)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def plot_reducer_quality_metrics(
    reducer_names: List[str],
    metrics_dict: dict,
    dataset_name: str,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """Plot quality metrics for different reducers.
    
    Args:
        reducer_names: List of reducer names
        metrics_dict: Dictionary with metric_name -> [values] mapping
        dataset_name: Name of the dataset
        title: Custom title
        save_path: Path to save the plot
        figsize: Figure size
        
    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get metrics
    metric_names = list(metrics_dict.keys())
    n_metrics = len(metric_names)
    
    # Set up bar positions
    x = np.arange(len(reducer_names))
    width = 0.8 / n_metrics
    
    # Plot bars for each metric
    for i, metric_name in enumerate(metric_names):
        values = metrics_dict[metric_name]
        ax.bar(x + i * width, values, width, label=metric_name, alpha=0.8)
    
    # Formatting
    ax.set_xlabel('Reducer')
    ax.set_ylabel('Score')
    
    if title:
        ax.set_title(title)
    
    ax.set_xticks(x + width * (n_metrics - 1) / 2)
    ax.set_xticklabels(reducer_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def plot_reducer_embedding_evolution(
    projections_over_time: List[np.ndarray],
    labels: np.ndarray,
    time_points: List[Any],
    reducer_name: str,
    dataset_name: str,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """Plot evolution of reducer embedding over training/time.
    
    Args:
        projections_over_time: List of projection arrays at different time points
        labels: Class labels
        time_points: Time points (epochs, iterations, etc.)
        reducer_name: Name of the reducer
        dataset_name: Name of the dataset
        title: Custom title
        save_path: Path to save the plot
        figsize: Figure size
        
    Returns:
        Figure object
    """
    n_timepoints = len(projections_over_time)
    cols = min(4, n_timepoints)
    rows = (n_timepoints + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    if n_timepoints == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    # Flatten axes for easy indexing
    axes_flat = axes.flatten()
    
    # Get color mapping
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for i, (projection, time_point) in enumerate(zip(projections_over_time, time_points)):
        ax = axes_flat[i]
        
        # Ensure 2D
        if projection.shape[1] > 2:
            projection = projection[:, :2]
        
        # Plot each class
        for j, class_label in enumerate(unique_labels):
            mask = labels == class_label
            if mask.sum() > 0:
                ax.scatter(
                    projection[mask, 0], 
                    projection[mask, 1],
                    c=[colors[j]], 
                    alpha=0.7,
                    s=10,
                    edgecolors='none'
                )
        
        # Removed subplot title for academic article format
        ax.grid(True, alpha=0.3)
        
        # Only label axes on leftmost and bottom subplots
        if i % cols == 0:
            ax.set_ylabel('Component 2')
        if i >= (rows - 1) * cols:
            ax.set_xlabel('Component 1')
    
    # Hide unused subplots
    for i in range(n_timepoints, len(axes_flat)):
        axes_flat[i].axis('off')
    
    if title:
        fig.suptitle(title)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        save_figure(fig, save_path)
    
    return fig


# Convenience function for ReducerManager
def create_reducer_visualization(
    reducer, 
    training_data: torch.Tensor, 
    labels: torch.Tensor,
    reducer_key: str, 
    dataset_name: str,
    save_path: Optional[str] = None,
    query_class: Optional[int] = None
) -> str:
    """Create and save a reducer visualization (convenience function for ReducerManager).
    
    Args:
        reducer: Fitted reducer instance
        training_data: Training data tensor
        labels: Class labels tensor
        reducer_key: Reducer identifier
        dataset_name: Dataset name
        save_path: Optional save path
        query_class: Optional class to highlight with different marker
        
    Returns:
        Path to saved visualization
    """
    # Transform data
    reduced_data = reducer.transform(training_data)
    
    # Convert labels to numpy
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if labels.ndim > 1:
        labels = labels.squeeze()
    
    # Generate save path if not provided
    if save_path is None:
        save_path = f"reducer_visualization_{reducer_key}.png"
    
    # Create visualization
    fig = plot_reducer_projection(
        reduced_data=reduced_data,
        labels=labels,
        reducer_name=reducer_key.upper(),
        dataset_name=dataset_name,
        save_path=save_path,
        query_class=query_class
    )
    
    plt.close(fig)
    return save_path 
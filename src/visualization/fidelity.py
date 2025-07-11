"""
Fidelity Visualization Functions

Provides plotting functions for fidelity analysis including metric comparisons
and distribution visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List, Tuple
from .config import get_color_palette
from .common import save_figure


def plot_fidelity_metrics(
    metrics: Dict[str, float],
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6),
    colors: Optional[Dict[str, str]] = None
) -> None:
    """
    Plot fidelity metrics as a horizontal bar chart.
    
    Args:
        metrics: Dictionary of metric names to values
        save_path: Optional path to save the plot
        title: Plot title
        figsize: Figure size tuple
        colors: Optional color scheme override
    """
    if colors is None:
        colors = get_color_palette('fidelity')
    
    fig, ax = plt.subplots(figsize=figsize)
    
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    
    # Assign colors to metrics
    bar_colors = []
    for name in metric_names:
        name_lower = name.lower()
        if name_lower in colors:
            bar_colors.append(colors[name_lower])
        else:
            # Default color if metric not in color scheme
            bar_colors.append(colors.get('fid', '#6A4C93'))
    
    bars = ax.barh(metric_names, metric_values, color=bar_colors)
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        width = bar.get_width()
        ax.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                f'{value:.3f}', ha='left', va='center')
    
    ax.set_xlabel('Score')
    if title:

        ax.set_title(title)
    ax.set_xlim(0, max(metric_values) * 1.15)
    
    plt.tight_layout()
    save_figure(fig, save_path)


def create_fidelity_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 8),
    colors: Optional[Dict[str, str]] = None
) -> None:
    """
    Create a comparison plot of fidelity metrics across different models/conditions.
    
    Args:
        metrics_dict: Nested dictionary {model_name: {metric_name: value}}
        save_path: Optional path to save the plot
        title: Plot title
        figsize: Figure size tuple
        colors: Optional color scheme override
    """
    if colors is None:
        colors = get_color_palette('fidelity')
    
    # Extract all unique metrics
    all_metrics = set()
    for model_metrics in metrics_dict.values():
        all_metrics.update(model_metrics.keys())
    all_metrics = sorted(list(all_metrics))
    
    model_names = list(metrics_dict.keys())
    n_models = len(model_names)
    n_metrics = len(all_metrics)
    
    # Set up the figure
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    
    x = np.arange(n_models)
    width = 0.6
    
    for i, metric in enumerate(all_metrics):
        ax = axes[i]
        
        # Get values for this metric across all models
        values = []
        for model_name in model_names:
            values.append(metrics_dict[model_name].get(metric, 0))
        
        # Choose color for this metric
        metric_lower = metric.lower()
        color = colors.get(metric_lower, colors.get('fid', '#6A4C93'))
        
        bars = ax.bar(x, values, width, label=metric, color=color)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{value:.3f}', ha='center', va='bottom', rotation=90)
        
        ax.set_xlabel('Model')
        ax.set_ylabel(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_ylim(0, max(values) * 1.15 if values else 1)
    
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    save_figure(fig, save_path)


def plot_metric_distribution(
    metric_values: List[float],
    metric_name: str,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6),
    bins: int = 30,
    color: Optional[str] = None
) -> None:
    """
    Plot distribution of a fidelity metric.
    
    Args:
        metric_values: List of metric values
        metric_name: Name of the metric
        save_path: Optional path to save the plot
        title: Optional plot title (defaults to metric distribution)
        figsize: Figure size tuple
        bins: Number of histogram bins
        color: Optional color override
    """
    if color is None:
        colors = get_color_palette('fidelity')
        color = colors.get(metric_name.lower(), colors.get('fid', '#6A4C93'))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.hist(metric_values, bins=bins, color=color, alpha=0.7, edgecolor='black')
    
    # Add statistics
    mean_val = np.mean(metric_values)
    std_val = np.std(metric_values)
    
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
    ax.axvline(mean_val + std_val, color='orange', linestyle=':', linewidth=2, label=f'+1σ: {mean_val + std_val:.3f}')
    ax.axvline(mean_val - std_val, color='orange', linestyle=':', linewidth=2, label=f'-1σ: {mean_val - std_val:.3f}')
    
    ax.set_xlabel(metric_name)
    ax.set_ylabel('Frequency')
    if title:

        ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=plt.rcParams.get('grid.alpha', 0.3))
    
    plt.tight_layout()
    save_figure(fig, save_path)


def plot_fidelity_heatmap(
    metrics_matrix: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 8),
    cmap: str = 'viridis',
    annotate: bool = True
) -> None:
    """
    Create a heatmap of fidelity metrics.
    
    Args:
        metrics_matrix: 2D array of metric values
        row_labels: Labels for rows (e.g., model names)
        col_labels: Labels for columns (e.g., metric names)
        save_path: Optional path to save the plot
        title: Plot title
        figsize: Figure size tuple
        cmap: Colormap name
        annotate: Whether to annotate cells with values
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(metrics_matrix, cmap=cmap, aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Metric Value')
    
    # Annotate cells with values
    if annotate:
        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                text = ax.text(j, i, f'{metrics_matrix[i, j]:.3f}',
                             ha="center", va="center", color="white" if metrics_matrix[i, j] < np.mean(metrics_matrix) else "black")
    
    if title:

    
        ax.set_title(title)
    plt.tight_layout()
    save_figure(fig, save_path)


def plot_fidelity_radar(
    metrics: Dict[str, float],
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 8),
    color: str = '#1f77b4',
    alpha: float = 0.25
) -> None:
    """
    Create a radar chart of fidelity metrics.
    
    Args:
        metrics: Dictionary of metric names to values
        save_path: Optional path to save the plot
        title: Plot title
        figsize: Figure size tuple
        color: Fill color
        alpha: Fill transparency
    """
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    
    # Number of variables
    N = len(metric_names)
    
    # Compute angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    # Add values for completing the circle
    metric_values += metric_values[:1]
    
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
    
    # Plot the polygon
    ax.plot(angles, metric_values, 'o-', linewidth=2, color=color)
    ax.fill(angles, metric_values, alpha=alpha, color=color)
    
    # Add labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names)
    
    # Set y-axis limits
    ax.set_ylim(0, max(metric_values) * 1.1)
    
    if title:

    
        ax.set_title(title, pad=20)
    ax.grid(True)
    
    plt.tight_layout()
    save_figure(fig, save_path) 
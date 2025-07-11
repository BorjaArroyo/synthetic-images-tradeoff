"""
Common Visualization Utilities

Shared functions for figure management, saving, and layout.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Optional, Tuple, List, Union
import os
import mlflow


def save_figure(
    fig: plt.Figure, 
    save_path: Optional[str] = None,
    dpi: Optional[int] = None,
    bbox_inches: str = 'tight',
    pad_inches: Optional[float] = None,
    show: bool = False,
    close: bool = True
) -> None:
    """
    Save or show a matplotlib figure with proper handling.
    
    Args:
        fig: Matplotlib figure object
        save_path: Path to save the figure (if None, figure is shown)
        dpi: DPI for saving (if None, uses rcParams setting)
        bbox_inches: Bounding box mode for saving
        pad_inches: Padding for tight bbox (if None, uses rcParams setting)
        show: Whether to show the figure (in addition to saving)
        close: Whether to close the figure after saving/showing
    """
    try:
        if save_path:
            # Use rcParams defaults if not specified
            save_dpi = dpi or plt.rcParams.get('savefig.dpi', 300)
            save_pad = pad_inches or plt.rcParams.get('savefig.pad_inches', 0.1)
            
            # Ensure directory exists (only if path has a directory)
            dir_path = os.path.dirname(save_path)
            if dir_path:  # Only create directory if path contains one
                os.makedirs(dir_path, exist_ok=True)
            
            # Save the figure
            fig.savefig(
                save_path, 
                dpi=save_dpi, 
                bbox_inches=bbox_inches,
                pad_inches=save_pad
            )
            
            print(f"📊 Saved figure: {save_path}")
        
        if show or save_path is None:
            plt.show()
            
    finally:
        if close:
            plt.close(fig)


def create_figure_grid(
    nrows: int,
    ncols: int,
    figsize: Optional[Tuple[float, float]] = None,
    hspace: float = 0.3,
    wspace: float = 0.3,
    **subplot_kw
) -> Tuple[plt.Figure, Union[plt.Axes, List[plt.Axes]]]:
    """
    Create a figure with a grid of subplots.
    
    Args:
        nrows: Number of rows
        ncols: Number of columns
        figsize: Figure size (if None, automatically calculated)
        hspace: Height spacing between subplots
        wspace: Width spacing between subplots
        **subplot_kw: Additional arguments for subplots
        
    Returns:
        Tuple of (figure, axes)
    """
    if figsize is None:
        # Auto-calculate reasonable figure size
        figsize = (4 * ncols, 3 * nrows)
    
    fig, axes = plt.subplots(
        nrows, ncols, 
        figsize=figsize,
        **subplot_kw
    )
    
    # Adjust spacing
    plt.subplots_adjust(hspace=hspace, wspace=wspace)
    
    return fig, axes


def add_subplot_labels(
    axes: Union[plt.Axes, List[plt.Axes]],
    labels: List[str],
    position: str = 'top-left',
    fontsize: Optional[int] = None,
    fontweight: str = 'bold',
    **text_kwargs
) -> None:
    """
    Add labels to subplots (e.g., 'A', 'B', 'C', etc.).
    
    Args:
        axes: Axes or list of axes
        labels: List of labels to add
        position: Position of labels ('top-left', 'top-right', etc.)
        fontsize: Font size (if None, uses current style)
        fontweight: Font weight
        **text_kwargs: Additional text formatting arguments
    """
    if not isinstance(axes, (list, tuple)):
        axes = [axes]
    
    if fontsize is None:
        fontsize = plt.rcParams.get('axes.titlesize', 16)
    
    # Define position coordinates
    positions = {
        'top-left': (0.02, 0.98),
        'top-right': (0.98, 0.98),
        'bottom-left': (0.02, 0.02),
        'bottom-right': (0.98, 0.02),
    }
    
    # Define alignment based on position
    alignments = {
        'top-left': ('left', 'top'),
        'top-right': ('right', 'top'),
        'bottom-left': ('left', 'bottom'),
        'bottom-right': ('right', 'bottom'),
    }
    
    x, y = positions.get(position, positions['top-left'])
    ha, va = alignments.get(position, alignments['top-left'])
    
    for ax, label in zip(axes, labels):
        ax.text(
            x, y, label,
            transform=ax.transAxes,
            fontsize=fontsize,
            fontweight=fontweight,
            ha=ha, va=va,
            **text_kwargs
        ) 
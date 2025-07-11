"""
Visualization Configuration Module

Handles matplotlib styling and configuration for publication-quality plots.
Provides presets for different output formats (articles, presentations, posters).
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Dict, Any, Optional
import warnings

# Style presets for different use cases
ARTICLE_STYLE = {
    'font.size': 22,
    'axes.titlesize': 22,
    'axes.labelsize': 22,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'legend.fontsize': 20,
    'figure.titlesize': 18,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.8,
    'lines.linewidth': 2.5,
    'lines.markersize': 6,
    'patch.linewidth': 1.2,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'figure.dpi': 100,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'axes.grid.axis': 'y',
    'grid.alpha': 0.3,
    'axes.axisbelow': True,
}

PRESENTATION_STYLE = {
    'font.size': 18,
    'axes.titlesize': 22,
    'axes.labelsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
    'figure.titlesize': 24,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'sans-serif'],
    'axes.linewidth': 1.5,
    'grid.linewidth': 1.0,
    'lines.linewidth': 3.0,
    'lines.markersize': 8,
    'patch.linewidth': 1.5,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
    'figure.dpi': 100,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'axes.grid.axis': 'y',
    'grid.alpha': 0.4,
    'axes.axisbelow': True,
}

POSTER_STYLE = {
    'font.size': 24,
    'axes.titlesize': 28,
    'axes.labelsize': 26,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 24,
    'figure.titlesize': 32,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'sans-serif'],
    'axes.linewidth': 2.0,
    'grid.linewidth': 1.2,
    'lines.linewidth': 4.0,
    'lines.markersize': 10,
    'patch.linewidth': 2.0,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.2,
    'figure.dpi': 100,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'axes.grid.axis': 'y',
    'grid.alpha': 0.5,
    'axes.axisbelow': True,
}

# Current active style
_current_style = None


def set_publication_style(style_dict: Dict[str, Any], style_name: str = "custom") -> None:
    """
    Set custom publication style for matplotlib.
    
    Args:
        style_dict: Dictionary of matplotlib rcParams
        style_name: Name of the style for tracking
    """
    global _current_style
    
    # Apply the style
    plt.rcParams.update(style_dict)
    _current_style = style_name
    
    # Suppress font warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
    
    print(f"📊 Applied '{style_name}' visualization style")


def set_article_style() -> None:
    """Set style optimized for journal articles and papers."""
    set_publication_style(ARTICLE_STYLE, "article")


def set_presentation_style() -> None:
    """Set style optimized for presentations and slides.""" 
    set_publication_style(PRESENTATION_STYLE, "presentation")


def set_poster_style() -> None:
    """Set style optimized for posters and large displays."""
    set_publication_style(POSTER_STYLE, "poster")


def reset_style() -> None:
    """Reset matplotlib to default settings."""
    global _current_style
    plt.rcdefaults()
    _current_style = None
    print("📊 Reset to default matplotlib style")


def get_current_style() -> Optional[str]:
    """Get the name of the currently active style."""
    return _current_style


def customize_style(
    base_style: str = "article",
    font_scale: float = 1.0,
    line_scale: float = 1.0,
    **kwargs
) -> None:
    """
    Create a customized style based on a preset.
    
    Args:
        base_style: Base style to modify ("article", "presentation", or "poster")
        font_scale: Scale factor for all font sizes
        line_scale: Scale factor for all line widths
        **kwargs: Additional rcParams to override
    """
    # Get base style
    if base_style == "article":
        style = ARTICLE_STYLE.copy()
    elif base_style == "presentation":
        style = PRESENTATION_STYLE.copy()
    elif base_style == "poster":
        style = POSTER_STYLE.copy()
    else:
        raise ValueError(f"Unknown base style: {base_style}")
    
    # Apply scaling
    font_params = ['font.size', 'axes.titlesize', 'axes.labelsize', 
                   'xtick.labelsize', 'ytick.labelsize', 'legend.fontsize', 
                   'figure.titlesize']
    
    line_params = ['axes.linewidth', 'grid.linewidth', 'lines.linewidth',
                   'patch.linewidth']
    
    for param in font_params:
        if param in style:
            style[param] = int(style[param] * font_scale)
    
    for param in line_params:
        if param in style:
            style[param] = style[param] * line_scale
    
    # Apply custom overrides
    style.update(kwargs)
    
    # Set the style
    custom_name = f"{base_style}_custom"
    set_publication_style(style, custom_name)


# Color schemes for consistent styling
COLORS = {
    'privacy': {
        'reference': '#2E86AB',      # Blue
        'synthetic': '#A23B72',      # Purple
        'query': '#000000',          # Shining black
        'victim': '#C73E1D',         # Red
        'ablated': '#1B998B',        # Teal
    },
    'utility': {
        'original': '#264653',       # Dark green
        'synthetic': '#2A9D8F',      # Teal
        'gradcam': '#E76F51',        # Orange-red
        'accuracy': '#F4A261',       # Light orange
    },
    'fidelity': {
        'fid': '#6A4C93',           # Purple
        'lpips': '#1982C4',         # Blue
        'psnr': '#8AC926',          # Green
        'ssim': '#FFCA3A',          # Yellow
        'is': '#FF595E',            # Red
    }
}


def get_color_palette(category: str = 'privacy') -> Dict[str, str]:
    """
    Get color palette for a specific visualization category.
    
    Args:
        category: Category name ('privacy', 'utility', 'fidelity')
        
    Returns:
        Dictionary mapping semantic names to hex colors
    """
    return COLORS.get(category, COLORS['privacy']).copy()


# Initialize with article style when module is imported
# This happens automatically when importing src.visualization
def _initialize_default_style():
    """Initialize default visualization style silently."""
    global _current_style
    if _current_style is None:
        # Apply article style without printing message
        plt.rcParams.update(ARTICLE_STYLE)
        _current_style = "article"
        
        # Suppress font warnings
        warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


# Call initialization
_initialize_default_style() 
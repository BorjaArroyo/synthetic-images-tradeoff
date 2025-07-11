"""
Visualization Module for Tradeoff Framework

This module provides publication-ready visualization functions for privacy, utility, 
and fidelity analysis. It automatically configures matplotlib for high-quality plots
suitable for academic publications.

Usage:
    Simply import this module to automatically set up publication fonts:
    
    >>> import src.visualization as viz
    >>> # Fonts are now configured for publication quality
    >>> viz.plot_ecdf_comparison(ecdf1, ecdf2)
    
    Or import specific components:
    
    >>> from src.visualization.privacy import plot_ecdf_comparison
    >>> from src.visualization.utility import plot_gradcam_analysis
"""

import matplotlib.pyplot as plt
from typing import Dict, Any

# Import visualization components
from .config import (
    set_publication_style,
    set_article_style, 
    set_presentation_style,
    set_poster_style,
    reset_style,
    ARTICLE_STYLE,
    PRESENTATION_STYLE,
    POSTER_STYLE
)

from .privacy import (
    plot_ecdf_comparison,
    plot_hexbin_difference,
    plot_projections_comparison,
    plot_projections_by_class,
    create_privacy_visualizations,
    create_victim_model_query_grid,
    create_privacy_impact_visualization
)

from .utility import (
    plot_gradcam_analysis,
    plot_gradcam_comparison,
    create_synthetic_image_grid,
    plot_utility_metrics,
    plot_training_curves
)

from .fidelity import (
    plot_fidelity_metrics,
    create_fidelity_comparison
)

from .common import (
    save_figure,
    create_figure_grid,
    add_subplot_labels
)

from .reducers import (
    plot_reducer_projection,
    plot_reducer_comparison,
    plot_reducer_quality_metrics,
    plot_reducer_embedding_evolution,
    create_reducer_visualization
)

# Automatically configure for article-quality plots on import
set_article_style()

__all__ = [
    # Configuration
    'set_publication_style',
    'set_article_style',
    'set_presentation_style', 
    'set_poster_style',
    'reset_style',
    'ARTICLE_STYLE',
    'PRESENTATION_STYLE',
    'POSTER_STYLE',
    
    # Privacy visualizations
    'plot_ecdf_comparison',
    'plot_hexbin_difference',
    'plot_projections_comparison',
    'plot_projections_by_class',
    'create_privacy_visualizations',
    'create_victim_model_query_grid',
    'create_privacy_impact_visualization',
    
    # Utility visualizations
    'plot_gradcam_analysis',
    'plot_gradcam_comparison',
    'create_synthetic_image_grid',
    'plot_utility_metrics',
    'plot_training_curves',
    
    # Fidelity visualizations
    'plot_fidelity_metrics',
    'create_fidelity_comparison',
    
    # Reducer visualizations
    'plot_reducer_projection',
    'plot_reducer_comparison',
    'plot_reducer_quality_metrics',
    'plot_reducer_embedding_evolution',
    'create_reducer_visualization',
    
    # Common utilities
    'save_figure',
    'create_figure_grid', 
    'add_subplot_labels'
] 
"""
Visualization functionality for epitope predictions.
"""

import logging
from typing import Optional, Dict, Any, Union
import numpy as np
import matplotlib.pyplot as plt
import nglview as nv
from Bio.PDB import Structure

logger = logging.getLogger(__name__)

def visualize_epitopes(
    structure: Structure,
    scores: np.ndarray,
    viewport: str = "ngl",
    threshold: float = 0.5,
    **kwargs: Any
) -> Union[nv.NGLWidget, plt.Figure]:
    """
    Visualize epitope predictions on the protein structure.
    
    Parameters
    ----------
    structure : Structure
        Bio.PDB Structure object
    scores : np.ndarray
        Epitope scores for each residue
    viewport : str, default="ngl"
        Visualization backend ("ngl" or "matplotlib")
    threshold : float, default=0.5
        Score threshold for highlighting residues
    **kwargs : Any
        Additional visualization parameters
        
    Returns
    -------
    Union[nv.NGLWidget, plt.Figure]
        Visualization object (depends on viewport)
    """
    if viewport == "ngl":
        return _visualize_ngl(structure, scores, threshold, **kwargs)
    elif viewport == "matplotlib":
        return _visualize_matplotlib(structure, scores, threshold, **kwargs)
    else:
        raise ValueError(f"Unsupported viewport: {viewport}")

def _visualize_ngl(
    structure: Structure,
    scores: np.ndarray,
    threshold: float = 0.5,
    **kwargs: Any
) -> nv.NGLWidget:
    """
    Visualize epitope predictions using NGLView.
    
    Parameters
    ----------
    structure : Structure
        Bio.PDB Structure object
    scores : np.ndarray
        Epitope scores for each residue
    threshold : float, default=0.5
        Score threshold for highlighting residues
    **kwargs : Any
        Additional visualization parameters
        
    Returns
    -------
    nv.NGLWidget
        NGLView widget for Jupyter notebooks
    """
    # Create NGLView widget
    view = nv.show_biopython(structure)
    
    # Add cartoon representation
    view.add_cartoon(selection="protein", color="white", opacity=0.3)
    
    # Color residues by epitope score
    for i, score in enumerate(scores):
        if score >= threshold:
            # Red for high-scoring residues
            color = "red"
            opacity = min(1.0, score)
        else:
            # Blue for low-scoring residues
            color = "blue"
            opacity = 0.3
        
        # Add sphere representation for each residue
        view.add_ball_and_stick(
            selection=f"protein and {i+1}",
            color=color,
            opacity=opacity
        )
    
    # Set camera and other display options
    view.camera = "orthographic"
    view.background = "white"
    
    return view

def _visualize_matplotlib(
    structure: Structure,
    scores: np.ndarray,
    threshold: float = 0.5,
    **kwargs: Any
) -> plt.Figure:
    """
    Visualize epitope predictions using Matplotlib.
    
    Parameters
    ----------
    structure : Structure
        Bio.PDB Structure object
    scores : np.ndarray
        Epitope scores for each residue
    threshold : float, default=0.5
        Score threshold for highlighting residues
    **kwargs : Any
        Additional visualization parameters
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot score distribution
    ax1.hist(scores, bins=20, color="gray", alpha=0.5)
    ax1.axvline(threshold, color="red", linestyle="--", label="Threshold")
    ax1.set_xlabel("Epitope Score")
    ax1.set_ylabel("Count")
    ax1.set_title("Score Distribution")
    ax1.legend()
    
    # Plot score vs. residue position
    positions = np.arange(len(scores))
    ax2.plot(positions, scores, color="gray", alpha=0.5)
    ax2.fill_between(
        positions,
        scores,
        where=scores >= threshold,
        color="red",
        alpha=0.3,
        label="Epitope"
    )
    ax2.axhline(threshold, color="red", linestyle="--", label="Threshold")
    ax2.set_xlabel("Residue Position")
    ax2.set_ylabel("Epitope Score")
    ax2.set_title("Score vs. Position")
    ax2.legend()
    
    plt.tight_layout()
    return fig

def save_visualization(
    viz: Union[nv.NGLWidget, plt.Figure],
    output_path: str,
    **kwargs: Any
) -> None:
    """
    Save visualization to file.
    
    Parameters
    ----------
    viz : Union[nv.NGLWidget, plt.Figure]
        Visualization object
    output_path : str
        Path to save visualization
    **kwargs : Any
        Additional save parameters
    """
    if isinstance(viz, nv.NGLWidget):
        # Save NGLView widget as HTML
        viz.download_image(output_path)
    elif isinstance(viz, plt.Figure):
        # Save Matplotlib figure
        viz.savefig(output_path, **kwargs)
    else:
        raise ValueError(f"Unsupported visualization type: {type(viz)}") 
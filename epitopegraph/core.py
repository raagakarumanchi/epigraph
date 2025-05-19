"""
Core functionality for the EpitopeGraph package.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import logging

import torch
import numpy as np
from Bio.PDB import Structure

from epitopegraph.data import fetch_structure, fetch_sequence
from epitopegraph.graph import build_residue_graph
from epitopegraph.models import predict_epitopes
from epitopegraph.viz import visualize_epitopes

logger = logging.getLogger(__name__)

@dataclass
class EpitopeGraph:
    """
    Main class for epitope prediction using graph neural networks.
    
    Parameters
    ----------
    uniprot_id : str
        UniProt identifier for the target protein
    cache_dir : str, optional
        Directory to cache downloaded structures and sequences
    device : str, optional
        Device to run the model on ('cpu' or 'cuda')
    """
    
    uniprot_id: str
    cache_dir: str = ".cache"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        """Initialize internal state."""
        self.sequence: Optional[str] = None
        self.structure: Optional[Structure] = None
        self.graph: Optional[torch_geometric.data.Data] = None
        self.epitope_scores: Optional[np.ndarray] = None
        
    def fetch_sequence(self) -> str:
        """
        Fetch protein sequence from UniProt.
        
        Returns
        -------
        str
            Protein sequence
        """
        self.sequence = fetch_sequence(self.uniprot_id, cache_dir=self.cache_dir)
        return self.sequence
    
    def fetch_structure(self) -> Structure:
        """
        Fetch protein structure from AlphaFold DB.
        
        Returns
        -------
        Structure
            Bio.PDB Structure object
        """
        self.structure = fetch_structure(self.uniprot_id, cache_dir=self.cache_dir)
        return self.structure
    
    def build_residue_graph(self, cutoff: float = 8.0) -> torch_geometric.data.Data:
        """
        Convert protein structure to residue-level graph.
        
        Parameters
        ----------
        cutoff : float, default=8.0
            Distance cutoff in Angstroms for edge creation
            
        Returns
        -------
        torch_geometric.data.Data
            Graph representation of the protein
        """
        if self.structure is None:
            raise ValueError("Structure must be fetched before building graph")
            
        self.graph = build_residue_graph(
            self.structure,
            cutoff=cutoff,
            device=self.device
        )
        return self.graph
    
    def predict_epitopes(self) -> np.ndarray:
        """
        Predict epitope scores for each residue.
        
        Returns
        -------
        np.ndarray
            Array of epitope scores per residue
        """
        if self.graph is None:
            raise ValueError("Graph must be built before prediction")
            
        self.epitope_scores = predict_epitopes(
            self.graph,
            device=self.device
        )
        return self.epitope_scores
    
    def visualize_epitopes(
        self,
        viewport: str = "ngl",
        threshold: float = 0.5,
        **kwargs: Any
    ) -> Any:
        """
        Visualize epitope predictions on the structure.
        
        Parameters
        ----------
        viewport : str, default="ngl"
            Visualization backend ("ngl" or "matplotlib")
        threshold : float, default=0.5
            Score threshold for highlighting residues
        **kwargs : Any
            Additional visualization parameters
            
        Returns
        -------
        Any
            Visualization object (depends on viewport)
        """
        if self.epitope_scores is None:
            raise ValueError("Epitope scores must be computed before visualization")
            
        return visualize_epitopes(
            self.structure,
            self.epitope_scores,
            viewport=viewport,
            threshold=threshold,
            **kwargs
        )
    
    def get_epitope_residues(self, threshold: float = 0.5) -> List[int]:
        """
        Get list of residues predicted as epitopes.
        
        Parameters
        ----------
        threshold : float, default=0.5
            Score threshold for epitope classification
            
        Returns
        -------
        List[int]
            List of residue indices predicted as epitopes
        """
        if self.epitope_scores is None:
            raise ValueError("Epitope scores must be computed first")
            
        return list(np.where(self.epitope_scores >= threshold)[0]) 
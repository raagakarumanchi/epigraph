"""
Graph construction and processing functionality.
"""

import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
from torch_geometric.data import Data
from Bio.PDB import Structure, Residue, Atom

logger = logging.getLogger(__name__)

def _get_residue_features(residue: Residue) -> torch.Tensor:
    """
    Extract features for a single residue.
    
    Parameters
    ----------
    residue : Residue
        Bio.PDB Residue object
        
    Returns
    -------
    torch.Tensor
        Feature vector for the residue
    """
    # One-hot encode amino acid type (20 standard AAs)
    aa_dict = {
        'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4,
        'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
        'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
        'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19
    }
    
    features = torch.zeros(20)
    if residue.resname in aa_dict:
        features[aa_dict[residue.resname]] = 1.0
    
    # Add secondary structure (placeholder - would need DSSP)
    ss_features = torch.zeros(3)  # [helix, sheet, coil]
    features = torch.cat([features, ss_features])
    
    # Add surface accessibility (placeholder - would need SASA)
    sasa_feature = torch.tensor([0.0])  # Normalized SASA
    features = torch.cat([features, sasa_feature])
    
    return features

def _get_residue_center(residue: Residue) -> np.ndarray:
    """
    Calculate the center of mass for a residue.
    
    Parameters
    ----------
    residue : Residue
        Bio.PDB Residue object
        
    Returns
    -------
    np.ndarray
        3D coordinates of residue center
    """
    coords = []
    for atom in residue:
        if atom.id == "CA":  # Use C-alpha atom
            return atom.get_coord()
    
    # Fallback to center of mass if no CA
    for atom in residue:
        coords.append(atom.get_coord())
    return np.mean(coords, axis=0)

def build_residue_graph(
    structure: Structure,
    cutoff: float = 8.0,
    device: str = "cpu"
) -> Data:
    """
    Convert protein structure to residue-level graph.
    
    Parameters
    ----------
    structure : Structure
        Bio.PDB Structure object
    cutoff : float, default=8.0
        Distance cutoff in Angstroms for edge creation
    device : str, default="cpu"
        Device to store tensors on
        
    Returns
    -------
    Data
        PyTorch Geometric Data object containing:
        - x: Node features (residue properties)
        - edge_index: Graph connectivity
        - edge_attr: Edge features (distances)
        - pos: Node positions (residue centers)
    """
    model = structure[0]  # Get first model
    chain = next(model.get_chains())  # Get first chain
    
    # Get residues and their features
    residues = list(chain.get_residues())
    n_residues = len(residues)
    
    # Extract node features and positions
    x = []
    pos = []
    for residue in residues:
        if residue.id[0] == " ":  # Skip hetero atoms
            x.append(_get_residue_features(residue))
            pos.append(_get_residue_center(residue))
    
    x = torch.stack(x).to(device)
    pos = torch.tensor(pos, dtype=torch.float).to(device)
    
    # Build edge list based on distance cutoff
    edge_index = []
    edge_attr = []
    
    for i in range(n_residues):
        for j in range(i + 1, n_residues):
            if residues[i].id[0] != " " or residues[j].id[0] != " ":
                continue  # Skip hetero atoms
                
            dist = np.linalg.norm(
                _get_residue_center(residues[i]) - 
                _get_residue_center(residues[j])
            )
            
            if dist <= cutoff:
                # Add both directions for undirected graph
                edge_index.extend([[i, j], [j, i]])
                edge_attr.extend([[dist], [dist]])
    
    if not edge_index:  # Handle case with no edges
        edge_index = torch.zeros((2, 0), dtype=torch.long).to(device)
        edge_attr = torch.zeros((0, 1), dtype=torch.float).to(device)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().to(device)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float).to(device)
    
    # Create PyTorch Geometric Data object
    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=pos
    )

def get_graph_metadata(graph: Data) -> Dict[str, Any]:
    """
    Extract metadata from a protein graph.
    
    Parameters
    ----------
    graph : Data
        PyTorch Geometric Data object
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing graph metadata
    """
    return {
        "num_nodes": graph.num_nodes,
        "num_edges": graph.num_edges,
        "node_feature_dim": graph.x.size(1),
        "edge_feature_dim": graph.edge_attr.size(1) if graph.edge_attr is not None else 0,
        "is_directed": False,  # We always create undirected graphs
        "has_self_loops": False,  # We don't create self-loops
    } 
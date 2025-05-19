"""
EpitopeGraph: A graph-based toolkit for conformational B-cell epitope prediction.
"""

__version__ = "0.1.0"

from epitopegraph.core import EpitopeGraph
from epitopegraph.data import fetch_structure, fetch_sequence
from epitopegraph.graph import build_residue_graph
from epitopegraph.models import predict_epitopes
from epitopegraph.viz import visualize_epitopes

__all__ = [
    "EpitopeGraph",
    "fetch_structure",
    "fetch_sequence",
    "build_residue_graph",
    "predict_epitopes",
    "visualize_epitopes",
] 
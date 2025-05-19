"""
Tests for core EpitopeGraph functionality.
"""

import os
import pytest
import numpy as np
from Bio.PDB import Structure

from epitopegraph import EpitopeGraph
from epitopegraph.data import fetch_sequence, fetch_structure
from epitopegraph.graph import build_residue_graph
from epitopegraph.models import predict_epitopes

# Test data
TEST_UNIPROT_ID = "P0DTC2"  # SARS-CoV-2 spike protein
TEST_CACHE_DIR = ".test_cache"

@pytest.fixture
def eg():
    """Create EpitopeGraph instance for testing."""
    return EpitopeGraph(
        uniprot_id=TEST_UNIPROT_ID,
        cache_dir=TEST_CACHE_DIR
    )

def test_fetch_sequence(eg):
    """Test sequence fetching."""
    sequence = eg.fetch_sequence()
    assert isinstance(sequence, str)
    assert len(sequence) > 0
    assert all(c in "ACDEFGHIKLMNPQRSTVWY" for c in sequence)

def test_fetch_structure(eg):
    """Test structure fetching."""
    structure = eg.fetch_structure()
    assert isinstance(structure, Structure)
    assert len(list(structure.get_chains())) > 0
    assert len(list(next(structure.get_chains()).get_residues())) > 0

def test_build_residue_graph(eg):
    """Test graph construction."""
    eg.fetch_structure()  # Need structure first
    graph = eg.build_residue_graph(cutoff=8.0)
    
    assert graph.num_nodes > 0
    assert graph.num_edges > 0
    assert graph.x.size(1) == 24  # 20 AA types + 3 SS + 1 SASA
    assert graph.edge_attr.size(1) == 1  # Distance features

def test_predict_epitopes(eg):
    """Test epitope prediction."""
    eg.fetch_structure()
    eg.build_residue_graph()
    scores = eg.predict_epitopes()
    
    assert isinstance(scores, np.ndarray)
    assert len(scores) == eg.graph.num_nodes
    assert np.all((scores >= 0) & (scores <= 1))

def test_visualize_epitopes(eg):
    """Test visualization."""
    eg.fetch_structure()
    eg.build_residue_graph()
    eg.predict_epitopes()
    
    # Test NGLView visualization
    ngl_viz = eg.visualize_epitopes(viewport="ngl")
    assert hasattr(ngl_viz, "add_cartoon")  # Basic NGLView check
    
    # Test Matplotlib visualization
    mpl_viz = eg.visualize_epitopes(viewport="matplotlib")
    assert hasattr(mpl_viz, "savefig")  # Basic Matplotlib check

def test_get_epitope_residues(eg):
    """Test epitope residue extraction."""
    eg.fetch_structure()
    eg.build_residue_graph()
    eg.predict_epitopes()
    
    residues = eg.get_epitope_residues(threshold=0.5)
    assert isinstance(residues, list)
    assert all(isinstance(r, int) for r in residues)
    assert all(0 <= r < eg.graph.num_nodes for r in residues)

def teardown_module(module):
    """Clean up test cache directory."""
    if os.path.exists(TEST_CACHE_DIR):
        for file in os.listdir(TEST_CACHE_DIR):
            os.remove(os.path.join(TEST_CACHE_DIR, file))
        os.rmdir(TEST_CACHE_DIR) 
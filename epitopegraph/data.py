"""
Data retrieval and processing functionality.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import json

import requests
from Bio import SeqIO
from Bio.PDB import PDBParser, Structure
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import warnings

logger = logging.getLogger(__name__)

# Suppress PDB construction warnings
warnings.filterwarnings("ignore", category=PDBConstructionWarning)

# API endpoints
UNIPROT_API = "https://rest.uniprot.org/uniprotkb/{}.fasta"
ALPHAFOLD_API = "https://alphafold.ebi.ac.uk/files/AF-{}-F1-model_v4.pdb"

def _ensure_cache_dir(cache_dir: str) -> Path:
    """Ensure cache directory exists."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path

def fetch_sequence(uniprot_id: str, cache_dir: str = ".cache") -> str:
    """
    Fetch protein sequence from UniProt.
    
    Parameters
    ----------
    uniprot_id : str
        UniProt identifier
    cache_dir : str, default=".cache"
        Directory to cache downloaded sequences
        
    Returns
    -------
    str
        Protein sequence
        
    Raises
    ------
    requests.exceptions.RequestException
        If sequence cannot be fetched from UniProt
    """
    cache_path = _ensure_cache_dir(cache_dir)
    seq_file = cache_path / f"{uniprot_id}.fasta"
    
    # Check cache first
    if seq_file.exists():
        logger.info(f"Loading sequence from cache: {seq_file}")
        with open(seq_file) as f:
            return str(next(SeqIO.parse(f, "fasta")).seq)
    
    # Fetch from UniProt
    logger.info(f"Fetching sequence from UniProt: {uniprot_id}")
    response = requests.get(UNIPROT_API.format(uniprot_id))
    response.raise_for_status()
    
    # Parse and cache
    sequence = str(next(SeqIO.parse(response.text.splitlines(), "fasta")).seq)
    with open(seq_file, "w") as f:
        f.write(response.text)
    
    return sequence

def fetch_structure(
    uniprot_id: str,
    cache_dir: str = ".cache",
    model_id: int = 1
) -> Structure:
    """
    Fetch protein structure from AlphaFold DB.
    
    Parameters
    ----------
    uniprot_id : str
        UniProt identifier
    cache_dir : str, default=".cache"
        Directory to cache downloaded structures
    model_id : int, default=1
        AlphaFold model ID (usually 1 for best model)
        
    Returns
    -------
    Structure
        Bio.PDB Structure object
        
    Raises
    ------
    requests.exceptions.RequestException
        If structure cannot be fetched from AlphaFold DB
    """
    cache_path = _ensure_cache_dir(cache_dir)
    pdb_file = cache_path / f"AF-{uniprot_id}-F1-model_v4.pdb"
    
    # Check cache first
    if pdb_file.exists():
        logger.info(f"Loading structure from cache: {pdb_file}")
        parser = PDBParser(QUIET=True)
        return parser.get_structure(uniprot_id, str(pdb_file))
    
    # Fetch from AlphaFold DB
    logger.info(f"Fetching structure from AlphaFold DB: {uniprot_id}")
    response = requests.get(ALPHAFOLD_API.format(uniprot_id))
    response.raise_for_status()
    
    # Cache and parse
    with open(pdb_file, "w") as f:
        f.write(response.text)
    
    parser = PDBParser(QUIET=True)
    return parser.get_structure(uniprot_id, str(pdb_file))

def get_structure_metadata(structure: Structure) -> Dict[str, Any]:
    """
    Extract metadata from a protein structure.
    
    Parameters
    ----------
    structure : Structure
        Bio.PDB Structure object
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing structure metadata
    """
    model = structure[0]  # Get first model
    chain = next(model.get_chains())  # Get first chain
    
    # Get sequence from structure
    seq = ""
    for residue in chain:
        if residue.id[0] == " ":  # Skip hetero atoms
            seq += residue.resname
    
    return {
        "num_chains": len(list(model.get_chains())),
        "num_residues": len(list(chain.get_residues())),
        "num_atoms": len(list(chain.get_atoms())),
        "sequence": seq,
    } 
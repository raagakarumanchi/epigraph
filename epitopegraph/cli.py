"""Command-line interface for EpitopeGraph."""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
from tqdm import tqdm

from epitopegraph import EpitopeGraph

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="EpitopeGraph: Graph-based epitope prediction"
    )
    
    # Input/output
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input file with UniProt IDs (one per line) or single UniProt ID"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file for predictions (TSV format)"
    )
    
    # Prediction parameters
    parser.add_argument(
        "--graph-cutoff",
        type=float,
        default=8.0,
        help="Distance cutoff for graph edges (Å)"
    )
    parser.add_argument(
        "--include-ss",
        action="store_true",
        help="Include secondary structure information"
    )
    parser.add_argument(
        "--include-sasa",
        action="store_true",
        help="Include solvent accessibility"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Score threshold for epitope classification"
    )
    
    # Batch processing
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=".cache",
        help="Cache directory for downloaded data"
    )
    
    # Optional arguments
    parser.add_argument(
        "--min-length",
        type=int,
        default=5,
        help="Minimum epitope length"
    )
    parser.add_argument(
        "--max-gap",
        type=int,
        default=2,
        help="Maximum gap between residues"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()

def read_uniprot_ids(input_path: str) -> List[str]:
    """Read UniProt IDs from input file or string.
    
    Args:
        input_path: Path to input file or UniProt ID
        
    Returns:
        List of UniProt IDs
    """
    path = Path(input_path)
    
    if path.is_file():
        # Read from file
        with open(path) as f:
            ids = [line.strip() for line in f if line.strip()]
    else:
        # Treat as single UniProt ID
        ids = [input_path]
    
    # Validate IDs
    valid_ids = []
    for uniprot_id in ids:
        if not uniprot_id.startswith(("P", "Q", "O", "A", "B", "C", "D")):
            logger.warning(f"Invalid UniProt ID format: {uniprot_id}")
            continue
        valid_ids.append(uniprot_id)
    
    if not valid_ids:
        raise ValueError("No valid UniProt IDs found")
    
    return valid_ids

def predict_epitopes(
    uniprot_ids: List[str],
    args: argparse.Namespace
) -> pd.DataFrame:
    """Predict epitopes for a list of proteins.
    
    Args:
        uniprot_ids: List of UniProt IDs
        args: Command-line arguments
        
    Returns:
        DataFrame with predictions
    """
    results = []
    
    for uniprot_id in tqdm(uniprot_ids, desc="Processing proteins"):
        try:
            # Initialize model
            eg = EpitopeGraph(
                uniprot_id=uniprot_id,
                cache_dir=args.cache_dir
            )
            
            # Predict epitopes
            scores = eg.predict_epitopes(
                distance_cutoff=args.graph_cutoff,
                include_ss=args.include_ss,
                include_sasa=args.include_sasa,
                threshold=args.threshold
            )
            
            # Get epitope regions
            epitope_regions = eg.get_epitope_residues(
                threshold=args.threshold,
                min_length=args.min_length,
                max_gap=args.max_gap
            )
            
            # Add to results
            for start, end in epitope_regions:
                results.append({
                    "uniprot_id": uniprot_id,
                    "start": start,
                    "end": end,
                    "length": end - start + 1,
                    "sequence": eg.sequence[start:end],
                    "mean_score": scores[start:end].mean(),
                    "max_score": scores[start:end].max()
                })
            
        except Exception as e:
            logger.error(f"Error processing {uniprot_id}: {str(e)}")
            continue
    
    return pd.DataFrame(results)

def main():
    """Main entry point."""
    args = parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    try:
        # Read input
        uniprot_ids = read_uniprot_ids(args.input)
        logger.info(f"Found {len(uniprot_ids)} UniProt IDs")
        
        # Create output directory
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Process in batches
        all_results = []
        for i in range(0, len(uniprot_ids), args.batch):
            batch_ids = uniprot_ids[i:i + args.batch]
            logger.info(f"Processing batch {i//args.batch + 1}")
            
            batch_results = predict_epitopes(batch_ids, args)
            all_results.append(batch_results)
        
        # Combine results
        results = pd.concat(all_results, ignore_index=True)
        
        # Save results
        results.to_csv(
            args.output,
            sep="\t",
            index=False,
            float_format="%.3f"
        )
        logger.info(f"Results saved to {args.output}")
        
        # Print summary
        print("\nPrediction Summary:")
        print(f"Total proteins processed: {len(uniprot_ids)}")
        print(f"Total epitope regions found: {len(results)}")
        print(f"Average epitope length: {results['length'].mean():.1f}")
        print(f"Mean epitope score: {results['mean_score'].mean():.3f}")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
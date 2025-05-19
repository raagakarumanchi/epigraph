"""Benchmark datasets and evaluation for EpitopeGraph."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, auc, f1_score, precision_recall_curve, roc_curve

from epitopegraph import EpitopeGraph

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BenchmarkDataset:
    """Benchmark dataset for epitope prediction evaluation."""
    
    def __init__(self, name: str = "default"):
        """Initialize benchmark dataset.
        
        Args:
            name: Dataset name (default/iedb/pdb)
        """
        self.name = name
        self.data_dir = Path(__file__).parent / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Load dataset
        self.dataset = self._load_dataset()
        
    def _load_dataset(self) -> pd.DataFrame:
        """Load benchmark dataset.
        
        Returns:
            DataFrame with benchmark data
        """
        dataset_file = self.data_dir / f"{self.name}.json"
        
        if not dataset_file.exists():
            # Create default dataset if not exists
            self._create_default_dataset()
        
        with open(dataset_file) as f:
            data = json.load(f)
        
        return pd.DataFrame(data)
    
    def _create_default_dataset(self):
        """Create default benchmark dataset."""
        # Example benchmark dataset with known epitopes
        dataset = [
            {
                "uniprot_id": "P0DTC2",  # SARS-CoV-2 spike
                "epitope_residues": [417, 484, 501],  # RBD residues
                "source": "alphafold",
                "reference": "DOI:10.1038/s41586-020-2196-x"
            },
            {
                "uniprot_id": "P59594",  # SARS-CoV-1 spike
                "epitope_residues": [417, 484, 501],
                "source": "alphafold",
                "reference": "DOI:10.1038/s41586-020-2196-x"
            },
            {
                "uniprot_id": "P0C6X7",  # MERS-CoV spike
                "epitope_residues": [484, 501],
                "source": "alphafold",
                "reference": "DOI:10.1038/s41586-020-2196-x"
            }
        ]
        
        # Save dataset
        with open(self.data_dir / "default.json", "w") as f:
            json.dump(dataset, f, indent=2)
    
    def evaluate(
        self,
        model: EpitopeGraph,
        metrics: Optional[List[str]] = None,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """Evaluate model on benchmark dataset.
        
        Args:
            model: EpitopeGraph model
            metrics: List of metrics to compute
            threshold: Score threshold for binary predictions
            
        Returns:
            Dictionary of metric scores
        """
        metrics = metrics or ["accuracy", "auc", "f1"]
        results = {}
        
        for _, row in self.dataset.iterrows():
            # Get predictions
            scores = model.predict_epitopes(
                uniprot_id=row["uniprot_id"],
                distance_cutoff=8.0
            )
            
            # Get ground truth
            y_true = np.zeros(len(scores))
            y_true[row["epitope_residues"]] = 1
            
            # Compute metrics
            if "accuracy" in metrics:
                y_pred = (scores >= threshold).astype(int)
                results["accuracy"] = accuracy_score(y_true, y_pred)
            
            if "auc" in metrics:
                fpr, tpr, _ = roc_curve(y_true, scores)
                results["auc"] = auc(fpr, tpr)
            
            if "f1" in metrics:
                y_pred = (scores >= threshold).astype(int)
                results["f1"] = f1_score(y_true, y_pred)
        
        return results
    
    def compare_models(
        self,
        models: Dict[str, EpitopeGraph],
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Compare multiple models on benchmark dataset.
        
        Args:
            models: Dictionary of model names to EpitopeGraph instances
            metrics: List of metrics to compute
            
        Returns:
            DataFrame with comparison results
        """
        results = []
        
        for name, model in models.items():
            scores = self.evaluate(model, metrics=metrics)
            scores["model"] = name
            results.append(scores)
        
        return pd.DataFrame(results)
    
    def plot_results(
        self,
        results: pd.DataFrame,
        metric: str = "auc",
        save_path: Optional[Union[str, Path]] = None
    ):
        """Plot benchmark results.
        
        Args:
            results: DataFrame with benchmark results
            metric: Metric to plot
            save_path: Path to save plot
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=results, x="model", y=metric)
        plt.title(f"{metric.upper()} Scores")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def save_results(
        self,
        results: pd.DataFrame,
        name: Optional[str] = None
    ):
        """Save benchmark results.
        
        Args:
            results: DataFrame with benchmark results
            name: Optional name for results file
        """
        name = name or f"benchmark_{self.name}"
        results.to_csv(self.data_dir / f"{name}.csv", index=False)
        results.to_json(self.data_dir / f"{name}.json", orient="records", indent=2) 
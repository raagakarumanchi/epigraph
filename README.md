# EpitopeGraph

A graph-based toolkit for conformational B-cell epitope prediction using graph neural networks.

## Overview

EpitopeGraph is a Python package that uses graph neural networks to predict B-cell epitopes from protein structures. It combines structural information, sequence features, and machine learning to identify potential epitope regions on protein surfaces.

## Features

- Automated data retrieval from UniProt and PDB
- Graph-based representation of protein structures
- GNN-based epitope prediction
- Interactive visualization with NGLView
- Benchmark evaluation tools
- RESTful API access
- Command-line interface for batch processing

## Installation

### Using conda (recommended)

```bash
conda env create -f environment.yml
conda activate epitopegraph
```

### Using pip

```bash
pip install epitopegraph
```

## Quick Start

```python
from epitopegraph import EpitopeGraph

# Initialize with SARS-CoV-2 spike protein
eg = EpitopeGraph(uniprot_id="P0DTC2")

# Predict epitopes
scores = eg.predict_epitopes(
    distance_cutoff=8.0,
    include_ss=True,
    include_sasa=True
)

# Get epitope regions
epitopes = eg.get_epitope_residues(
    threshold=0.5,
    min_length=5,
    max_gap=2
)

# Visualize results
eg.visualize_epitopes()
```

## Command-line Usage

```bash
python -m epitopegraph.predict \
    --input data/ids.txt \
    --output results/epitopes.tsv \
    --graph-cutoff 8.0 \
    --include-ss \
    --include-sasa \
    --batch 50
```

## Documentation

- [User Guide](docs/user_guide.md)
- [API Reference](docs/api.md)
- [Examples](examples/)
- [Benchmarks](docs/benchmarks.md)

## Development

### Setting up the development environment

1. Clone the repository:
```bash
git clone https://github.com/yourusername/epitopegraph.git
cd epitopegraph
```

2. Create and activate the development environment:
```bash
conda env create -f environment.yml
conda activate epitopegraph
```

3. Install development dependencies:
```bash
pip install -e ".[dev]"
```

4. Install pre-commit hooks:
```bash
pre-commit install
```

### Running tests

```bash
pytest
```

### Building documentation

```bash
cd docs
make html
```

## Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use EpitopeGraph in your research, please cite:

```
@software{epitopegraph2024,
  author = {Your Name},
  title = {EpitopeGraph: Graph-based epitope prediction},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/epitopegraph}
}
``` 
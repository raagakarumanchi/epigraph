# Installation Guide

This guide provides detailed instructions for installing EpitopeGraph and its dependencies.

## System Requirements

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)
- 8GB RAM minimum (16GB recommended)
- 10GB free disk space

## Installation Methods

### 1. Using Conda (Recommended)

```bash
# Create a new environment
conda create -n epitopegraph python=3.8
conda activate epitopegraph

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

# Install EpitopeGraph
pip install epitopegraph
```

### 2. Using Docker

```bash
# Pull the latest image
docker pull yourusername/epitopegraph:latest

# Run with GPU support
docker run --gpus all -p 8888:8888 yourusername/epitopegraph

# Run without GPU
docker run -p 8888:8888 yourusername/epitopegraph
```

### 3. From Source

```bash
# Clone the repository
git clone https://github.com/yourusername/epitopegraph.git
cd epitopegraph

# Create and activate environment
conda create -n epitopegraph-dev python=3.8
conda activate epitopegraph-dev

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Dependencies

### Core Dependencies
- numpy>=1.20.0
- pandas>=1.3.0
- torch>=2.0.0
- torch-geometric>=2.3.0
- biopython>=1.80
- requests>=2.28.0
- nglview>=3.0.0
- matplotlib>=3.5.0

### Optional Dependencies
- jupyter>=1.0.0
- ipywidgets>=8.0.0
- pytest>=7.0.0
- black>=23.0.0
- isort>=5.12.0
- flake8>=6.0.0
- mypy>=1.3.0

## CUDA Setup

For GPU acceleration, ensure you have:
1. NVIDIA GPU with CUDA support
2. CUDA Toolkit 11.7 or compatible
3. cuDNN 8.0 or compatible

Verify CUDA installation:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Troubleshooting

### Common Issues

1. **CUDA not found**
   ```bash
   # Reinstall PyTorch with CUDA
   conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
   ```

2. **Dependencies conflict**
   ```bash
   # Create fresh environment
   conda create -n epitopegraph-new python=3.8
   conda activate epitopegraph-new
   pip install epitopegraph
   ```

3. **Memory issues**
   - Reduce batch size in `predict_batch()`
   - Use CPU-only mode for large structures

### Getting Help

- Check [GitHub Issues](https://github.com/yourusername/epitopegraph/issues)
- Join [Discussions](https://github.com/yourusername/epitopegraph/discussions)
- Email: your.email@example.com

## Development Setup

For contributors:

1. Fork and clone the repository
2. Create a development environment:
   ```bash
   conda create -n epitopegraph-dev python=3.8
   conda activate epitopegraph-dev
   pip install -e ".[dev]"
   ```

3. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

4. Run tests:
   ```bash
   pytest
   ```

5. Build documentation:
   ```bash
   cd docs
   make html
   ```

## Updating

```bash
# Update using pip
pip install --upgrade epitopegraph

# Update from source
git pull origin main
pip install -e ".[dev]"
``` 
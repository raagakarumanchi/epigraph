# Contributing to EpitopeGraph

We love your input! We want to make contributing to EpitopeGraph as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

1. Fork the repo and create your branch from `main`
2. If you've added code that should be tested, add tests
3. If you've changed APIs, update the documentation
4. Ensure the test suite passes
5. Make sure your code lints
6. Issue that pull request!

## Pull Request Process

1. Update the README.md with details of changes to the interface, if applicable
2. Update the docs/ with any necessary documentation changes
3. The PR will be merged once you have the sign-off of at least one other developer
4. Make sure all CI checks pass

## Any contributions you make will be under the MIT Software License

In short, when you submit code changes, your submissions are understood to be under the same [MIT License](http://choosealicense.com/licenses/mit/) that covers the project. Feel free to contact the maintainers if that's a concern.

## Report bugs using GitHub's [issue tracker](https://github.com/yourusername/epitopegraph/issues)

We use GitHub issues to track public bugs. Report a bug by [opening a new issue](https://github.com/yourusername/epitopegraph/issues/new); it's that easy!

## Write bug reports with detail, background, and sample code

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/epitopegraph.git
cd epitopegraph
```

2. Create a new conda environment:
```bash
conda create -n epitopegraph-dev python=3.8
conda activate epitopegraph-dev
```

3. Install development dependencies:
```bash
pip install -e ".[dev]"
```

4. Install pre-commit hooks:
```bash
pre-commit install
```

## Code Style

We use:
- [Black](https://black.readthedocs.io/) for code formatting
- [isort](https://pycqa.github.io/isort/) for import sorting
- [flake8](https://flake8.pycqa.org/) for linting
- [mypy](https://mypy.readthedocs.io/) for type checking

Run the style checks:
```bash
pre-commit run --all-files
```

## Testing

We use pytest for testing. Run the test suite:
```bash
pytest
```

For coverage report:
```bash
pytest --cov=epitopegraph
```

## Documentation

We use Sphinx for documentation. Build the docs:
```bash
cd docs
make html
```

## License

By contributing, you agree that your contributions will be licensed under its MIT License. 
Development Environment Setup
============================

This guide explains how to set up a development environment for EpitopeGraph.

System Requirements
-----------------

* Python 3.8 or higher
* CUDA-capable GPU (recommended)
* Git
* Conda (recommended) or pip
* Docker (optional)

Basic Setup
----------

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/yourusername/epitopegraph.git
      cd epitopegraph

2. Create a conda environment:

   .. code-block:: bash

      conda create -n epitopegraph-dev python=3.8
      conda activate epitopegraph-dev

3. Install development dependencies:

   .. code-block:: bash

      pip install -e ".[dev]"

4. Install pre-commit hooks:

   .. code-block:: bash

      pre-commit install

Development Tools
---------------

The following tools are used in development:

* **Black**: Code formatting
* **isort**: Import sorting
* **flake8**: Code linting
* **mypy**: Type checking
* **pytest**: Testing
* **pre-commit**: Git hooks
* **Sphinx**: Documentation
* **Jupyter**: Notebooks

Running Tests
-----------

Run the test suite:

.. code-block:: bash

   pytest

Run tests with coverage:

.. code-block:: bash

   pytest --cov=epitopegraph

Generate coverage report:

.. code-block:: bash

   coverage html

Building Documentation
-------------------

1. Install documentation dependencies:

   .. code-block:: bash

      pip install -e ".[docs]"

2. Build documentation:

   .. code-block:: bash

      cd docs
      make html

3. View documentation:

   .. code-block:: bash

      python -m http.server -d _build/html

Docker Development
---------------

1. Build development image:

   .. code-block:: bash

      docker build -t epitopegraph-dev -f Dockerfile.dev .

2. Run development container:

   .. code-block:: bash

      docker run -it --gpus all \
          -v $(pwd):/app \
          -p 8888:8888 \
          epitopegraph-dev

3. Start Jupyter Lab:

   .. code-block:: bash

      jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

VSCode Setup
----------

1. Install recommended extensions:

   * Python
   * Pylance
   * Jupyter
   * GitLens
   * Docker

2. Configure settings:

   .. code-block:: json

      {
          "python.linting.enabled": true,
          "python.linting.flake8Enabled": true,
          "python.formatting.provider": "black",
          "editor.formatOnSave": true,
          "editor.codeActionsOnSave": {
              "source.organizeImports": true
          }
      }

3. Select the conda environment:

   * Press ``Ctrl+Shift+P`` (Windows/Linux) or ``Cmd+Shift+P`` (macOS)
   * Type "Python: Select Interpreter"
   * Choose the ``epitopegraph-dev`` environment

PyCharm Setup
-----------

1. Open the project in PyCharm

2. Configure the Python interpreter:
   * Go to Settings/Preferences → Project → Python Interpreter
   * Add the conda environment

3. Enable code style:
   * Go to Settings/Preferences → Editor → Code Style
   * Set Python code style to follow PEP 8
   * Enable "Optimize imports on the fly"

4. Configure run configurations:
   * Add pytest configuration
   * Add documentation build configuration
   * Add Jupyter notebook configuration

Common Issues
-----------

1. CUDA Issues
   ~~~~~~~~~~~

   If you encounter CUDA-related errors:

   * Verify CUDA installation: ``nvidia-smi``
   * Check PyTorch CUDA support: ``torch.cuda.is_available()``
   * Ensure CUDA version matches PyTorch requirements

2. Memory Issues
   ~~~~~~~~~~~~

   For out-of-memory errors:

   * Reduce batch size
   * Use gradient checkpointing
   * Enable mixed precision training
   * Use CPU for small models

3. Import Errors
   ~~~~~~~~~~~~

   If you see import errors:

   * Verify environment activation
   * Check PYTHONPATH
   * Reinstall package in development mode
   * Clear Python cache: ``find . -name "*.pyc" -delete``

4. Test Failures
   ~~~~~~~~~~~~

   For test failures:

   * Run tests in verbose mode: ``pytest -v``
   * Check test data availability
   * Verify environment variables
   * Clear test cache: ``pytest --cache-clear``

Best Practices
------------

1. Code Style
   ~~~~~~~~~~

   * Follow PEP 8 guidelines
   * Use type hints
   * Write docstrings
   * Keep functions small and focused
   * Use meaningful variable names

2. Testing
   ~~~~~~~

   * Write tests for new features
   * Maintain test coverage
   * Use fixtures for common setup
   * Mock external dependencies
   * Test edge cases

3. Documentation
   ~~~~~~~~~~~~

   * Update docstrings
   * Add examples
   * Keep README current
   * Document API changes
   * Include type information

4. Version Control
   ~~~~~~~~~~~~~~

   * Use meaningful commit messages
   * Create feature branches
   * Keep commits focused
   * Review changes before committing
   * Use pre-commit hooks

5. Performance
   ~~~~~~~~~~~

   * Profile code regularly
   * Use appropriate data structures
   * Optimize critical paths
   * Cache expensive computations
   * Monitor memory usage

Additional Resources
-----------------

* `Python Development Guide <https://docs.python.org/devguide/>`_
* `PyTorch Documentation <https://pytorch.org/docs/stable/>`_
* `Sphinx Documentation <https://www.sphinx-doc.org/>`_
* `pytest Documentation <https://docs.pytest.org/>`_
* `Black Documentation <https://black.readthedocs.io/>`_
* `mypy Documentation <https://mypy.readthedocs.io/>`_ 
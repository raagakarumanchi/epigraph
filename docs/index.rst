Welcome to EpitopeGraph's documentation!
====================================

EpitopeGraph is a graph-based toolkit for conformational B-cell epitope prediction. It uses graph neural networks to predict potential epitope regions in protein structures.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   getting_started
   user_guide/index
   api_reference/index
   examples/index
   benchmarks/index
   development/index
   contributing
   changelog
   roadmap

Quick Start
----------

Install EpitopeGraph using pip:

.. code-block:: bash

   pip install epitopegraph

Or using conda:

.. code-block:: bash

   conda install -c bioconda epitopegraph

Basic usage:

.. code-block:: python

   from epitopegraph import EpitopeGraph

   # Initialize with SARS-CoV-2 spike protein
   eg = EpitopeGraph(uniprot_id="P0DTC2")

   # Predict epitopes
   scores = eg.predict_epitopes()

   # Visualize results
   eg.visualize_epitopes()

For more examples, see the :doc:`examples/index` section.

Key Features
-----------

* Automated data retrieval from UniProt and AlphaFold DB
* Graph-based representation of protein structures
* GNN-based epitope prediction
* Interactive visualization
* Comprehensive benchmark evaluation
* RESTful API for programmatic access

Documentation Sections
--------------------

* :doc:`installation` - Detailed installation instructions
* :doc:`getting_started` - Quick start guide
* :doc:`user_guide/index` - User guide and tutorials
* :doc:`api_reference/index` - API documentation
* :doc:`examples/index` - Example notebooks and use cases
* :doc:`benchmarks/index` - Benchmark datasets and evaluation
* :doc:`development/index` - Development guide
* :doc:`contributing` - Contributing guidelines
* :doc:`changelog` - Version history
* :doc:`roadmap` - Future development plans

Indices and Tables
----------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search` 
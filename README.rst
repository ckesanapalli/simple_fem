.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

    .. image:: https://api.cirrus-ci.com/github/<USER>/truss_modal.svg?branch=main
        :alt: Built Status
        :target: https://cirrus-ci.com/github/<USER>/truss_modal
    .. image:: https://readthedocs.org/projects/truss_modal/badge/?version=latest
        :alt: ReadTheDocs
        :target: https://truss_modal.readthedocs.io/en/stable/
    .. image:: https://img.shields.io/coveralls/github/<USER>/truss_modal/main.svg
        :alt: Coveralls
        :target: https://coveralls.io/r/<USER>/truss_modal
    .. image:: https://img.shields.io/pypi/v/truss_modal.svg
        :alt: PyPI-Server
        :target: https://pypi.org/project/truss_modal/
    .. image:: https://img.shields.io/conda/vn/conda-forge/truss_modal.svg
        :alt: Conda-Forge
        :target: https://anaconda.org/conda-forge/truss_modal
    .. image:: https://pepy.tech/badge/truss_modal/month
        :alt: Monthly Downloads
        :target: https://pepy.tech/project/truss_modal
    .. image:: https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter
        :alt: Twitter
        :target: https://twitter.com/truss_modal

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

|

===========
truss_modal
===========


The module takes nodes and element DataFrames of truss systems and returns global matrices and modal parameters of the truss. 


.. _pyscaffold-notes:

Making Changes & Contributing
=============================

This project uses `pre-commit`_, please make sure to install it before making any
changes::

    pip install pre-commit
    cd truss_modal
    pre-commit install

It is a good idea to update the hooks to the latest version::

    pre-commit autoupdate

Don't forget to tell your contributors to also install and use pre-commit.

.. _pre-commit: https://pre-commit.com/

Note
====

This project has been set up using PyScaffold 4.2.1. For details and usage
information on PyScaffold see https://pyscaffold.org/.

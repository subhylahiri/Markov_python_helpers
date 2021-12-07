"""Tools for working with Markov processes and drawing them as graphs.

The module `markov` contains tools fot wotking with Markov transsition
matrices in `numpy`, including indexing and parameterising routines.

The module `graphs` contains tools for representing Markov processes as
graphs using `networkx`, and plotting them in `matplotlib`.

This package assumes probability distributions are represented by row vectors,
so :math:`Q_{ij}` is the transition rate from :math:`i` to :math:`j`.
"""
from . import markov, graphs, _options, _utilities
__all__ = [
   "markov",
   "graphs",
]

"""Tools for working with Markov processes and drawing them as graphs.

Notes
-----
This package assumes probability distributions are represented by row vectors,
so :math:`Q_{ij}` is the transition rate from :math:`i` to :math:`j`.
"""
from . import markov, graphs, _options, _utilities
__all__ = [
   "markov",
   "graphs",
]

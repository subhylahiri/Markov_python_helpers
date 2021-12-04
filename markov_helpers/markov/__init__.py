"""Tools for working with and parameterising Markov processes

Notes
-----
This package assumes probability distributions are represented by row vectors,
so :math:`Q_{ij}` is the transition rate from :math:`i` to :math:`j`.

.. autosummary::
   :toctree: markov_helpers
   :recursive:

   indices
   params
"""
from . import indices, params
from .options import TopologyOptions
from ._helpers import (stochastify_c, stochastify_d, stochastify_pd,
                       unstochastify_c)
from .markov import (isstochastic_c, isstochastic_d, rand_trans, rand_trans_d,
                     calc_peq, calc_peq_d, sim_markov_c, sim_markov_d,
                     adjoint, mean_dwell)
__all__ = [
   "indices",
   "params",
   "stochastify_c",
   "stochastify_d",
   "stochastify_pd",
   "unstochastify_c",
   "isstochastic_c",
   "isstochastic_d",
   "rand_trans",
   "rand_trans_d",
   "calc_peq",
   "calc_peq_d",
   "sim_markov_c",
   "sim_markov_d",
   "adjoint",
   "mean_dwell",
   "TopologyOptions",
]

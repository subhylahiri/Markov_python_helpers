"""Utilities for working with and parameterising Markov processes


.. autosummary::
   :toctree: markov_helpers/markov
   :recursive:

   indices
   params
"""
from . import indices, params
from .options import TopologyOptions
from ._helpers import stochastify_c, stochastify_d, unstochastify_c, stochastify_pd
from .markov import (adjoint, calc_peq, calc_peq_d, isstochastic_c,
                     isstochastic_d, mean_dwell, rand_trans, rand_trans_d,
                     sim_markov_c, sim_markov_d)
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
   "adjoint",
   "calc_peq",
   "calc_peq_d",
   "mean_dwell",
   "sim_markov_c",
   "sim_markov_d",
   "TopologyOptions",
]

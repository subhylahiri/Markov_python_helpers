# -*- coding: utf-8 -*-
"""Utilities for Markov processes

Notes
-----
This package assumes probability distributions are represented by row vectors,
so :math:`Q_{ij}` is the transition rate from :math:`i` to :math:`j`.

:noindex:
"""
from typing import Optional, Tuple

import numpy as np
import scipy.linalg as sla

from ._helpers import num_param, stochastify_c, stochastify_d, stochastify_pd
from .params import params_to_mat

RNG: np.random.Generator = np.random.default_rng()

__all__ = [
   "stochastify_c",
   "stochastify_d",
   'stochastify_pd',
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
]
# =============================================================================


def _allfinite(*arrays) -> bool:
    """Check if all array elements are finite

    Returns `True` if no element of any array is `nan` or `inf`.
    """
    return all(np.isfinite(arr).all() for arr in arrays)


def _anyclose(first, second, *args, **kwds) -> bool:
    """Are any elements close?

    Like numpy.allclose but with any instead of all.
    """
    return np.isclose(first, second, *args, **kwds).any()


def _tri_low_rank(array, *args, **kwds):
    """Check for low rank triangular matrix

    Returns `True` if any diagonal element is close to 0.
    Does not check if the array is triangular. It can be used on the 'raw'
    forms of lu/qr factors.
    """
    return _anyclose(np.diagonal(array), 0., *args, **kwds)


def _ravelaxes(arr: np.ndarray, start: int = 0, stop: Optional[int] = None
             ) -> np.ndarray:
    """Partial flattening.

    Flattens those axes in the range [start:stop).

    Parameters
    ----------
    arr : np.ndarray (...,L,M,N,...,P,Q,R,...)
        Array to be partially flattened.
    start : int, optional, default: 0
        First axis of group to be flattened.
    stop : int or None, optional, default: None
        First axis *after* group to be flattened. Goes to end if it is None.

    Returns
    -------
    new_arr : np.ndarray (...,L,M*N*...*P*Q,R,...)
        Partially flattened array.

    Raises
    ------
    ValueError
        If `start > stop`.
    """
    if stop is None:
        stop = arr.ndim
    newshape = arr.shape[:start] + (-1,) + arr.shape[stop:]
    if len(newshape) > arr.ndim + 1:
        raise ValueError(f"start={start} > stop={stop}")
    return np.reshape(arr, newshape)


def _transpose(arr: np.ndarray) -> np.ndarray:
    """Transpose last two axes.

    Transposing last two axes fits better with `numpy.linalg`'s broadcasting,
    which treats multi-dim arrays as stacks of matrices.

    Parameters
    ----------
    arr : np.ndarray, (..., M, N)

    Returns
    -------
    transposed : np.ndarray, (..., N, M)
    """
    if arr.ndim < 2:
        return arr
    return arr.swapaxes(-2, -1)


def _col(arr: np.ndarray) -> np.ndarray:
    """Treat multi-dim array as a stack of column vectors.

    Achieves this by inserting a singleton dimension in last slot.
    You'll have an extra singleton after any linear algebra operation from the
    left.

    Parameters
    ----------
    arr : np.ndarray, (..., N)

    Returns
    -------
    expanded : np.ndarray, (..., N, 1)
    """
    return np.expand_dims(arr, -1)


def _row(arr: np.ndarray) -> np.ndarray:
    """Treat multi-dim array as a stack of row vectors.

    Achieves this by inserting a singleton dimension in second-to-last slot.
    You'll have an extra singleton after any linear algebra operation from the
    right.

    Parameters
    ----------
    arr : np.ndarray, (..., N)

    Returns
    -------
    expanded : np.ndarray, (..., 1, N)
    """
    return np.expand_dims(arr, -2)


# =============================================================================


def isstochastic_c(mat: np.ndarray, thresh: float = 1e-5) -> bool:
    """Are row sums zero?
    """
    nonneg = _ravelaxes(mat, -2) >= -thresh
    nonneg[..., ::mat.shape[-1]+1] = True
    return nonneg.all() and (np.fabs(mat.sum(axis=-1)) < thresh).all()


def isstochastic_d(mat: np.ndarray, thresh: float = 1e-5) -> bool:
    """Are row sums one?
    """
    return (np.fabs(mat.sum(axis=-1) - 1) < thresh).all() and (mat >= 0).all()


def rand_trans(nst: int, npl: int = 1, sparsity: float = 1.,
               rng: np.random.Generator = RNG, **kwds) -> np.ndarray:
    """
    Make a random transition matrix (continuous time).

    Parameters
    ----------
    n : int
        total number of states
    npl : int
        number of matrices
    sparsity : float, optional
        sparsity, by default 1

    Returns
    -------
    mat : np.ndarray
        transition matrix
    """
    params = rng.random((npl, num_param(nst, **kwds)))
    if sparsity < 1.:
        ind = rng.random(params.shape)
        params[ind > sparsity] = 0.
    return params_to_mat(params, **kwds)


def rand_trans_d(nst: int, npl: int = 1, sparsity: float = 1.,
                 rng: np.random.Generator = RNG, **kwds) -> np.ndarray:
    """
    Make a random transition matrix (discrete time).

    Parameters
    ----------
    n : int
        total number of states
    npl : int
        number of matrices
    sparsity : float, optional
        sparsity, by default 1

    Returns
    -------
    mat : np.ndarray
        transition matrix
    """
    if any(kwds.get(opt, False) for opt in ('uniform', 'serial', 'ring')):
        trans = rand_trans(nst, npl, sparsity, rng, **kwds)
        stochastify_pd(trans)
        return trans
    trans = rng.random((npl, nst, nst)).squeeze()
    stochastify_d(trans)
    return trans


def calc_peq(rates: np.ndarray,
             luf: bool = False) -> Tuple[np.ndarray, Tuple[np.ndarray, ...]]:
    """Calculate steady state distribution.

    Parameters
    ----------
    rates : np.ndarray (...,n,n) or tuple(np.ndarray) ((...,n,n), (...,n,))
        Continuous time stochastic matrix or LU factors of inverse fundamental.
    luf : bool, optional
        Return LU factorisation of inverse transpose fundamental matrix as
        well? Will not broadcast if True. Default: False.

    Returns
    -------
    peq : np.ndarray (...,n,)
        Steady-state distribution.
    (z_lu, ipv) : tuple(np.ndarray) ((...,n,n),(...,n,))
        LU factors of inverse transposed fundamental matrix.
    """
    if isinstance(rates, tuple):
        z_lu, ipv = [np.asanyarray(r) for r in rates]
        evc = np.ones(z_lu.shape[0])
    else:
        rates = np.asanyarray(rates)
        evc = np.ones(rates.shape[0])
        fund_inv = np.ones_like(rates) - rates
        if not luf:
            return np.linalg.solve(_transpose(fund_inv), evc)
        zlu, ipv = sla.lu_factor(_transpose(fund_inv))
    # check for singular matrix
    if _allfinite(z_lu) and not _tri_low_rank(z_lu):
        peq = sla.lu_solve((zlu, ipv), evc)
    else:
        peq = np.full_like(evc, np.nan)
    if luf:
        return peq, (z_lu, ipv)
    return peq


def calc_peq_d(jump: np.ndarray,
               luf: bool = False) -> Tuple[np.ndarray, Tuple[np.ndarray, ...]]:
    """Calculate steady state distribution.

    Parameters
    ----------
    jump : np.ndarray (...,n,n) or tuple(np.ndarray) ((...,n,n), (...,n,))
        Discrete time stochastic matrix or LU factors of inverse fundamental.
    luf : bool, optional
        Return LU factorisation of inverse fundamental as well? default: False

    Returns
    -------
    peq : np.ndarray (...,n,)
        Steady-state distribution.
    (z_lu, ipv) : tuple(np.ndarray) ((...,n,n),(...,n,))
        LU factors of inverse fundamental matrix.
    """
    if isinstance(jump, tuple):
        return calc_peq(jump, luf)
    return calc_peq(jump - np.eye(jump.shape[-1]), luf)


def adjoint(tensor: np.ndarray, measure: np.ndarray) -> np.ndarray:
    """Adjoint with respect to L2 inner product with measure

    Parameters
    ----------
    tensor : np.ndarray (...,n,n) or (...,1,n) or (...,n,1)
        The matrix/row/column vector to be adjointed.
    measure : np.ndarray (...,n)
        The measure for the inner-product wrt which we adjoint

    Returns
    -------
    tensor : np.ndarray (...,n,n) or (...,n,1) or (...,1,n)
        The adjoint matrix/column/row vector.
    """
    adj = _transpose(tensor.copy())
    if adj.shape[-1] == 1:  # row -> col
        adj /= _col(measure)
    elif adj.shape[-2] == 1:  # col -> row
        adj *= _row(measure)
    else:  # mat -> mat
        adj *= _row(measure) / _col(measure)
    return adj


def mean_dwell(rates: np.ndarray, peq: Optional[np.ndarray] = None) -> float:
    """Mean time spent in any state.

    Parameters
    ----------
    rates : np.ndarray (n,n)
        Continuous time stochastic matrix.
    peq : np.ndarray (n,), optional
        Steady-state distribution, default: calculate frm `rates`.
    """
    if peq is None:
        peq = calc_peq(rates, False)
    dwell = -1. / np.diagonal(rates)
    return 1. / (peq / dwell).sum()


def sim_markov_d(jump: np.ndarray, peq: Optional[np.ndarray] = None,
                 num_jump: int = 10, rng: np.random.Generator = RNG
                 ) -> np.ndarray:
    """Simulate Markov process trajectory.

    Parameters
    ----------
    jump : np.ndarray (n,n)
        Discrete time stochastic matrix.
    peq : np.ndarray (n,), optional
        Initial-state distribution, default: use steady-state.
    num_jump : int, optional, default: 10
        Stop after this many jumps.

    Returns
    -------
    states : np.ndarray (w,)
        Vector of states visited.
    """
    jump = np.asanyarray(jump)
    if peq is None:
        peq = calc_peq_d(jump, False)

    state_inds = np.arange(len(peq))
    states_from = np.array([rng.choice(state_inds, size=num_jump-1, p=p)
                            for p in jump])
    states = np.empty(num_jump)
    states[0] = rng.choice(state_inds, p=peq)
    for num in range(num_jump-1):
        states[num+1] = states_from[states[num], num]
    return states


def sim_markov_c(rates: np.ndarray, peq: Optional[np.ndarray] = None,
                 num_jump: Optional[int] = None,
                 max_time: Optional[float] = None,
                 rng: np.random.Generator = RNG) -> Tuple[np.ndarray, ...]:
    """Simulate Markov process trajectory.

    Parameters
    ----------
    rates : np.ndarray (n,n)
        Continuous time stochastic matrix.
    peq : np.ndarray (n,), optional
        Initial-state distribution, default: use steady-state.
    num_jump : int, optional, default: None
        Stop after this many jumps.
    max_time : float, optional, default: None
        Stop after this much time.

    Returns
    -------
    states : np.ndarray (w,)
        Vector of states visited.
    dwells : np.ndarray (w,)
        Time spent in each state.
    """
    rates = np.asanyarray(rates)
    if peq is None:
        peq = calc_peq(rates, False)
    num_states = len(peq)
    dwell = -1. / np.diagonal(rates)
    jump = rates * dwell[..., None]
    jump[np.diag_indices(num_states)] = 0.

    est_num = num_jump
    if num_jump is None:
        if max_time is None:
            raise ValueError("Must specify either num_jump or max_time")
        est_num = int(5 * max_time / mean_dwell(rates, peq))
    if max_time is None:
        max_time = np.inf
    est_num = max(est_num, 1)

    dwells_from = - dwell[..., None] * np.log(rng.random(est_num))
    states = sim_markov_d(jump, peq, est_num, rng)
    dwells = dwells_from[states, np.arange(est_num)]

    states, dwells = states[slice(num_jump)], dwells[slice(num_jump)]
    cum_dwell = np.cumsum(dwells)
    mask = cum_dwell < max_time
    if not mask[-1]:
        ind = np.nonzero(~mask)[0][0]
        mask[ind] = True
        dwells[ind] -= cum_dwell[ind] - max_time
    states, dwells = states[mask], dwells[mask]

    return states, dwells

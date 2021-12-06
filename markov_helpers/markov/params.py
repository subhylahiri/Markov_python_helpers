# -*- coding: utf-8 -*-
"""Utilities for parameterising Markov processes

For general topology, the parameters are:
    mat_01, mat_02, ..., mat_0n-1, mat10, mat_12, ..., mat_n-2,n-1.
For a serial topology:
    mat_01, mat_12, ..., mat_n-2,n-1,
    mat_10, mat_21, ..., mat_n-1,n-2.
For a ring topology:
    mat_01, mat_12, ..., mat_n-2,n-1, mat_n-1,0,
    mat_0,n-1, mat_10, mat_21, ..., mat_n-1,n-2.

If `drn` is positive/negative, we only consider the upper/lower triangle.
If `drn == 0`, we consider both.

When `uniform`, we assume that all non-zero elements in the upper/lower
triangle are equal. When extracting `uniform` parameters we average them,
unless `grad` is `True` when we take the sum.

Notes
-----
This package assumes probability distributions are represented by row vectors,
so :math:`Q_{ij}` is the transition rate from :math:`i` to :math:`j`.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from . import _helpers as _h
from . import indices as _in
from ._helpers import (Array, Axes, AxesOrSeq, IntOrSeq, Sized,
                       mat_type_siz, num_param, num_state)
from ._to_matrix import (cascade_params_to_mat, gen_params_to_mat, matify,
                         params_to_mat, ring_params_to_mat,
                         serial_params_to_mat, std_cascade_params_to_mat,
                         uni_gen_params_to_mat, uni_ring_params_to_mat,
                         uni_serial_params_to_mat)
from ._to_params import (cascade_mat_to_params, gen_mat_to_params,
                         mat_to_params, paramify, ring_mat_to_params,
                         serial_mat_to_params, std_cascade_mat_to_params,
                         uni_gen_mat_to_params, uni_ring_mat_to_params,
                         uni_serial_mat_to_params)

__all__ = [
    "num_param",
    "num_state",
    "matify",
    "paramify",
    "params_to_mat",
    "mat_to_params",
    "mat_update_params",
    "mat_type_val",
    "mat_type_siz",
    "mat_type_dict",
    "gen_params_to_mat",
    "ring_params_to_mat",
    "serial_params_to_mat",
    "cascade_params_to_mat",
    "uni_gen_params_to_mat",
    "uni_ring_params_to_mat",
    "uni_serial_params_to_mat",
    "std_cascade_params_to_mat",
    "gen_mat_to_params",
    "ring_mat_to_params",
    "serial_mat_to_params",
    "cascade_mat_to_params",
    "uni_gen_mat_to_params",
    "uni_ring_mat_to_params",
    "uni_serial_mat_to_params",
    "std_cascade_mat_to_params",
]
# =============================================================================
# Counts & types
# =============================================================================


def mat_type_val(mat: np.ndarray, axes: Axes = (-2, -1), **kwds
                 ) -> Tuple[bool, bool, int, bool]:
    """Is process (uniform) ring/serial/... from elements

    Parameters
    ----------
    mat : ndarray (...,n,n)
        Continuous time stochastic matrix.
    axes : Tuple[int, int] or None
        Axes to treat as (from, to) axes, by default: (-2, -1)

    Returns
    -------
    serial : bool
        Is the rate vector meant for `serial_params_to_mat` or
        `gen_params_to_mat`?
    ring : bool
        Is the rate vector meant for `ring_params_to_mat` or
        `gen_params_to_mat`?
    drn: int
        If nonzero, only include transitions in direction `i -> i + sgn(drn)`.
    uniform : bool
        Is the matrix from `*_params_to_mat` or `uni_*_params_to_mat`?
        * = gen, serial or ring.
    """
    nst = mat.shape[axes[-1]]
    mat = np.moveaxis(mat, axes, (-2, -1))
    mat = mat[(0,) * (mat.ndim - 2)].ravel()
    rng_inds = _in.ring_inds(nst, 0)
    gen_inds = np.setdiff1d(_in.offdiag_inds(nst, 0), rng_inds, True)
    rng_inds = np.setdiff1d(rng_inds, _in.serial_inds(nst, 0), True)
    ring = np.allclose(mat[gen_inds], 0, **kwds)
    serial = ring and np.allclose(mat[rng_inds], 0, **kwds)
    ring &= not serial
    if np.allclose(mat[_in.offdiag_inds(nst, -1)], 0, **kwds):
        drn = 1
    elif np.allclose(mat[_in.offdiag_inds(nst, 1)], 0, **kwds):
        drn = -1
    else:
        drn = 0
    ifun = _in.ind_fun(serial, ring)
    uniform = np.allclose([np.std(mat[ifun(nst, 1)]),
                           np.std(mat[ifun(nst, -1)])], 0)
    return serial, ring, drn, uniform


def mat_type_dict(params: Sized, states: Sized, **kwds) -> Tuple[bool, ...]:
    """Is it a (uniform) ring

    Parameters
    ----------
    params : int or ndarray (np,)
        Number of rate parameters, or vector of rates.
    states : int or ndarray (n,...)
        Number of states, or array over states.

    Returns
    -------
    dict containing:
        serial : bool
            Is the rate vector meant for `serial_params_to_mat` or
            `gen_params_to_mat`?
        ring : bool
            Is the rate vector meant for `ring_params_to_mat` or
            `gen_params_to_mat`?
        drn: bool
            If nonzero, only include transitions in direction `i->i+sgn(drn)`.
            Can only determine `|drn|`, not its sign.
        uniform : bool
            Is the rate vector for `*_params_to_mat` or `uni_*_params_to_mat`?
            * = general, serial or ring.
    """
    serial, ring, drn, uniform = mat_type_siz(params, states, **kwds)
    if uniform and isinstance(states, np.ndarray):
        bad_keys = set(kwds) - {'axes', 'rtol', 'atol', 'equal_nan'}
        for k in bad_keys:
            kwds.pop(k)
        serial, ring, drn, uniform = mat_type_val(states, **kwds)
    return {'serial': serial, 'ring': ring, 'drn': drn, 'uniform': uniform}


# =============================================================================
# Update matrix in-place from parameters
# =============================================================================


def mat_update_params(mat: Array, params: np.ndarray, *, drn: IntOrSeq = 0,
                      maxes: AxesOrSeq = (-2, -1), paxis: IntOrSeq = -1,
                      mdaxis: IntOrSeq = 0, pdaxis: IntOrSeq = 0,
                      **kwds) -> None:
    """Change independent parameters of transition matrix.

    Parameters
    ----------
    mat : ndarray (n,n)
        Continuous time stochastic matrix.
    params : ndarray (n(n-1),) or (2(n-1),) or (2n,) or (2,) or half of them
        Vector of independent elements. For the order, see docs for `*_inds`.

    Other Parameters
    ----------------
    serial : bool, optional, default: False
        Is the rate vector meant for a model with the serial topology?
    ring : bool, optional, default: False
        Is the rate vector meant for a model with the ring topology?
    uniform : bool, optional, default: False
        Do the nonzero transition rates (in one direction) have the same value?
    drn: int, optional, default: 0
        If nonzero, only include transitions in direction `i -> i + sgn(drn)`.
    grad : bool, optional, default: True
        Is the output for a gradient (True) or a transition matrix (False).
        If True, return sum of each group of equal transitions.
        If False, return the mean.
    maxes : Tuple[int, int] or None
        Axes of `mat` to treat as (from, to) axes, by default: (-2, -1)
    paxis : int, optional
        Axis of `params` along which each set of parameters lie, by default -1.
    mdaxis : int, optional
        Axis of `mat` to broadcast non-scalar `drn` over, by default: 0
    pdaxis : int, optional
        Axis of `params` to broadcast non-scalar `drn` over, by default: 0

    Returns
    -------
    None
        modifies `mat` in place.

    See Also
    --------
    param_inds, mat_to_params
    """
    if not isinstance(paxis, int):
        keys = ('maxes', 'paxis', 'mdaxis', 'pdaxis')
        for axes in zip(maxes, paxis, mdaxis, pdaxis):
            kwds.update(zip(keys, axes))
            mat_update_params(mat, params, drn=drn, **kwds)
        return
    std_axes = (-3, -2, -1)
    if isinstance(drn, int):
        paxis, std_axes = (paxis,), std_axes[1:]
    else:
        paxis, maxes = (pdaxis, paxis), (mdaxis,) + tuple(maxes)
    stochastifier = kwds.pop('stochastifier', _h.stochastify)
    params = np.moveaxis(np.asanyarray(params), paxis, std_axes[1:])
    nmat = np.moveaxis(mat, maxes, std_axes)
    nst = nmat.shape[-1]
    if kwds.get('uniform', False):
        params = _h.uni_to_any(params, nst, **kwds)
    kwds.update(drn=drn, ravel=False)
    nmat[_in.param_subs(nst, **kwds)] = params
    stochastifier(nmat)
    if not np.may_share_memory(nmat, mat):
        mat[...] = np.moveaxis(nmat, std_axes, maxes)

"""Generate indices for parameters of Markov processes

This package assumes probability distributions are represented by row vectors,
so :math:`Q_{ij}` is the transition rate from :math:`i` to :math:`j`.
"""
from __future__ import annotations

import numpy as np

from . import _helpers as _h
from ._helpers import IndsFun, IntOrSeq, Subs, SubsFun
from .. import _utilities as _util

__all__ = [
    "param_inds",
    "param_subs",
    "offdiag_inds",
    "offdiag_subs",
    "offdiag_split_inds",
    "offdiag_split_subs",
    "ring_inds",
    "ring_subs",
    "serial_inds",
    "serial_subs",
    "cascade_inds",
    "cascade_subs",
]
# =============================================================================
# Indices of parameters
# =============================================================================


def offdiag_inds(nst: int, drn: IntOrSeq = 0, ravel: bool = True) -> np.ndarray:
    """Ravel indices of independent parameters of transition matrix.

    Parameters
    ----------
    nst : int
        Number of states, `M`.
    drn : int|Sequence[int], optional
        If nonzero, only include transitions in direction `i -> i+sgn(drn)`.
        If a sequence of length `P`, return indices for a `(P,M,M)` array.
        By default 0.
    ravel : bool, optional
        Return a ravelled array, or use first axis for different matrices if
        `drn` is a sequence. By default `True`.

    Returns
    -------
    inds : np.ndarray (M(M-1),)
        Vector of ravel indices of nonzero off-diagonal elements, in order:
        mat_01, mat_02, ..., mat_0n-1, mat10, mat_12, ..., mat_n-1,n-2.
    """
    if not isinstance(drn, int):
        return _h.bcast_inds(offdiag_inds, nst, drn, ravel)
    if drn:
        return offdiag_split_inds(nst, drn)
    # To get the next diagonal, down one, right one: nst + 1
    return np.delete(np.arange(nst**2), np.s_[::nst+1])


def offdiag_subs(nst: int, drn: IntOrSeq = 0, ravel: bool = True) -> Subs:
    """Row and column indices of independent parameters of transition matrix.

    Parameters
    ----------
    nst : int
        Number of states, `M`.
    drn : int|Sequence[int], optional
        If nonzero, only include transitions in direction `i -> i+sgn(drn)`.
        If a sequence of length `P`, return indices for a `(P,M,M)` array.
        By default 0.
    ravel : bool, optional
        Return a ravelled array, or use first axis for different matrices if
        `drn` is a sequence. By default `True`.

    Returns
    -------
    [mats : np.ndarray (PM(M-1),)
        Which transition matrix, in a `(P,M,M)` array of matrices?
        Not returned if `drn` is an `int`.]
    rows : np.ndarray (PM(M-1),)
        Vector of row indices of nonzero off-diagonal elements.
    cols : np.ndarray (PM(M-1),)
        Vector of column indices of nonzero off-diagonal elements in order:
        mat_01, mat_02, ..., mat_0n-1, mat10, mat_12, ..., mat_n-1,n-2.
    """
    if not isinstance(drn, int):
        return _h.bcast_subs(offdiag_subs, nst, drn, ravel)
    if drn:
        return offdiag_split_subs(nst, drn)
    grid = np.mgrid[:nst, :nst].reshape(2, -1)
    # To get the next diagonal, down one, right one: nst + 1
    return tuple(np.delete(grid, np.s_[::nst+1], axis=-1))


@_h.sub_fun_bcast
def offdiag_split_subs(nst: int, drn: IntOrSeq = 0, ravel: bool = True
                       ) -> Subs:
    """Row and column indices of independent parameters of transition matrix.

    Parameters
    ----------
    nst : int
        Number of states, `M`.
    drn : int|Sequence[int], optional
        If nonzero, only include transitions in direction `i -> i+sgn(drn)`.
        If a sequence of length `P`, return indices for a `(P,M,M)` array.
        By default 0.
    ravel : bool, optional
        Return a ravelled array, or use first axis for different matrices if
        `drn` is a sequence. By default `True`.

    Returns
    -------
    [mats : np.ndarray (PM(M-1),)
        Which transition matrix, in a `(P,M,M)` array of matrices?
        Not returned if `drn` is an `int`.]
    rows : np.ndarray (PM(M-1),)
        Vector of row indices of nonzero off-diagonal elements.
    cols : np.ndarray (PM(M-1),)
        Vector of column indices of nonzero off-diagonal elements in order:
        First, the upper/right triangle -
        mat_01, ..., mat_0n-1, mat_12, ..., mat_n-3n-2, mat_n-3n-1, mat_n-2n-1,
        - followed by the lower/left triangle -
        mat_10, mat_20, mat_21, ..., mat_n-2n-3, mat_n-10, ... mat_n-1n-2.
    """
    _util.dummy(ravel)
    return np.triu_indices(nst, 1) if drn > 0 else np.tril_indices(nst, -1)


def ring_inds(nst: int, drn: IntOrSeq = 0, ravel: bool = True) -> np.ndarray:
    """Ravel indices of non-zero elements of ring transition matrix.

    Parameters
    ----------
    nst : int
        Number of states, `M`.
    drn : int|Sequence[int], optional
        If nonzero, only include transitions in direction `i -> i+sgn(drn)`.
        If a sequence of length `P`, return indices for a `(P,M,M)` array.
        By default 0.
    ravel : bool, optional
        Return a ravelled array, or use first axis for different matrices if
        `drn` is a sequence. By default `True`.

    Returns
    -------
    inds : np.ndarray (2M,)
        Vector of ravel indices of nonzero off-diagonal elements, in order:
        mat_01, mat_12, ..., mat_n-2,n-1, mat_n-1,0,
        mat_0,n-1, mat_10, mat_21, ..., mat_n-1,n-2.
    """
    if not isinstance(drn, int):
        return _h.bcast_inds(ring_inds, nst, drn, ravel)
    if drn > 0:
        return np.r_[1:nst**2:nst+1, nst*(nst-1)]
    if drn < 0:
        return np.r_[nst-1, 1:nst**2:nst+1]
    return _h.stack_inds(ring_inds, nst)


@_h.sub_fun_bcast
def ring_subs(nst: int, drn: IntOrSeq = 0, ravel: bool = True) -> Subs:
    """Row and column indices of non-zero elements of ring transition matrix.

    Parameters
    ----------
    nst : int
        Number of states, `M`.
    drn : int|Sequence[int], optional
        If nonzero, only include transitions in direction `i -> i+sgn(drn)`.
        If a sequence of length `P`, return indices for a `(P,M,M)` array.
        By default 0.
    ravel : bool, optional
        Return a ravelled array, or use first axis for different matrices if
        `drn` is a sequence. By default `True`.

    Returns
    -------
    [mats : np.ndarray (2PM,)
        Which transition matrix, in a `(P,M,M)` array of matrices?
        Not returned if `drn` is an `int`.]
    rows : np.ndarray (2PM,)
        Vector of row indices of nonzero off-diagonal elements.
    cols : np.ndarray (2PM,)
        Vector of column indices of nonzero off-diagonal elements in order:
        mat_01, mat_12, ..., mat_n-2,n-1, mat_n-1,0,
        mat_0,n-1, mat_10, mat_21, ..., mat_n-1,n-2.
    """
    _util.dummy(ravel)
    rows = np.arange(nst)
    return rows, np.roll(rows, -drn // abs(drn))


def serial_inds(nst: int, drn: IntOrSeq = 0, ravel: bool = True) -> np.ndarray:
    """Ravel indices of non-zero elements of serial transition matrix.

    Parameters
    ----------
    nst : int
        Number of states, `M`.
    drn : int|Sequence[int], optional
        If nonzero, only include transitions in direction `i -> i+sgn(drn)`.
        If a sequence of length `P`, return indices for a `(P,M,M)` array.
        By default 0.
    ravel : bool, optional
        Return a ravelled array, or use first axis for different matrices if
        `drn` is a sequence. By default `True`.

    Returns
    -------
    inds : np.ndarray (2(M-1),)
        Vector of ravel indices of nonzero off-diagonal elements, in order:
        mat_01, mat_12, ..., mat_n-2,n-1,
        mat_10, mat_21, ..., mat_n-1,n-2.
    """
    if not isinstance(drn, int):
        return _h.bcast_inds(serial_inds, nst, drn, ravel)
    if drn > 0:
        return np.arange(1, nst**2, nst+1)
    if drn < 0:
        return np.arange(nst, nst**2, nst+1)
    return _h.stack_inds(serial_inds, nst)


@_h.sub_fun_bcast
def serial_subs(nst: int, drn: IntOrSeq = 0, ravel: bool = True) -> Subs:
    """Row and column indices of non-zero elements of serial transition matrix.

    Parameters
    ----------
    nst : int
        Number of states, `M`.
    drn : int|Sequence[int], optional
        If nonzero, only include transitions in direction `i -> i+sgn(drn)`.
        If a sequence of length `P`, return indices for a `(P,M,M)` array.
        By default 0.
    ravel : bool, optional
        Return a ravelled array, or use first axis for different matrices if
        `drn` is a sequence. By default `True`.

    Returns
    -------
    [mats : np.ndarray (2P(M-1),)
        Which transition matrix, in a `(P,M,M)` array of matrices?
        Not returned if `drn` is an `int`.]
    rows : np.ndarray (2P(M-1),)
        Vector of row indices of nonzero off-diagonal elements.
    cols : np.ndarray (2P(M-1),)
        Vector of column indices of nonzero off-diagonal elements in order:
        mat_01, mat_12, ..., mat_n-2,n-1,
        mat_10, mat_21, ..., mat_n-1,n-2.
    """
    _util.dummy(ravel)
    return (np.arange(nst - 1), np.arange(1, nst))[::drn // abs(drn)]


@_h.sub_fun_bcast
def cascade_subs(nst: int, drn: IntOrSeq = 0, ravel: bool = True) -> Subs:
    """Row and column indices of non-zero elements of cascade transition matrix

    Parameters
    ----------
    nst : int
        Number of states, `M == 2n`.
    drn : int|Sequence[int], optional
        If nonzero, only include transitions in direction `i -> i+sgn(drn)`.
        If a sequence of length `P`, return indices for a `(P,M,M)` array.
        By default 0.
    ravel : bool, optional
        Return a ravelled array, or use first axis for different matrices if
        `drn` is a sequence. By default `True`.

    Returns
    -------
    [mats : np.ndarray (2P(M-1),)
        Which transition matrix, in a `(P,M,M)` array of matrices?
        Not returned if `drn` is an `int`.]
    rows : np.ndarray (2P(M-1),)
        Vector of row indices of nonzero off-diagonal elements.
    cols : np.ndarray (2P(M-1),)
        Vector of column indices of nonzero off-diagonal elements in order:
        mat_0n, mat_1n, ..., mat_n-1,n,
        mat_n,n+1, mat_n+1,n+2, ... mat_2n-2,2n-1,
        mat_2n-1,n-1, ..., mat_n+1,n-1, mat_n,n-1,
        mat_n-1,n-2, ..., mat_21, mat_10.
    """
    _util.dummy(ravel)
    npt = nst // 2
    rows = np.arange(nst-1)
    cols = np.r_[[npt] * npt, rows[npt:] + 1]
    return (rows, cols) if (drn > 0) else (nst - 1 - rows, nst - 1 - cols)


def ind_fun(serial: bool, ring: bool, uniform: bool = False, **kws) -> IndsFun:
    """Which ravel index function to use.

    Parameters
    ----------
    serial : bool, optional, default: False
        Is the rate vector meant for a model with the serial topology?
    ring : bool, optional, default: False
        Is the rate vector meant for a model with the ring topology?
    uniform : bool, optional, default: False
        Do the nonzero transition rates (in one direction) have the same value?

    Returns
    -------
    ifun : Callable[[int, int] -> ndarray[int]]
        Function that computes ravelled indices of nonzero elements from
        number of states and direction.
    """
    if serial:
        return serial_inds
    if ring:
        return ring_inds
    if kws.get('cascade', False):
        return cascade_inds
    if uniform:
        return offdiag_split_inds
    return offdiag_inds


def sub_fun(serial: bool, ring: bool, uniform: bool = False, **kws) -> SubsFun:
    """Which Row and column index function to use.

    Parameters
    ----------
    serial : bool, optional, default: False
        Is the rate vector meant for a model with the serial topology?
    ring : bool, optional, default: False
        Is the rate vector meant for a model with the ring topology?
    uniform : bool, optional, default: False
        Do the nonzero transition rates (in one direction) have the same value?

    Returns
    -------
    sfun : Callable[[int, int] -> Tuple[ndarray[int], ndarray[int]]]
        Function that computes unravelled indices of nonzero elements from
        number of states and direction.
    """
    if serial:
        return serial_subs
    if ring:
        return ring_subs
    if kws.get('cascade', False):
        return cascade_subs
    if uniform:
        return offdiag_split_subs
    return offdiag_subs


def param_inds(nst: int, *, serial: bool = False, ring: bool = False,
               uniform: bool = False, drn: IntOrSeq = 0, ravel: bool = True,
               **kwds) -> np.ndarray:
    """Ravel indices of independent parameters of transition matrix.

    Parameters
    ----------
    nst : int
        Number of states, `M`.
    serial : bool, optional, default: False
        Is the rate vector meant for a model with the serial topology?
    ring : bool, optional, default: False
        Is the rate vector meant for a model with the ring topology?
    uniform : bool, optional, default: False
        Do the nonzero transition rates (in one direction) have the same value?
    drn : int|Sequence[int], optional
        If nonzero, only include transitions in direction `i -> i+sgn(drn)`.
        If a sequence of length `P`, return indices for a `(P,M,M)` array.
        By default 0.
    ravel : bool, optional
        Return a ravelled array, or use first axis for different matrices if
        `drn` is a sequence. By default `True`.

    Returns
    -------
    inds : np.ndarray (k,), k in (M(M-1), 2(M-1), 2M, 2)
        Indices of independent elements. For the order, see docs for `*_inds`.
    """
    return ind_fun(serial, ring, uniform, **kwds)(nst, drn, ravel)


def param_subs(nst: int, *, serial: bool = False, ring: bool = False,
               uniform: bool = False, drn: IntOrSeq = 0, ravel: bool = True,
               **kwds) -> Subs:
    """Row and column indices of independent parameters of transition matrix.

    Parameters
    ----------
    nst : int
        Number of states, `M`.
    serial : bool, optional, default: False
        Is the rate vector meant for a model with the serial topology?
    ring : bool, optional, default: False
        Is the rate vector meant for a model with the ring topology?
    uniform : bool, optional, default: False
        Do the nonzero transition rates (in one direction) have the same value?
    drn : int|Sequence[int], optional
        If nonzero, only include transitions in direction `i -> i+sgn(drn)`.
        If a sequence of length `P`, return indices for a `(P,M,M)` array.
        By default 0.
    ravel : bool, optional
        Return a ravelled array, or use first axis for different matrices if
        `drn` is a sequence. By default `True`.

    Returns
    -------
    [mats : np.ndarray (PQ,)
        Which transition matrix, in a `(P,M,M)` array of matrices?
        Not returned if `drn` is an `int`.]
    rows : np.ndarray
        Vector of row indices of nonzero off-diagonal elements.
    cols : np.ndarray
        Vector of column indices of nonzero off-diagonal elements.
    For the order, see docs for `*_subs`.
    """
    return sub_fun(serial, ring, uniform, **kwds)(nst, drn, ravel)


def _unravel_ind_fun(func: IndsFun) -> SubsFun:
    """Convert a function that returns ravelled indices to one that returns
    unravelled indices.

    Parameters
    ----------
    func : Callable[[int, int], np.ndarray]
        Function that returns ravelled indices.

    Returns
    -------
    new_func : Callable[[int, int], Tuple[np.ndarray, np.ndarray]]
        Function that returns unravelled indices.
    """
    def new_fun(nst: int, drn: IntOrSeq = 0, ravel: bool = True) -> Subs:
        """Row and column indices of \\1 of \\2 transition matrix.

        Parameters
        ----------
        nst : int
            Number of states, `M`.
        drn : int|Sequence[int], optional
            If nonzero only include transitions in direction `i -> i+sgn(drn)`.
            If it is a sequence of length `P`, return the indices for a
            `(P,M,M)` array of matrices. By default 0.
        ravel : bool, optional
            Return a ravelled array, or use first axis for different matrices
            if `drn` is a sequence. By default `True`.

        Returns
        -------
        [mats : np.ndarray (PQ,)
            Which transition matrix, in a `(P,M,M)` array of matrices?
            Not returned if `drn` is an `int`.]
        rows : np.ndarray (PQ,)
            Vector of row indices of independent parameters.
        cols : np.ndarray (PQ,)
            Vector of column indices of independent parameters.
        """
        shape = (nst, nst) if isinstance(drn, int) else (len(drn), nst, nst)
        inds = func(nst, drn, ravel)
        return np.unravel_index(inds, shape)

    doc_lines = func.__doc__.splitlines(keepends=True)
    new_doc_lines = new_fun.__doc__.splitlines(keepends=True)
    new_doc_lines[0] = doc_lines[0].replace("Ravel", "Row and column")
    doc_ind = func.__doc__.rfind("in order:") + 10
    line_ind = len(func.__doc__[doc_ind:].splitlines())
    new_doc_lines.extend(doc_lines[-line_ind:])

    new_fun.__doc__ = "".join(new_doc_lines) + func.__doc__[doc_ind:]
    new_fun.__name__ = func.__name__.replace('inds', 'subs')
    new_fun.__qualname__ = func.__qualname__.replace('inds', 'subs')
    return new_fun


def _ravel_sub_fun(func: SubsFun):
    """Convert a function that returns unravelled indices to one that returns
    ravelled indices.

    Parameters
    ----------
    func : Callable[[int, int], Tuple[np.ndarray, np.ndarray]]
        Function that returns unravelled indices

    Returns
    -------
    new_fun : Callable[[int, int], np.ndarray]
        Function that returns ravelled indices
    """
    def new_fun(nst: int, drn: IntOrSeq = 0, ravel: bool = True) -> np.ndarray:
        """Ravel indices of \\1 of \\2 transition matrix.

        Parameters
        ----------
        nst : int
            Number of states, `M`.
        drn : int|Sequence[int], optional, default: 0
            If nonzero only include transitions in direction `i -> i+sgn(drn)`.
            If it is a sequence of length `P`, return the subscripts for a
            `(P,M,M)` array of matrices
        ravel : bool, optional
            Return a ravelled array, or use first axis for different matrices
            if `drn` is a sequence. By default `True`.

        Returns
        -------
        inds : np.ndarray (PQ,)
            Vector of ravelled indices of independent parameters.
        """
        shape = (nst, nst) if isinstance(drn, int) else (len(drn), nst, nst)
        subs = func(nst, drn, ravel)
        return np.ravel_multi_index(subs, shape)

    doc_lines = func.__doc__.splitlines(keepends=True)
    new_doc_lines = new_fun.__doc__.splitlines(keepends=True)
    new_doc_lines[0] = doc_lines[0].replace("Row and column", "Ravel")
    doc_ind = func.__doc__.rfind("in order:") + 10
    line_ind = func.__doc__.count("\n", 0, doc_ind)
    new_doc_lines.extend(doc_lines[line_ind:])

    new_fun.__doc__ = "".join(new_doc_lines)
    new_fun.__name__ = func.__name__.replace('subs', 'inds')
    new_fun.__qualname__ = func.__qualname__.replace('subs', 'inds')
    return new_fun


# =============================================================================
# Aliases
# =============================================================================
# offdiag_subs = _unravel_ind_fun(offdiag_inds)
# offdiag_split_subs = _unravel_ind_fun(offdiag_split_inds)
# ring_subs = _unravel_ind_fun(ring_inds)
# serial_subs = _unravel_ind_fun(serial_inds)
# cascade_subs = _unravel_ind_fun(cascade_inds)
# offdiag_inds = _ravel_sub_fun(offdiag_subs)
offdiag_split_inds = _ravel_sub_fun(offdiag_split_subs)
# ring_inds = _ravel_sub_fun(ring_subs)
# serial_inds = _ravel_sub_fun(serial_subs)
cascade_inds = _ravel_sub_fun(cascade_subs)

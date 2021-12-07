# -*- coding: utf-8 -*-
"""Base classes for storing options/settings.
"""
from __future__ import annotations

import collections as _cn
import operator as _op
import typing as _ty
import re as _re

# import numpy as np
import matplotlib as mpl

from . import _helpers as _h
from .._options import Options


__all__ = [
    "TopologyOptions",
]
# =============================================================================

def _norm_str(norm: mpl.colors.Normalize) -> str:
    """string rep of Normalize"""
    return norm.__class__.__name__ + f"({norm.vmin}, {norm.vmax})"


# =============================================================================
# Class for specifying topology of parameterised synapses
# =============================================================================


# pylint: disable=too-many-ancestors
class TopologyOptions(Options, key_last=('directions', 'npl')):
    """Class that contains topology specifying options.

    The individual options can be accessed as object instance attributes
    (e.g. `obj.name`) or as dictionary items (e.g. `obj['name']`) for both
    getting and setting. You can also subscript attributes of attributes with
    dotted keys: `options['suboptions.name']`.

    Parameters
    ----------
    serial : bool, optional keyword
        Restrict to models of serial topology? By default `False`.
    ring : bool, optional keyword
        Restrict to models of ring topology? By default `False`.
    uniform : bool, optional keyword
        Restrict to models with equal rates per direction? By default `False`.
    directions: Tuple[int] (P,), optional keyword
        If nonzero, only include transitions in direction `i -> i + sgn(drn)`,
        one value for each transition matrix. By default `(0,)`.
    discrete : bool, optional keyword
        Are the transition matrices for a discrete-time Markov process?
        By default `False`.

        All parameters are optional keywords. Any dictionary passed as
        positional parameters will be popped for the relevant items. Keyword
        parameters must be valid keys, otherwise a `KeyError` is raised.
    """
    serial: bool = False
    """Restrict to models of serial topology?"""
    ring: bool = False
    """Restrict to models of ring topology?"""
    uniform: bool = False
    """Restrict to models with equal rates per direction?"""
    directions: _ty.Tuple[int, ...] = (0,)
    """If nonzero, only use transitions in direction `i -> i + sgn(drn)`."""
    discrete: bool = False
    """Are the transition matrices for a discrete-time Markov process?"""

    def __init__(self, *args, **kwds) -> None:
        self.serial = self.serial
        self.ring = self.ring
        self.uniform = self.uniform
        self.directions = self.directions
        self.discrete = self.discrete
        super().__init__(*args, **kwds)
        if self.constrained and 'directions' not in kwds:
            # different default if any(serial, ring, uniform)
            self.directions = (1, -1)
            if 'npl' in kwds:
                self.npl = kwds['npl']

    def directed(self, which: _ty.Union[int, slice, None] = slice(None), **kwds
                 ) -> _ty.Dict[str, _ty.Any]:
        """Dictionary of Markov parameter options

        Parameters
        ----------
        which : int, slice, None, optional
            Which element of `self.directions` to use as the `'drn'` value,
            where `None` -> omit `'drn'` item. By default `slice(None)`

            Extra arguments are default values or unknown keys in `opts`

        Returns
        -------
        opts : Dict[str, Any]
            All options for `sl_py_tools.numpy_tricks.markov.params`.
        """
        if which is not None:
            kwds['drn'] = self.directions[which]
        kwds.update(serial=self.serial, ring=self.ring, uniform=self.uniform)
        if self.discrete:
            kwds['stochastifier'] = _h.stochastify_pd
        return kwds

    @property
    def constrained(self) -> bool:
        """Are there any constraints on the topology?
        """
        return any((self.serial, self.ring, self.uniform) + self.directions)

    @constrained.setter
    def constrained(self, value: _ty.Optional[bool]) -> None:
        """Remove all constraints on topology by setting it `False`.

        Does nothing if `value` is `None`. Raises `ValueError if it is `True`.
        """
        if value is None:
            return
        if value:
            raise ValueError("Cannot directly set `constrained=True`. "
                             + "Set a specific constraint instead.")
        self.serial = False
        self.ring = False
        self.uniform = False
        self.directions = (0,) * self.npl

    @property
    def npl(self) -> int:
        """Number of transition matrices
        """
        return len(self.directions)

    @npl.setter
    def npl(self, value: _ty.Optional[int]) -> None:
        """Set the number of transition matrices.

        Does nothing if `value` is `None`. Removes end elements of `directions`
        if shortening. Appends zeros if lengthening.
        """
        if value is None:
            return
        self.directions = self.directions[:value] + (0,) * (value - self.npl)


# =============================================================================
# Type hints and constants
_FIX_STR = {mpl.colors.Colormap: _op.attrgetter('name'),
            mpl.colors.Normalize: _norm_str,
            _cn.abc.Callable: _op.attrgetter('__name__')}
_LINE_SEP = _re.compile('\n {4,}')
Key = _ty.TypeVar('Key', bound=_ty.Hashable)
Val = _ty.TypeVar('Val')
Dictable = _ty.Union[_ty.Mapping[Key, Val], _ty.Iterable[_ty.Tuple[Key, Val]]]
StrDict = _ty.Dict[str, Val]
StrDictable = Dictable[str, Val]
Attrs = _ty.ClassVar[_ty.Tuple[str, ...]]

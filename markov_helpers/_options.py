# -*- coding: utf-8 -*-
"""Class for specifying topology of Markovian models
"""
from __future__ import annotations

import collections as _cn
import operator as _op
import typing as _ty
import re as _re

# import numpy as np
import matplotlib as mpl


__all__ = [
    'Options',
    'AnyOptions',
]
# =============================================================================

def _public(key: str) -> bool:
    """Is it a name of a public member?"""
    return not key.startswith('_')


def _norm_str(norm: mpl.colors.Normalize) -> str:
    """string rep of Normalize"""
    return norm.__class__.__name__ + f"({norm.vmin}, {norm.vmax})"


def _fmt_sep(format_spec: str) -> _ty.Tuple[str, str, str]:
    """helper for Options.__format__: process `format_spec`."""
    if '#' not in format_spec:
        conv, next_spec = '', format_spec
    else:
        conv, next_spec = format_spec.split('#', maxsplit=1)
    sep = ',' + next_spec if next_spec else ', '
    conv = "!" + conv if conv else conv
    return sep, conv, next_spec


def _fmt_help(key: str, val: _ty.Any, conv: str, next_spec: str) -> str:
    """helper for Options.__format__: entry for one item"""
    for cls, fun in _FIX_STR.items():
        if isinstance(val, cls):
            val = fun(val)
            break
    if conv != '!r' or _LINE_SEP.fullmatch(next_spec) is None:
        item = "{}={" + conv + "}"
        return item.format(key, val)
    val = repr(val).replace('\n', next_spec)
    return "{}={}".format(key, val)


# =============================================================================


def _sort_dict(unordered: Dictable[Key, Val], order: _ty.Sequence[Key],
              default: _ty.Optional[int] = None) -> _ty.Dict[Key, Val]:
    """Sort a dict by the order the keys appear in another list.

    Parameters
    ----------
    unordered : Dict[str, Any]
        Dictionary whose entries we want to sort.
    order : Sequence[str]
        Keys in order we want.
    default : int|None, optional
        Sort keys for items that do not appear in `order`.
        By default `None` -> `len(order)`.

    Returns
    -------
    ordered : Dict[str, Any]
        Dictionary copy whose keys are in the same order as `order`.
    """
    default = len(order) if default is None else default
    def key_fn(item: _ty.Tuple[str, _ty.Any]) -> int:
        """Key function for sorting"""
        return order.index(item[0]) if item[0] in order else default

    if isinstance(unordered, _cn.abc.Mapping):
        return dict(sorted(unordered.items(), key=key_fn))
    return _sort_dict(dict(unordered), order, default)


def _sort_ends_dict(unordered: Dictable[Key, Val],
                   to_start: _ty.Sequence[Key] = (),
                   to_end: _ty.Sequence[Key] = ()) -> _ty.Dict[Key, Val]:
    """Sort a dictionary so that some items are at the start, some at the end.

    Parameters
    ----------
    unordered : Dictable[Key, Val]
        The dictionary that needs sorting.
    to_start : Sequence[Key], optional
        The keys that must go at the start, in order. By default ().
    to_end : Sequence[Key], optional
        The keys that must go at the end, in order. By default ().

    Returns
    -------
    ordered : Dict[Key, Val]
        The sorted dictionary copy.
    """
    reordered = _sort_dict(unordered, to_start, None)
    return _sort_dict(reordered, to_end, -1)


# =============================================================================
# Base options class
# =============================================================================


# pylint: disable=too-many-ancestors
class Options(_cn.abc.MutableMapping):
    """Base class for options classes

    The individual options can be accessed as object instance attributes
    (e.g. `obj.name`) or as dictionary items (e.g. `obj['name']`) for both
    getting and setting. You can also subscript attributes of attributes with
    dotted keys: `options['suboptions.name']`.

    If an attribute's name is found in `map_attributes`, the attribute is
    updated when set rather than replaced like other attributes. This
    statement does not apply to setting as an attribute. New keys may be added
    by setting as attributes, e.g. `obj.name=val` or `setattr(obj,'name',val)`.

    Iterating and unpacking does not include properties and private
    attributes, unless their names are included in `prop_attributes`.
    If an attribute's value is only set by a default value in a type hint, and
    not set in `__init__`, it will be omitted when iterating, unpacking or
    printing. If it is both a member of `self.__dict__` and listed in
    `prop_attributes`, it will appear twice.

    Parameters
    ----------
        All parameters are optional keywords. Any dictionary passed as
        positional parameters will be popped for the relevant items. Keyword
        parameters must be valid keys, otherwise a `KeyError` is raised.

    Raises
    ------
    KeyError
        If an invalid key is used when subscripting. This does not apply to use
        as attributes (either `obj.name` or `getattr(obj, 'name')`).
    """
    _map_attributes: Attrs = ()
    _prop_attributes: Attrs = ()
    _key_last: Attrs = ()
    _key_first: Attrs = ()

    def __init_subclass__(cls,
                          map_attributes: PropNames = (),
                          prop_attributes: PropNames = (),
                          key_last: PropNames = (),
                          key_first: PropNames = (),
                          **kwds) -> None:
        cls._map_attributes += map_attributes
        cls._prop_attributes += prop_attributes
        cls._key_first += key_first
        cls._key_last += key_last
        return super().__init_subclass__(**kwds)

    def __init__(self, *args, **kwds) -> None:
        """The recommended approach to a subclass constructor is
        ```
        def __init__(self, *args, **kwds) -> None:
            self.my_attr = its_default
            self.other_attr = other_default
            ...
            self.last_attr = last_default
            order = ('my_attr', 'other_attr', ..., 'last_attr')
            args = sort_dicts(args, order, -1)
            kwds = sort_dict(kwds, order, -1)
            super().__init__(*args, **kwds)
        ```
        Mappings provided as positional arguments will be popped for the
        relevant items.
        """
        # put kwds in order
        for mapping in args:
            self.pop_my_args(mapping)
        self.update(kwds)

    def __format__(self, format_spec: str) -> str:
        """formatted string representing object.

        Parameters
        ----------
        format_spec : str
            Formating choice. If it does not contain a `'#'` it is added to
            `","` as a separator and inserted before the first member.
            When it takes the form `'x#blah'`, any non `Options` members are
            converted as `"{}={!x}".format(key, val)`. `Options` members are
            converted as "{}={:x#blah   }".format(key, val)` if blah consists
            only of a newline followed by a minimum of four spaces, or
            "{}={!x:blah}".format(key, val)` otherwise.

        Returns
        -------
        str
            String representation of object.
        """
        sep, conv, next_spec = _fmt_sep(format_spec)
        attrs = sep.join(_fmt_help(key, val, conv, next_spec)
                         for key, val in self.items())
        return type(self).__name__ + f"({sep[1:]}{attrs})"

    def __repr__(self) -> str:
        return self.__format__('r#\n    ')

    def __getitem__(self, key: str) -> _ty.Any:
        """Get an attribute"""
        if '.' in key:
            *args, attr = key.split('.')
            obj = self
            for arg in args:
                obj = getattr(obj, arg)
            return obj[attr]
        return getattr(self, key)

    def __setitem__(self, key: str, value: _ty.Any) -> None:
        """Set an existing attribute"""
        if '.' in key:
            *args, attr = key.split('.')
            obj = self
            for arg in args:
                obj = getattr(obj, arg)
            obj[attr] = value
            return
        if key in self._map_attributes:
            self[key].update(value)
        else:
            setattr(self, key, value)

    def __delitem__(self, key: str) -> None:
        if key in self._map_attributes + self._prop_attributes:
            raise TypeError(f"`del {type(self).__name__}['{key}']` disallowed")
        if '.' in key:
            *args, attr = key.split('.')
            obj = self
            for arg in args:
                obj = getattr(obj, arg)
            del obj[attr]
            return
        delattr(self, key)

    def __len__(self) -> int:
        # len(self.__dict__) + len(self.prop_attributes) includes privates.
        # tuple(self) appears to call len(self) -> infinite recursion.
        # return len(tuple(x for x in self))
        # barely any speed difference:
        count = 0
        for _ in self:
            count += 1
        return count

    def __iter__(self) -> _ty.Iterator[str]:
        yield from filter(_public, self.__dict__)
        yield from self._prop_attributes

    def update(self, __m: StrDictable = (), /, **kwargs) -> None:
        """Update from mappings/iterables"""
        # put kwds in order
        __m, kwargs = self.order_keys(__m, kwargs)
        super().update(__m, **kwargs)

    def copy(self) -> Options:
        """Get a shallow copy of the object.

        Only copies those attributes that appear when iterating.
        """
        return type(self)(**self)

    def pop_my_args(self, kwds: StrDict) -> None:
        """Pop any key from dict that can be set and use the value to set.
        """
        # put kwds in order
        sorted_kwds, = self.order_keys(kwds)
        to_pop = []
        # update what we can
        for key, val in sorted_kwds.items():
            if key in self:
                try:
                    self[key] = val
                except KeyError:
                    pass
                else:
                    to_pop.append(key)
        # pop the ones we used
        for key in to_pop:
            del kwds[key]

    @classmethod
    def order_keys(cls, *kwds: StrDict) -> _ty.List[StrDict]:
        """Sort dicts given keys in order for start and end"""
        key_first = cls._key_first + cls._map_attributes
        key_last = cls._prop_attributes + cls._key_last
        return [_sort_ends_dict(arg, key_first, key_last) for arg in kwds]
# pylint: enable=too-many-ancestors


# =============================================================================
# Fallback options class
# =============================================================================


# pylint: disable=too-many-ancestors
class AnyOptions(Options):
    """Same to `Options`, except it stores unknown keys as attributes.

    This can be used as a default place to store unknown items.
    """
    def __setitem__(self, key: str, val: _ty.Any) -> None:
        try:
            super().__setitem__(key, val)
        except KeyError:
            setattr(self, key, val)
# pylint: enable=too-many-ancestors


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
PropNames = _ty.Tuple[str, ...]
Attrs = _ty.ClassVar[PropNames]

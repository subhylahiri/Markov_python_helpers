"""Utility functions
"""
from __future__ import annotations

import typing as _ty
import itertools as _it
import collections as _cn
import functools as _ft
import abc as _abc
import types as _tys
import numbers as _nm

import math as _math

# =============================================================================
# Useful functions
# =============================================================================


def dummy(*args) -> None:  # pylint: disable=unused-argument
    """Doesn't do anything."""


def default(optional: _ty.Optional[Some], default_value: Some) -> Some:
    """Replace optional with default value if it is None

    Parameters
    ----------
    optional : Some or None
        The optional argument, where `None` indicates that the default value
        should be used instead.
    default_value : Some
        Default value for the argument, used when `optional` is `None`.

    Returns
    -------
    use_value : Some
        Either `optional`, if it is not `None` or `default_value` if it is.
    """
    return default_value if (optional is None) else optional


def default_eval(optional: _ty.Optional[Some],
                 default_fn: _ty.Callable[[], Some], *args, **kwds) -> Some:
    """Replace optional with evaluation of default function if it is None

    Parameters
    ----------
    optional : Some or None
        The optional argument, where `None` indicates that the default value
        should be used instead.
    default_fn : Callable[()->Some]
        Evaluates to default value for the argument, only evaluated and used
        when `optional` is `None`. Does not take any arguments.

    Returns
    -------
    use_value : Some
        Either `optional`, if it is not `None` or `default_fn()` if it is.
    """
    if optional is None:
        return default_fn(*args, **kwds)
    return optional


def eval_or_default(optional: _ty.Optional[Some],
                    non_default_fn: _ty.Callable[[Some], Other],
                    default_value: Other, *args, **kwds) -> Other:
    """Evaluate function on optional if it is not None

    Parameters
    ----------
    optional : Some or None
        The optional argument, where `None` indicates that the default value
        should be used instead.
    non_default_fn : Callable[(Some)->Other]
        Evaluated on `optional`if it is not `None`.
    default_value : Other
        Default value for the argument, used when `optional` is `None`.

    Returns
    -------
    use_value : Other
        Either `non_default_fn(optional)`, if `optional` is not `None` or
        `default_value` if it is.
    """
    if optional is None:
        return default_value
    return non_default_fn(optional, *args, **kwds)


@_ty.overload
def tuplify(arg: _ty.Iterable[Var], num: int = 1, exclude: Excludable = ()
            ) -> _ty.Tuple[Var, ...]:
    pass

@_ty.overload
def tuplify(arg: Var, num: int = 1, exclude: Excludable = ()
            ) -> _ty.Tuple[Var, ...]:
    pass

def tuplify(arg, num=1, exclude=()):
    """Make argument a tuple.

    If it is an iterable (except `str`, `dict`), it is converted to a `tuple`.
    Otherwise, it is placed in a `tuple`.

    Parameters
    ----------
    arg : Var|Iterable[Var]
        Thing to be turned / put into a `tuple`.
    num : int, optional
        Number of times to put `arg` in `tuple`, default: 1. Not used for
        conversion of iterables.
    exclude : Tuple[Type, ...], optional
        Additional iterable types to exclude from conversion, by default `()`.

    Returns
    -------
    tuplified : Tuple[Var, ...]
        Tuple containing `arg`.
    """
    return tuple(arg) if _is_iter(arg, exclude) else (arg,) * num


@_ty.overload
def repeatify(arg: _ty.Iterable[Var], times: _ty.Optional[int] = None,
              exclude: Excludable = ()) -> _ty.Iterable[Var]:
    pass

@_ty.overload
def repeatify(arg: Var, times: _ty.Optional[int] = None,
              exclude: Excludable = ()
              ) -> _ty.Iterable[Var]:
    pass

def repeatify(arg, times=None, exclude=()):
    """Repeat argument if not iterable

    Parameters
    ----------
    arg : Var or Iterable[Var]
        Argument to repeat
    times : int, optional
        Maximum number of times to repeat, by default `None`.
    exclude : Tuple[Type, ...], optional
        Additional iterable types to exclude from conversion, by default `()`.

    Returns
    -------
    repeated : Iterable[Var]
        Iterable version of `arg`.
    """
    opt = eval_or_default(times, tuple, ())
    return arg if _is_iter(arg, exclude) else _it.repeat(arg, *opt)


def unseqify(arg: _ty.Sequence[Var]) -> _ty.Optional[InstanceOrSeq[Var]]:
    """Unpack sequence before returning, if not longer than 1.

    If a sequence has a single element, return that. If empty, return `None`.
    Otherwise return the sequence.

    Parameters
    ----------
    arg : Sequence[Var]
        Sequence to be unpacked.

    Returns
    -------
    val : Sequence[Var] or Var or None
        The sequence or its contents (if there are not more than one).
    """
    if len(arg) == 0:
        return None
    if len(arg) == 1:
        return arg[0]
    return arg


def seq_get(seq: _ty.Sequence[Val], ind: _ty.Union[int, slice],
            default: _ty.Optional[Val] = None) -> Val:
    """Get an element from a sequence, or default if index is out of range

    Parameters
    ----------
    seq : Sequence[Val]
        The sequence from which we get the element.
    ind : int or slice
        The index of the element we want from `seq`.
    default : Optional[Val], optional
        Value to return if `ind` is out of range for `seq`, by default `None`.

    Returns
    -------
    element : Val
        Element of the sequence, `seq[ind]`, or `default`.
    """
    try:
        return seq[ind]
    except IndexError:
        return default


def rev_seq(seq: _ty.Reversible) -> _ty.Reversible:
    """reverse a sequence, leaving it a sequence if possible

    Parameters
    ----------
    seq : _ty.Reversible
        Either a sequence or a reversible iterator

    Returns
    -------
    rseq : _ty.Reversible
        If `seq` is a sequence, this is the sequence in reversed order,
        otherwise it is a reversed iterator over `seq`.
    """
    if isinstance(seq, _cn.abc.Sequence):
        return seq[::-1]
    return reversed(seq)


def _and_reverse(it_func: _ty.Callable[..., _ty.Iterable]) -> _ty.Callable[..., _ty.Iterable]:
    """Wrap iterator factory with reversed
    """
    @_ft.wraps(it_func)
    def rev_it_func(*args, **kwds):
        return reversed(it_func(*args, **kwds))

    new_name = it_func.__name__ + '.rev'
    rev_it_func.__name__ = new_name
    it_func.rev = rev_it_func
    rev_it_func.rev = it_func


@_and_reverse
def zenumerate(*iterables: _it.Iterable, start=0, step=1) -> ZipSequences:
    """Combination of enumerate and unpacked zip.

    Behaves like `enumerate`, but accepts multiple iterables.
    The output of `next` is a `tuple`: (counter, iter0, iter1, ...)
    `start` and `step` can only be passed as keyword arguments.

    Example
    -------
    >>> words = [''] * 6
    >>> letters = 'xyz'
    >>> counts = [1, 7, 13]
    >>> for idx, key, num in zenumerate(letters, counts, start=1, step=2):
    >>>     words[idx] = key * num
    >>>     time.sleep(0.1)
    >>> print(words)
    """
    counter = erange(start, _it.min_len(*iterables), step)
    return ZipSequences(counter, *iterables)


# =============================================================================
# ABC mixin with __subclasshook__
# =============================================================================


class ABCauto(_abc.ABC):
    """Base class for ABCs with automatic subclass check for abstract methods.
    """

    @classmethod
    def __subclasshook__(cls, subcls):
        return _subclass_hook(cls, subcls)

    def __init_subclass__(cls, typecheckonly: bool = False):
        if not typecheckonly:
            supname = _supername(cls, ABCauto)
            raise TypeError(f'{supname} should not be used as a superclass.'
                            ' It is meant for instance/subclass checks only.')


# =============================================================================
# ABCs & mixins
# =============================================================================


class RangeIsh(ABCauto, typecheckonly=True):
    """ABC for range-ish objects - those with start, stop, step attributes.

    Intended for instance/subclass checks only.
    """

    @property
    @_abc.abstractmethod
    def start(self) -> RangeArg:
        """Start of range"""

    @property
    @_abc.abstractmethod
    def stop(self) -> RangeArg:
        """End of range"""

    @property
    @_abc.abstractmethod
    def step(self) -> RangeArg:
        """Step between members of range"""


class RangeLike(RangeIsh, typecheckonly=True):
    """ABC for range-like objects: range-ish objects with index, count methods.

    Intended for instance/subclass checks only.
    """

    @_abc.abstractmethod
    def count(self, value: Eint) -> int:
        """return number of occurences of value"""

    @_abc.abstractmethod
    def index(self, value: Eint) -> Eint:
        """return index of value.
        Raise ValueError if the value is not present.
        """


class ContainerMixin:
    """Mixin class to add extra Collection methods to RangeIsh classes

    Should be used with `RangeCollectionMixin`
    """

    def count(self, value: Eint) -> int:
        """return number of occurences of value"""
        return int(value in self)

    def index(self, value: Eint) -> Eint:
        """return index of value.
        Raise ValueError if the value is not present.
        """
        if value not in self:
            raise ValueError(f"{value} is not in range")
        return (value - self.start) // self.step

    @_abc.abstractmethod
    def __contains__(self, arg: Eint) -> bool:
        pass


class RangeCollectionMixin(ContainerMixin):
    """Mixin class to add range-container methods to RangeIsh classes"""

    def __len__(self) -> int:
        # iterable behaviour
        if _isinfslice(self):
            return _math.inf
        return len(range(*_range_args_def(self)))

    def __contains__(self, arg: Eint) -> bool:
        # iterable behaviour
        if ((arg - self.start) * self.step < 0
                or (arg - self.start) % self.step != 0):
            return False
        if not _isinfslice(self) and (arg - self.stop) * self.step >= 0:
            return False
        return True


# =============================================================================
# Classes
# =============================================================================


class ZipSequences(_cn.abc.Sequence):
    """Like zip, but sized, subscriptable and reversible (if arguments are).

    Parameters
    ----------
    sequence1, sequence2, ...
        sequences to iterate over
    usemax : bool, keyword only, default=False
        If True, we continue until all sequences are exhausted. If False, we
        stop when we reach the end of the shortest sequence.

    Raises
    ------
    TypeError
        When calling `len` if any memeber is is not `Sized`.
        When calling `reverse` if any memeber is is not `Reversible`.
        When subscripting if any memeber is is not subscriptable.

    Notes
    -----
    If sequences are not of equal length, the reversed iterator will not yield
    the same tuples as the original iterator. Each sequence is reversed as is,
    without omitting end-values or adding fill-values. Similar considerations
    apply to negative indices.

    Indexing with an integer returns a (tuple of) sequence content(s).
    Indexing with a slice returns a (tuple of) sub-sequence(s).
    """
    _seqs: _ty.Tuple[_ty.Sequence, ...]
    _max: bool

    def __init__(self, *sequences: _ty.Sequence, usemax: bool = False) -> None:
        self._seqs = sequences
        self._max = usemax

    def __len__(self) -> int:
        if self._max:
            return max(len(obj) for obj in self._seqs)
        return min(len(obj) for obj in self._seqs)

    def __iter__(self) -> _ty.Union[zip, _it.zip_longest]:
        if self._max:
            return iter(_it.zip_longest(*self._seqs))
        return iter(zip(*self._seqs))

    def __getitem__(self, index: _ty.Union[int, slice]):
        if self._max:
            return unseqify(tuple(seq_get(obj, index) for obj in self._seqs))
        return unseqify(tuple(obj[index] for obj in self._seqs))

    def __reversed__(self) -> ZipSequences:
        return ZipSequences(*(rev_seq(obj) for obj in self._seqs),
                            usemax=self._max)

    def __repr__(self) -> str:
        return type(self).__name__ + repr(self._seqs)

    def __str__(self) -> str:
        seqs = ','.join(type(s).__name__ for s in self._seqs)
        return type(self).__name__ + f'({seqs})'


# =============================================================================
# Extended range
# =============================================================================


class ExtendedRange(RangeCollectionMixin):
    """Combination of range and itertools.count

    Any parameter can be given as `None` and the default will be used. `stop`
    can also be `+/-inf`.

    Parameters
    ----------
    start : int or None, optional, default=0
        initial counter value (inclusive).
    stop : int or None, optional, default=inf*sign(step)
        value of counter at or above which the loop terminates (exclusive).
    step : int or None, optional, default=1
        increment of counter after each loop.
    """
    start: _nm.Integral
    stop: Eint
    step: _nm.Integral
    _iter: _ty.Union[range, _it.count]

    def __init__(self, *args, **kwds):
        super().__init__()
        self.start, self.stop, self.step = _extract_slice(args, kwds)
        self.start, self.stop, self.step = _range_args_def(self)
        if self.step == 0:
            raise ValueError("step cannot be 0.")
        if _isinfslice(self):
            self._iter = _it.count(self.start, self.step)
        else:
            self._iter = range(self.start, self.stop, self.step)

    def index(self, value: Eint) -> Eint:
        """return index of value.
        Raise ValueError if the value is not present.
        """
        if not _isinfslice(self):
            return self._iter.index(value)
        return super().index(value)

    def __iter__(self) -> _ty.Union[range, _it.count]:
        yield from self._iter

    def __next__(self) -> int:
        return next(self._iter)

    def __len__(self) -> int:
        if _isinfslice(self):
            return _math.inf
        return len(self._iter)

    def __contains__(self, arg: Eint) -> bool:
        if not _isinfslice(self):
            return arg in self._iter
        return super().__contains__(arg)

    def __reversed__(self) -> ExtendedRange:
        _raise_if_no_stop(self)
        args = self.stop - self.step, self.start - self.step, -self.step
        return type(self)(*args)

    def __repr__(self) -> str:
        return "erange" + _range_repr(self)

    def __getitem__(self, ind: _ty.Union[Eint, RangeIsh]
                    ) -> _ty.Union[Eint, ExtendedRange]:
        if isinstance(ind, (_nm.Integral, _nm.Real)):
            val = _nth_value(self, ind)
            if val is None:
                raise IndexError(f'{ind} out of range when len={len(self)}')
            return val
        start, stop, step = _range_args(ind)
        step = default(step, 1)
        if step < 0:
            return self.__reversed__()[start:stop:-step]
        start, stop, step = _range_args_def(ind)
        nstart, nstop = _nth_value(self, start), _nth_value(self, stop)
        nstart, nstop = default(nstart, self.start), default(nstop, self.stop)
        nstep = step * self.step
        return type(self)(nstart, nstop, nstep)


erange = ExtendedRange

# =============================================================================


def _nth_value(rng: RangeIsh, ind: Eint) -> Eint:
    """Get n'th value from iterating over range"""
    start, stop, step = _range_args_def(rng)
    val = (start if ind >= 0 else stop) + ind * step
    if (val - start) * step < 0 or (stop - val) * step <= 0:
        return None
    return val


def _supername(cls: type, base: type = object) -> str:
    """String of name of superclass

    Searches for first subclass of `base` in `cls.__mro__` other than `cls`.
    raises `ValueError` if not found.
    """
    for scls in cls.__mro__:
        if scls is not cls and issubclass(scls, base):
            return scls.__name__
    raise ValueError(f"{base.__name__} is not a superclass of {cls.__name__}")


def _check_dict(the_class: type, method: str) -> CheckResult:
    """Check if method is in class dictionary.
    """
    if method in the_class.__dict__:
        if the_class.__dict__[method] is None:
            return NotImplemented
        return True
    return False


def _check_annotations(the_class: type, prop: str) -> CheckResult:
    """Check if attribute is in class annotations.
    """
    return prop in getattr(the_class, '__annotations__', {})


def _check_property(the_class: type, prop: str) -> CheckResult:
    """Check if prop is in class dictionary (as a property) or annotation.
    """
    is_ok = _check_dict(the_class, prop)
    if is_ok is NotImplemented:
        return NotImplemented
    if is_ok:
        return isinstance(the_class.__dict__[prop], PROP_TYPES)
    return _check_annotations(the_class, prop)


def _check_generic(the_cls: type, check: Checker, *methods: str) -> CheckResult:
    """Check class for methods
    """
    mro = the_cls.__mro__
    for method in methods:
        for super_class in mro:
            is_ok = check(super_class, method)
            if is_ok is NotImplemented:
                return NotImplemented
            if is_ok:
                break
        else:
            return NotImplemented
    return True


def _check_methods(the_class: type, *methods: str) -> CheckResult:
    """Check if methods are in class dictionary.
    """
    return _check_generic(the_class, _check_dict, *methods)


def _check_properties(the_class: type, *properties: str) -> CheckResult:
    """Check if properties are in class dictionary (as property) or annotations
    """
    return _check_generic(the_class, _check_property, *properties)


def _get_abstracts(the_class: type) -> _ty.Tuple[_ty.List[str], ...]:
    """Get names of abstract methods and properties
    """
    abstracts = getattr(the_class, '__abstractmethods__', set())
    methods, properties = [], []
    for abt in abstracts:
        if isinstance(getattr(the_class, abt, None), property):
            properties.append(abt)
        else:
            methods.append(abt)
    return methods, properties


def _subclass_hook(cls: type, subcls: type) -> CheckResult:
    """Inheritable implementation of __subclasshook__.

    Use in `__subclasshook__(cls, subcls)` as
    `return subclass_hook(cls, subcls)`
    """
    methods, properties = _get_abstracts(cls)
    is_ok = _check_methods(subcls, *methods)
    if is_ok is not True:
        return is_ok
    return _check_properties(subcls, *properties)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def _isinfslice(obj: RangeIsh) -> bool:
    """is obj.stop None/inf? Can iterable go to +/- infinity?"""
    return _isinfnone(obj.stop)


def _isinfnone(val: _ty.Optional[Eint]) -> bool:
    """is val None/inf?"""
    return val is None or _math.isinf(val)

# -----------------------------------------------------------------------------
# Exceptions
# -----------------------------------------------------------------------------


def _default_str(optional: RangeArg, template: str) -> str:
    """Evaluate format on optional if it is not None/inf

    Parameters
    ----------
    optional : int, inf or None
        The optional argument, where `None`/`inf` indicates that the default
        value should be used instead.
    template : str
        Evaluate `template.format` on `optional`if it is not `None`/`inf`.
    """
    if _isinfnone(optional):
        return ''
    return template.format(optional)


def _range_repr(the_range: RangeIsh, bracket: bool = True) -> str:
    """Minimal string such that `f'range{range_repr(the_range)}'` evaluates to
    `the_range`

    Parameters
    ----------
    the_range : RangeIsh
        Instance to represent.
    bracket : bool, optional
        Do we enclose result in ()? default: True
    """
    if bracket:
        return f'({_range_repr(the_range, False)})'
    return (_default_str(the_range.start, '{},') + str(the_range.stop)
            + _default_str(the_range.step, ',{}'))


def _raise_if_no_stop(obj: RangeIsh):
    """raise ValueError if obj.stop is None/inf"""
    if _isinfslice(obj):
        raise ValueError("Need a finite value for stop")


def _range_args(the_range: RangeIsh) -> RangeArgs:
    """Extract start, stop, step from range
    """
    return the_range.start, the_range.stop, the_range.step


def _range_args_def(the_range: RangeIsh) -> RangeArgs:
    """Extract start, stop, step from range, using defaults where possible

    Also sets `stop = start + n * step` for some non-negative integer `n`
    (without changing last value) when possible.

    Parameters
    ----------
    the_range : RangeIsh
        An object that has integer attributes named `start`, `stop`, `step`,
        e.g. `slice`, `range`, `DisplayCount`

    Returns
    -------
    start : int or None
        Start of range, with default 0/inf for positive/negative step.
    stop : int or None
        Past end of range, with default sign(step) * inf.
    step : int
        Increment of range, with default 1.
    """
    start, stop, step = _range_args(the_range)
    step = default(step, 1)
    if step == 0:
        raise ValueError('range step cannot be zero')
    start = default(start, 0 if step > 0 else _math.inf)
    stop = default(stop, _math.inf * step)
    if (stop - start) * step < 0:
        stop = start
    if _math.isfinite(stop):
        remainder = (stop - start) % step
        if remainder:
            stop += step - remainder
    return start, stop, step


def _extract_slice(args: SliceArgs, kwargs: SliceKeys) -> SliceArgs:
    """Extract slice indices from args/kwargs

    Parameters
    ----------
    args
        `tuple` of arguments.
    kwargs
        `dict` of keyword arguments. Keywords below are popped if present.

    Returns
    -------
    start : int or None, optional, default=0
        initial counter value (inclusive).
    stop : int or None, optional, default=None
        value of counter at, or above which, the loop terminates (exclusive).
    step : int or None, optional, default=1
        increment of counter after each loop.
    """
    if not args:
        inds = slice(None)
    elif len(args) == 1 and isinstance(args[0], slice):
        inds = args[0]
    else:
        inds = slice(*args)
    start = kwargs.pop('start', inds.start)
    stop = kwargs.pop('stop', inds.stop)
    step = kwargs.pop('step', inds.step)
    step = default(step, 1)
    if step > 0:
        start = default(start, 0)
    else:
        stop = default(stop, -1)
    return start, stop, step


def _is_iter(arg: _ty.Any, exclude: Excludable = ()) -> bool:
    """Is it a non-exluded iterable?"""
    return (isinstance(arg, _cn.abc.Iterable)
            and not isinstance(arg, EXCLUDIFY + exclude))


# =============================================================================
# Type hints
# =============================================================================
PROP_TYPES = (property, _tys.MemberDescriptorType)
EXCLUDIFY = (str, dict, _cn.UserDict)
Excludable = _ty.Tuple[_ty.Type[_ty.Iterable], ...]
Some = _ty.TypeVar('Some')
Other = _ty.TypeVar('Other')
Var = _ty.TypeVar('Var')
Val = _ty.TypeVar('Val')
InstanceOrSeq = _ty.Union[Var, _ty.Sequence[Var]]
CheckResult = _ty.Union[bool, type(NotImplemented)]
Checker = _ty.Callable[[type, str], CheckResult]
Eint = _ty.Union[_nm.Integral, _nm.Real]
RangeArg = _ty.Optional[Eint]
RangeArgs = _ty.Tuple[RangeArg, ...]
NameArg = _ty.Optional[str]
SliceArg = _ty.Optional[int]
Arg = _ty.Union[SliceArg, _ty.Iterable]
SliceArgs = _ty.Tuple[SliceArg, ...]
Args = _ty.Tuple[Arg, ...]
SliceKeys = _ty.Dict[str, SliceArg]
Keys = _ty.Dict[str, Arg]
#  _ty.Literal[np.inf, -np.inf, np.nan]
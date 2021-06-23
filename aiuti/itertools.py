"""
Itertools
=========

This module contains various useful functions for working with iterators
beyond the scope the itertools library without overlapping with
more_itertools.

"""

__all__ = [
    'exhaust',
    'split',
]

import operator as op
from collections import deque
from itertools import tee, compress
from typing import Tuple, Iterable, Iterator, Callable, Union, Any

from .typing import T


def exhaust(iterable: Iterable[Any]) -> None:
    """
    Iterate through all the values of an iterable. This is useful for
    applying a mapped function which doesn't return a useful value
    without writing a dedicated for loop.

    >>> exhaust(map(print, range(3)))
    0
    1
    2

    """
    # deque is ~10% faster than writing a loop
    deque(iterable, maxlen=0)


def split(
    iterable: Iterable[T],
    condition: Union[Iterable[bool], Callable[[T], bool]],
) -> Tuple[Iterator[T], Iterator[T]]:
    """
    Similar to :func:`itertools.compress` except that it returns two
    iterators: one where with the values where the condition is True
    and one where with the values where the condition is False.

    >>> from itertools import cycle
    >>> evens, odds = split(range(5), cycle((True, False)))
    >>> list(evens)
    [0, 2, 4]
    >>> list(odds)
    [1, 3]

    The condition can also be a boolean callable:

    >>> evens, odds = split(range(5), lambda x: x % 2 == 0)
    >>> list(evens)
    [0, 2, 4]
    >>> list(odds)
    [1, 3]

    :param iterable:
        Iterable to split into two sub-iterators.
        If given as iterator, it must not be advanced directly later!
    :param condition:
        Either a boolean callable to map to each element or a boolean
        iterable indicating which values belong in which set.
        If given as iterator, it must not be advanced directly later!
    :return: Tuple of two iterators: (where_true, where_false)
    """
    if callable(condition):
        iterable, ci = tee(iterable)
        condition = map(condition, ci)
    i1, i2 = tee(iterable)
    c1, c2 = tee(condition)
    return compress(i1, c1), compress(i2, map(op.not_, c2))

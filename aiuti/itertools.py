"""
Itertools
=========

This module contains various useful functions for working with iterators
beyond the scope the itertools library without overlapping with
more_itertools.

"""

__all__ = [
    'exhaust',
]

from collections import deque
from typing import Iterable, Any


def exhaust(iterable: Iterable[Any]):
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

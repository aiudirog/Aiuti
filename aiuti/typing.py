"""
Typing
======

This module contains various useful typing definitions to simplify
using some of the advanced typing features and avoid creating the same
generic declarations from project to project.

It also fallbacks for the following common new typing features to avoid
import errors on older versions of Python:

    - ``Literal``
    - ``Protocol``
    - ``TypedDict``

"""

__all__ = [
    'Literal', 'Protocol', 'TypedDict',
    'MaybeIter', 'MaybeAwaitable', 'Yields', 'AYields',
    'T', 'T_co', 'T_contra', 'KT', 'VT', 'KT_co', 'VT_co',
]

import logging
from typing import (
    TypeVar, Dict, Union, Iterable, Awaitable,
    Generator, AsyncGenerator,
)

logger = logging.getLogger(__name__)


try:
    from typing import Literal, Protocol, TypedDict
except ImportError:
    try:
        from typing_extensions import (  # type: ignore
            Literal, Protocol, TypedDict,
        )
    except ImportError:
        # Shim implement common back ports to avoid requiring
        # typing_extensions to be installed for regular execution.

        # Note that this is not a replacement for typing_extensions which
        # should be made a normal dependency of any packages that use newer
        # typing features and wish to support older Python versions.
        logger.info("Creating shims for common typing features."
                    " Please install typing_extensions to avoid this.")
        Literal = type('Literal', (type,),  # type: ignore
                       {'__getitem__': lambda c, i: c})
        Protocol = type('Protocol', (type,),   # type: ignore
                        {'__getitem__': lambda c, i: c})
        TypedDict = type('TypedDict', (Dict,), {})

# Define common TypeVar instances for convenience

#: Generic invariant type var
T = TypeVar('T')
#: Generic covariant type var
T_co = TypeVar('T_co', covariant=True)
#: Generic contravariant type var
T_contra = TypeVar('T_contra', contravariant=True)

#: Generic invariant type var for keys
KT = TypeVar('KT')
#: Generic invariant type var for values
VT = TypeVar('VT')

#: Generic covariant type var for keys
KT_co = TypeVar('KT_co', covariant=True)
#: Generic covariant type var for values
VT_co = TypeVar('VT_co', covariant=True)

#: Generic type for an object which may be a single value of a given
#: type or an iterable of that type
MaybeIter = Union[T, Iterable[T]]

#: Generic type for an object which could be wrapped in an awaitable
MaybeAwaitable = Union[T, Awaitable[T]]

#: Generic type for a Generator which only yields
#:
#: This is different from ``Iterator[T]`` as it also declares that the
#: object implements the rest ``Generator`` interface such as
#: ``gen.close()`` and ``gen.throw()``.
Yields = Generator[T, None, None]

#: Generic type for an ``AsyncGenerator`` which only yields
AYields = AsyncGenerator[T, None]

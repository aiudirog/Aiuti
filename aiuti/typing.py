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
    - ``TypeAlias``
    - ``ParamSpec``
    - ``ParamSpecArgs``
    - ``ParamSpecKwargs``
    - ``Concatenate``

"""

__all__ = [
    'Literal', 'Protocol', 'TypedDict', 'TypeAlias',
    'ParamSpec', 'ParamSpecArgs', 'ParamSpecKwargs', 'Concatenate',
    'MaybeIter', 'MaybeAwaitable', 'Yields', 'AYields',
    'T', 'T_co', 'T_contra', 'KT', 'VT', 'KT_co', 'VT_co', 'F',
]

import logging
from typing import (
    Any, AsyncGenerator, Awaitable, Callable, Dict, Generator, Iterable,
    TypeVar, Union,
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
                       {'__class_getitem__': lambda c, i: c})
        Protocol = type('Protocol', (type,),   # type: ignore
                        {'__class_getitem__': lambda c, i: c})
        TypedDict = type('TypedDict', (Dict,), {})

try:
    from typing import ParamSpec, ParamSpecArgs, ParamSpecKwargs, Concatenate
except ImportError:
    try:
        from typing_extensions import (  # type: ignore
            ParamSpec, ParamSpecArgs, ParamSpecKwargs, Concatenate,  # noqa
        )
    except ImportError:
        logger.info("Creating shims for ParamSpec typing features."
                    " Please install typing_extensions to avoid this.")

        def _any_init(*_: Any, **__: Any) -> None: return None
        ParamSpecArgs = type('ParamSpecArgs', (),  # type: ignore
                             {'__init__': lambda *_, **__: None})
        ParamSpecKwargs = type('ParamSpecArgs', (),  # type: ignore
                               {'__init__': lambda *_, **__: None})
        ParamSpec = type('ParamSpec', (list,),  # type: ignore
                         {'__init__': lambda *_, **__: None,
                          'args': None, 'kwargs': None})
        Concatenate = type('Concatenate', (type,),   # type: ignore
                           {'__class_getitem__': lambda c, i: c})

try:
    from typing import TypeAlias
except ImportError:
    try:
        from typing_extensions import TypeAlias
    except ImportError:
        TypeAlias = type('TypeAlias', (), {})  # type: ignore

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

#: Generic type var for a callable function
F = TypeVar('F', bound=Callable[..., Any])

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

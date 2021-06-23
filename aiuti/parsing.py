"""
Parsing
=======

This module contains various useful tools for parsing input data.

"""

__all__ = ['parse_to_dict']

import ast
from typing import Iterable, Union, Callable, Any, Tuple, Mapping, Dict


def parse_to_dict(items: Union[Mapping[Any, Any],
                               Iterable[Union[Tuple[Any, ...], str]]],
                  *,
                  sep: str = '=',
                  parse: Callable[[str], Any] = ast.literal_eval,
                  parse_keys: bool = True) -> Dict[Any, Any]:
    """
    Normalize the given input into a dictionary and parse keys & values
    as necessary into literal objects.

    When given an iterable of strings, they are assumed to be in the
    form of ``<key><sep><value>`` like ``a=b`` and will be parsed as
    such.

    Keys or values which fail to parse will be retained as is.

    By default, both keys and values are parsed literally:

    >>> parse_to_dict({'a': '1', '2': '"b"'})
    {'a': 1, 2: 'b'}

    >>> parse_to_dict(['a=1', '2="b"', '"b"=1.4'])
    {'a': 1, 2: 'b', 'b': 1.4}

    Parsing keys can be disabled with the ``parse_keys`` parameter:

    >>> parse_to_dict({'a': '1', '2': '"b"'}, parse_keys=False)
    {'a': 1, '2': 'b'}

    >>> parse_to_dict(['a=1', '2=b'], parse_keys=False)
    {'a': 1, '2': 'b'}

    The default separator can be changed using the ``sep`` parameter:

    >>> parse_to_dict(['a:1', '2:"b"', '"b":1.4'], sep=':')
    {'a': 1, 2: 'b', 'b': 1.4}

    :param items: Mapping or iterable to parse to a dictionary
    :param sep:
        For elements that are strings, the separator between the key and
        value
    :param parse:
        Function which will be used to try to parse strings into literal
        values
    :param parse_keys: Try to parse the keys when they are strings?
    """

    def try_parse(x: Any) -> Any:
        if isinstance(x, str):
            try:
                return parse(x)
            except:  # noqa
                pass
        return x

    if parse_keys:
        def parse_tuple(key: str, value: str) -> Tuple[Any, Any]:
            return try_parse(key), try_parse(value)
    else:
        def parse_tuple(key: str, value: str) -> Tuple[Any, Any]:
            return key, try_parse(value)

    def parse_pair(pair: Union[str, Tuple[Any, Any]]) -> Tuple[Any, Any]:
        if isinstance(pair, str):
            try:
                k, v = pair.split(sep, 1)
            except ValueError as e:
                raise ValueError(f"{pair} is not like KEY{sep}VALUE") from e
            return parse_tuple(k, v)
        return parse_tuple(*pair)

    try:
        items = items.items()  # type: ignore
    except AttributeError:
        pass

    return dict(map(parse_pair, items))

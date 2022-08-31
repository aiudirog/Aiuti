import random
import asyncio as aio
from itertools import repeat
from typing import Set

import pytest

from aiuti.typing import AYields, Yields
from aiuti.asyncio import buffer_until_timeout, BufferAsyncCalls


@pytest.fixture
def buffered() -> Set[int]:
    return set()


@pytest.fixture
async def func(buffered: Set[int]) -> BufferAsyncCalls[int]:
    @buffer_until_timeout
    async def func(ints: Set[int]) -> None:
        nonlocal buffered
        buffered |= ints

    return func


# Test deadlock avoidance. The event is cleared whenever something is
# scheduled to be put on the queue. If that fails to happen, we need to
# make sure it doesn't block indefinitely waiting for an item.

async def test_error_await(func: BufferAsyncCalls[int],
                           buffered: Set[int]) -> None:
    """
    Verify that passing an awaitable which errors to .await_() doesn't
    cause a deadlock.
    """

    async def error() -> int:
        raise ValueError
        return 1  # noqa

    func.await_(error())
    await aio.wait_for(func.wait(), 10)
    assert not buffered  # sanity


async def test_error_await_many(func: BufferAsyncCalls[int],
                                buffered: Set[int]) -> None:
    """Same as test_error_await but with many awaitables."""

    async def error(i: int) -> int:
        await aio.sleep(0.01)
        if i % 10 == 0:  # Seemingly random errors
            raise ValueError
        return i

    for x in range(250):
        func.await_(error(x))
    await aio.wait_for(func.wait(), 10)
    assert buffered == set(range(250)) - set(range(0, 250, 10))


async def test_empty_iter(func: BufferAsyncCalls[int],
                          buffered: Set[int]) -> None:
    """
    Verify that passing an empty iterator to .map() doesn't cause
    a deadlock.
    """
    func.map(iter([]))
    await aio.wait_for(func.wait(), 10)
    assert not buffered  # sanity


async def test_error_iter(func: BufferAsyncCalls[int],
                          buffered: Set[int]) -> None:
    """
    Verify that passing an iterator which errors to .map() doesn't cause
    a deadlock.
    """

    def error_gen() -> Yields[int]:
        raise ValueError
        yield 1  # noqa

    func.map(error_gen())
    await aio.wait_for(func.wait(), 10)
    assert not buffered  # sanity


async def test_error_iter_ele(func: BufferAsyncCalls[int],
                              buffered: Set[int]) -> None:
    """Same as test_error_iter but with one element yielded."""

    def error_gen() -> Yields[int]:
        yield 1
        raise ValueError

    func.map(error_gen())
    await aio.wait_for(func.wait(), 10)
    assert buffered == {1}


async def test_error_iter_many(func: BufferAsyncCalls[int],
                               buffered: Set[int]) -> None:
    """
    Same as test_error_iter_ele but with many elements yielded across
    multiple iterators.
    """

    def error_gen(i: int) -> Yields[int]:
        for x in range(i, i + 10):
            yield x
        if i % 50 == 0:  # Seemingly random error
            raise ValueError

    for start in range(0, 250, 10):
        func.map(error_gen(start))
    await aio.wait_for(func.wait(), 10)
    assert buffered == set(range(250))


async def test_empty_aiter(func: BufferAsyncCalls[int],
                           buffered: Set[int]) -> None:
    """
    Verify that passing an empty async iterator to .amap() doesn't cause
    a deadlock.
    """

    async def empty_gen() -> AYields[int]:
        return
        yield 1  # noqa

    func.amap(empty_gen())
    await aio.wait_for(func.wait(), 10)
    assert not buffered  # sanity


async def test_error_aiter(func: BufferAsyncCalls[int],
                           buffered: Set[int]) -> None:
    """
    Verify that passing an async iterator which errors to .amap()
    doesn't cause a deadlock.
    """

    async def error_gen() -> AYields[int]:
        raise ValueError
        yield 1  # noqa

    func.amap(error_gen())
    await aio.wait_for(func.wait(), 10)
    assert not buffered  # sanity


async def test_error_aiter_ele(func: BufferAsyncCalls[int],
                               buffered: Set[int]) -> None:
    """Same as test_error_aiter but with one element yielded."""

    async def error_gen() -> AYields[int]:
        await aio.sleep(0.1)
        yield 1
        raise ValueError

    func.amap(error_gen())
    await aio.wait_for(func.wait(), 10)
    assert buffered == {1}


async def test_error_aiter_many(func: BufferAsyncCalls[int],
                                buffered: Set[int]) -> None:
    """
    Same as test_error_aiter_ele but with many elements yielded across
    multiple iterators.
    """

    async def error_gen(i: int) -> AYields[int]:
        for x in range(i, i + 10):
            await aio.sleep(0.01)
            yield x
        if i % 50 == 0:  # Seemingly random error
            raise ValueError

    for start in range(0, 250, 10):
        func.amap(error_gen(start))
    await aio.wait_for(func.wait(), 10)
    assert buffered == set(range(250))


async def test_stressed(func: BufferAsyncCalls[int],
                        buffered: Set[int]) -> None:
    """Combination of many existing tests in this module."""

    async def error(i: int) -> int:
        await aio.sleep(0)
        if i % 3 == 0:
            raise ValueError
        return i

    def error_gen(i: int) -> Yields[int]:
        for x in range(i, i + 10):
            yield x
        if i % 3 == 0:
            raise ValueError

    async def error_agen(i: int) -> AYields[int]:
        for x in range(i, i + 10):
            await aio.sleep(0)
            yield x
        if i % 3 == 0:
            raise ValueError

    # Generate repeating numbers of 0, 1, & 2 with each sequence being
    # in a random order
    stages = (r for t in repeat((0, 1, 2)) for r in random.sample(t, len(t)))

    output = set()
    for start in range(0, 1000, 10):
        stage = next(stages)
        if stage == 0:
            for n in range(start, start + 10):
                func.await_(error(n))
                if n % 3 != 0:
                    output.add(n)
        elif stage == 1:
            func.map(error_gen(start))
            output |= set(range(start, start + 10))
        else:
            func.amap(error_agen(start))
            output |= set(range(start, start + 10))
        await aio.sleep(random.random() / 10)

    await aio.wait_for(func.wait(), 30)
    if buffered != output:
        print(output)
    assert buffered == output

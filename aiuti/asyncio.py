"""
Asyncio
=======

This module contains various useful functions and classes for handling
common and awkward tasks in Asyncio.

"""

__all__ = [
    'to_async_iter',
    'to_sync_iter',
    'threadsafe_async_cache',
    'buffer_until_timeout',
    'BufferAsyncCalls',
]

import queue
import logging
import asyncio as aio
from threading import Lock
from collections import defaultdict
from functools import partial, wraps
from concurrent.futures import ThreadPoolExecutor
from typing import (
    Iterator, AsyncIterator, Callable, Set, Awaitable, Optional,
    Generic,
)

from .typing import T

logger = logging.getLogger(__name__)

_DONE = object()


async def to_async_iter(iterable: Iterator[T]) -> AsyncIterator[T]:
    """
    Convert the given iterator from synchronous to asynchrounous by
    iterating it in a background thread.

    This can be useful for interacting with libraries for APIs that do
    not have an asynchronous version yet.

    To demonstrate this, let's create a function using Requests which
    makes a synchronous calls to the free {JSON} Placeholder API and
    then yields the results:

    >>> import requests

    >>> def user_names() -> Iterator[str]:
    ...     for uid in range(3):
    ...         url = f'https://jsonplaceholder.typicode.com/users/{uid + 1}'
    ...         yield requests.get(url).json()['name']

    Now, we can convert this to an async iterator which won't block the
    event loop:

    >>> async def print_user_names():
    ...     async for name in to_async_iter(user_names()):
    ...         print(name)

    >>> aio.get_event_loop().run_until_complete(print_user_names())
    Leanne Graham
    Ervin Howell
    Clementine Bauch
    """

    def _queue_elements():
        try:
            for x in iterable:
                put(x)
        finally:
            put(_DONE)

    q = aio.Queue()
    loop = aio.get_event_loop()
    put = partial(loop.call_soon_threadsafe, q.put_nowait)
    with ThreadPoolExecutor(1) as pool:
        future = loop.run_in_executor(pool, _queue_elements)
        while True:
            i = await q.get()
            if i is _DONE:
                break
            yield i
        await future  # Bubble any errors


def to_sync_iter(iterable: AsyncIterator[T]) -> Iterator[T]:
    """
    Convert the given iterator from asynchronous to synchrounous by
    by using a background thread running a new event loop to iterate it.

    This can be useful for providing a synchronous interface to a new
    async library for backwards compatibility.

    To demonstrate this, let's create a function using HTTPX which
    makes asynchronous calls to the free {JSON} Placeholder API and
    then yields the results:

    >>> import httpx

    >>> async def user_name(uid: int) -> str:
    ...     url = f'https://jsonplaceholder.typicode.com/users/{uid + 1}'
    ...     async with httpx.AsyncClient() as client:
    ...         resp = await client.get(url)
    ...         return resp.json()['name']

    >>> async def user_names() -> AsyncIterator[str]:
    ...     for name in await aio.gather(*map(user_name, range(3))):
    ...         yield name

    Now, we can convert this to a synchronous iterator and show the
    results:

    >>> for name in to_sync_iter(user_names()):
    ...     print(name)
    Leanne Graham
    Ervin Howell
    Clementine Bauch
    """

    async def _queue_elements():
        try:
            async for x in iterable:
                put(x)
        finally:
            put(_DONE)

    q = queue.Queue()
    loop = aio.new_event_loop()
    put = partial(loop.call_soon_threadsafe, q.put_nowait)
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(loop.run_until_complete, _queue_elements())
        try:
            while True:
                i = q.get()
                if i is _DONE:
                    break
                yield i
        finally:
            future.result()


def threadsafe_async_cache(func: T) -> T:
    """
    Simple thread-safe asynchronous cache decorator which ensures that
    the decorated/wrapped function is only ever called once for a given
    set of inputs even across different event loops in multiple threads.

    To demonstrate this, we'll define an async function which sleeps
    to simulate work and then returns the square of the input:

    >>> import asyncio as aio

    >>> @threadsafe_async_cache
    ... async def square(x: int):
    ...     await aio.sleep(0.1)
    ...     print("Squaring:", x)
    ...     return x * x

    Then we'll define a function that will use a new event loop to call
    the function 10 times in parallel:

    >>> def call_func(x: int):
    ...     loop = aio.new_event_loop()
    ...     aio.set_event_loop(loop)
    ...     return loop.run_until_complete(aio.gather(*(
    ...         square(x) for _ in range(10)
    ...     )))

    And finally we'll call that function 10 times in parallel across 10
    threads:

    >>> with ThreadPoolExecutor(max_workers=10) as pool:
    ...     results = list(pool.map(call_func, [2] * 10))
    Squaring: 2

    As can be seen above, the function was only called once despite
    being invoked 100 times across 10 threads and 10 event loops. This
    can be confirmed by counting the total number of results:

    >>> len([x for row in results for x in row])
    100

    .. warning::
        This cache has no max size and can very easily be the source of
        memory leaks. I typically use this to wrap object methods on a
        per-instance basis during intilaztion so that the cache only
        lives as long as the object.
    """
    cache = {}

    locks = defaultdict(Lock)  # 1 lock per input key

    @wraps(func)
    async def _wrapper(*args, **kwargs):
        # Get the key from the input arguments
        key = frozenset(args + tuple(kwargs.values()))
        while True:
            try:  # to get the value from the cache
                return cache[key]
            except KeyError:
                pass
            # Need to calculate and cache the value once
            lock = locks[key]
            if lock.acquire(blocking=False):  # First to arrive, cache
                try:  # to run the function and get the result
                    result = await func(*args, **kwargs)
                except BaseException:
                    raise  # Don't cache errors, maybe timeouts or such
                else:  # Successfully got the result
                    cache[key] = result
                    del locks[key]  # Allow lock to be garbage collected
                finally:  # Ensure lock is always released
                    lock.release()
                return result
            # Wait for the other thread/task to cache the result by
            # acquiring the lock in another thread to avoid blocking the
            # current loop. Once this is done, the loop will continue
            # and the cached result can be retrieved.
            await aio.get_event_loop().run_in_executor(None, lock.acquire)
            lock.release()

    return _wrapper


def buffer_until_timeout(func: Callable[[Set[T]], Awaitable[None]] = None,
                         *,
                         timeout: float = 1) -> 'BufferAsyncCalls[T]':
    """
    Async function decorator/wrapper which buffers the arg passed in
    each call. After a given timeout has passed since the last call,
    the function is invoked on the event loop that the function was
    decorated nearest.

    >>> @buffer_until_timeout(timeout=0.1)
    ... async def buffer(args: Set[int]):
    ...     print("Buffered:", args)

    >>> for i in range(5):
    ...     buffer(i)

    >>> aio.get_event_loop().run_until_complete(aio.sleep(0.2))
    Buffered: {0, 1, 2, 3, 4}

    Another function can also wait for all elements to be processed
    before continueing:

    >>> async def after_buffered():
    ...     await buffer.wait()
    ...     print("All buffered!")

    >>> for i in range(5):
    ...     buffer(i)

    >>> aio.get_event_loop().run_until_complete(after_buffered())
    Buffered: {0, 1, 2, 3, 4}
    All buffered!

    :param func: Function to wrap
    :param timeout:
        Number of seconds to wait after the most recent call before
        executing the function.
    :return: Non-async function which will buffer the given argument
    """
    if func is None:
        return partial(buffer_until_timeout, timeout=timeout)
    return wraps(func)(BufferAsyncCalls(func, timeout=timeout))


class BufferAsyncCalls(Generic[T]):
    """
    Wrapper around an async callable to buffer inputs before calling
    it after a given timeout. This class should not be initialized
    directly, users should use :func:`buffer_until_timeout` instead.

    .. warning::
        While this object is thread-safe and arguments can be added from
        anywhere, the loop this object was created in must be running
        for it to work.
    """

    def __init__(self,
                 func: Callable[[Set[T]], Awaitable[None]],
                 *,
                 timeout: float = 1):
        #: Wrapped function which should take a set and will be called
        #: once inputs are buffered and timeout is reached
        self.func = func
        #: Timeout in seconds to wait after the last element from the
        #: queue before calling the function
        self.timeout = timeout

        #: Event loop that will be used to get args from the queue and
        #: execute the function
        self.loop = aio.get_event_loop()

        #: Asyncio queue which new arguments will be placed in
        self.q = aio.Queue()
        #: Asyncio event that is set when all current arguments have
        #: been processed and cleared whenever a new one is added.
        self.event = aio.Event()
        self.event.set()

        #: Task which is infinitely processing the queue
        self._waiting = self.loop.create_task(self._waiter())
        #: Current task that is waiting for a new element from the queue
        self._getting: Optional[aio.Task] = None

    def __call__(self, _arg: T):
        """
        Place the given argument on the queue to be processed on the
        next execution of the function.
        """
        self.event.clear()
        self.loop.call_soon_threadsafe(self.q.put_nowait, _arg)

    async def wait(self, *, cancel: bool = True):
        """
        Wait for the event to be set indicating that all arguments
        currently in the queue have been processed.

        :param cancel:
            If True (the default) and there is a task currently waiting
            for a new item from the queue, cancel it so that the queued
            arguments are processed immediately instead of after the
            timeout. Settings this to False will force the full timeout
            to be met before the function is called if it hasn't been
            already.
        """
        if cancel and self._getting and not self._getting.done():
            # Cancel the current q.get to avoid waiting for timeout
            # The sleep(0) forces task switching to ensure that
            # _process_queue gets at least one cycle to pull remaining
            # elements off the queue
            await aio.sleep(0)
            self._getting.cancel()
        await self.event.wait()

    async def _waiter(self):
        """
        Simple loop which tries to process the queue infinitely using
        :meth:`_process_queue`. This is spawned automatically upon

        TODO: Figure out how to prevent RuntimeErrors when the event
              is stopped as this task never finishes.
        """
        while True:
            await self._process_queue()

    async def _process_queue(self):
        """
        Internal method to retrieve all current elements from queue and
        execute the function on timeout or cancellation.
        """
        # Get first element, block infinitely until one appears
        inputs = {await self.q.get()}
        # Keep adding new args until the function has run successfully
        while not self.event.is_set():
            # Schedule the q.get() and save it as an attribute so it
            # can be cancelled as necessary
            self._getting = self._schedule_with_timeout(self.q.get())
            # Actually get the next argument to buffer. If this times
            # out or is cancelled, its time to run the function.
            try:
                inputs.add(await self._getting)
            except (aio.TimeoutError, aio.CancelledError):
                await self._run_func(inputs)

    async def _run_func(self, inputs: Set[T]):
        """
        Run :attr:`func` with the given set of inputs and set
        :attr:`event` once it has finished successfully.

        If an exception is raised, log it with its traceback and return
        without setting the event to prevent the buffered inputs from
        being lost.
        """
        try:
            await self.func(inputs)
        except BaseException as e:  # noqa
            logging.exception("Failed to run %s, retrying", self.func)
        else:
            self.event.set()

    def _schedule_with_timeout(self, coro: Awaitable) -> aio.Task:
        """
        Helper method to create a task for the given coroutine with the
        configured :attr:`timeout`.
        """
        return self.loop.create_task(aio.wait_for(coro, self.timeout))

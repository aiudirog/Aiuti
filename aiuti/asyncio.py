"""
Asyncio
=======

This module contains various useful functions and classes for handling
common and awkward tasks in Asyncio.

"""

__all__ = [
    'gather_excs',
    'raise_first_exc',
    'to_async_iter',
    'to_sync_iter',
    'threadsafe_async_cache',
    'buffer_until_timeout',
    'BufferAsyncCalls',
    'ensure_aw',
    'loop_in_thread',
    'run_aw_threadsafe',
    'DaemonTask',
]

import sys
import queue
import logging
import asyncio as aio
from threading import Lock, Thread
from collections import defaultdict
from functools import partial, wraps
from concurrent.futures import ThreadPoolExecutor
from typing import (
    Iterator, Iterable, AsyncIterable, Callable, Set, Awaitable, Optional,
    Type, Generic,
)

from .typing import T, Yields, AYields

Loop = aio.AbstractEventLoop

logger = logging.getLogger(__name__)

_DONE = object()


async def gather_excs(aws: Iterable[Awaitable],
                      only: Type[T] = BaseException) -> AYields[T]:
    """
    Gather the given awaitables and yield any exceptions they raise.

    This is useful when spawning a lot of tasks and you only need to
    know if any of them error.

    >>> async def x():
    ...     await aio.sleep(0.01)

    >>> async def e():
    ...     await aio.sleep(0.01)
    ...     raise RuntimeError("This failed")

    >>> async def show_errors(excs: AYields[BaseException]):
    ...     async for exc in excs:
    ...         print(type(exc).__name__ + ':', *exc.args)

    >>> run = aio.get_event_loop().run_until_complete

    >>> run(show_errors(gather_excs([x(), e()])))
    RuntimeError: This failed

    A filter can be applied to the specific kind of exception to look
    for. Here, no ValueErrors were raised so there is no output:

    >>> run(show_errors(gather_excs([x(), e()], only=ValueError)))

    Switch it to RuntimeError or a parent class and it will show again:

    >>> run(show_errors(gather_excs([x(), e()], only=RuntimeError)))
    RuntimeError: This failed

    :param aws: Awaitables to gather
    :param only:
        Optional specific type of exceptions to filter on and yield
    """
    for res in await aio.gather(*aws, return_exceptions=True):
        if isinstance(res, only):
            yield res


async def raise_first_exc(aws: Iterable[Awaitable],
                          only: Type[T] = BaseException):
    """
    Gather the given awaitables using :func:`gather_excs` and raise the
    first exception encountered.

    This is useful when you don't need the results of any tasks but need
    to know if there was at least one exception.

    >>> async def x():
    ...     await aio.sleep(0.01)

    >>> async def e():
    ...     await aio.sleep(0.01)
    ...     raise RuntimeError("This failed")

    >>> run = aio.get_event_loop().run_until_complete

    >>> run(raise_first_exc([x(), e()]))
    Traceback (most recent call last):
    ...
    RuntimeError: This failed

    A filter can be applied to the specific kind of exception to look
    for. Here, no ValueErrors were raised so there is no output:

    >>> run(raise_first_exc([x(), e()], only=ValueError))

    Switch it to RuntimeError or a parent class and it will raise again:

    >>> run(raise_first_exc([x(), e()], only=RuntimeError))
    Traceback (most recent call last):
    ...
    RuntimeError: This failed

    :param aws: Awaitables to gather
    :param only:
        Optional specific type of exceptions to filter on and raise
    """
    async for exc in gather_excs(aws, only):
        raise exc


async def to_async_iter(iterable: Iterable[T]) -> AYields[T]:
    """
    Convert the given iterable from synchronous to asynchrounous by
    iterating it in a background thread.

    This can be useful for interacting with libraries for APIs that do
    not have an asynchronous version yet.

    To demonstrate this, let's create a function using Requests which
    makes a synchronous calls to the free {JSON} Placeholder API and
    then yields the results:

    >>> import requests

    >>> def user_names() -> Yields[str]:
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

    # Quick optimization: if it isn't an iterator, just yield from
    # it directly to avoid using a background thread
    if not isinstance(iterable, Iterator):
        for x in iterable:
            yield x
        return

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


def to_sync_iter(iterable: AsyncIterable[T], *, loop: Loop = None) -> Yields[T]:
    """
    Convert the given iterable from asynchronous to synchrounous by
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

    >>> async def user_names() -> AYields[str]:
    ...     for name in await aio.gather(*map(user_name, range(3))):
    ...         yield name

    Now, we can convert this to a synchronous iterator and show the
    results:

    >>> for name in to_sync_iter(user_names()):
    ...     print(name)
    Leanne Graham
    Ervin Howell
    Clementine Bauch

    :param iterable: Asynchonrous iterable to process
    :param loop: Optional specific loop to use to process the iterable
    """

    def _set_loop_and_queue_elements():
        aio.set_event_loop(loop)
        return loop.run_until_complete(_queue_elements())

    async def _queue_elements():
        try:
            async for x in iterable:
                put(x)
        finally:
            put(_DONE)

    if loop is None:
        loop = aio.new_event_loop()

    q = queue.Queue()
    put = q.put_nowait
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(_set_loop_and_queue_elements)
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

    >>> aio.get_event_loop().run_until_complete(aio.sleep(0.5))
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
        #: Arguments are passed as async iterables (the most complex
        #: way of providing arguments) to ensure all elements are
        #: processed on time while avoiding deadlocks.
        self.q: 'aio.Queue[AsyncIterable[T]]' = aio.Queue()
        #: Asyncio event that is set when all current arguments have
        #: been processed and cleared whenever a new one is added.
        self.event = aio.Event()
        self.event.set()

        #: Task which is infinitely processing the queue
        self._waiting = DaemonTask(
            self._waiter(),
            loop=self.loop,
            name=f"Buffering {self.func!r}",
        )
        #: Current task that is waiting for a new element from the queue
        self._getting: Optional[aio.Task] = None

    def __call__(self, _arg: T):
        """
        Place the given argument on the queue to be processed on the
        next execution of the function.
        """
        self._put(_obj_to_aiter(_arg))

    def await_(self, _arg: Awaitable[T]):
        """
        Schedule the given awaitable to be be put onto the queue after
        it has been awaited.

        >>> @buffer_until_timeout(timeout=0.1)
        ... async def buffer(args: Set[int]):
        ...     print("Buffered:", args)

        >>> async def delay(x: T) -> T:
        ...     await aio.sleep(0)
        ...     return x

        >>> for i in range(5):
        ...     buffer.await_(delay(i))

        >>> aio.get_event_loop().run_until_complete(buffer.wait())
        Buffered: {0, 1, 2, 3, 4}
        """
        self._put(_awaitable_to_aiter(_arg))

    def map(self, _args: Iterable[T]):
        """
        Place an iterable of args onto the queue to be processed.

        >>> @buffer_until_timeout(timeout=0.1)
        ... async def buffer(args: Set[int]):
        ...     print("Buffered:", args)

        >>> buffer.map(range(5))

        >>> aio.get_event_loop().run_until_complete(buffer.wait())
        Buffered: {0, 1, 2, 3, 4}

        """
        self._put(to_async_iter(_args))

    def amap(self, _args: AsyncIterable[T]):
        """
        Schedule an async iterable of args to be put onto the queue.

        >>> @buffer_until_timeout(timeout=0.1)
        ... async def buffer(args: Set[int]):
        ...     print("Buffered:", args)

        >>> buffer.amap(to_async_iter(range(5)))

        >>> aio.get_event_loop().run_until_complete(buffer.wait())
        Buffered: {0, 1, 2, 3, 4}

        """
        self._put(_args)

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
        await self.q.join()  # Process all queued tasks
        if cancel and self._getting and not self._getting.done():
            # Cancel the current q.get to avoid waiting for timeout
            # The sleep(0) forces task switching to ensure that
            # _process_queue gets at least one cycle to pull remaining
            # elements off the queue
            await aio.sleep(0)
            self._getting.cancel()
        # Wait for the function to finish processing
        await self.event.wait()

    async def wait_from_anywhere(self, *, cancel: bool = True):
        """
        Wrapper around :meth:`wait` which uses :func:`ensure_aw` to
        handle waiting from a possibly different event loop.
        """
        return await ensure_aw(self.wait(cancel=cancel), self.loop)

    async def _waiter(self):
        """
        Simple loop which tries to process the queue infinitely using
        :meth:`_process_queue`. This is spawned automatically upon
        """
        while True:
            await self._process_queue()

    async def _process_queue(self):
        """
        Internal method to retrieve all current elements from queue and
        execute the function on timeout or cancellation.
        """
        inputs: Set[T] = set()

        async def _load_inputs(iterable: AsyncIterable[T]):
            try:
                async for i in iterable:
                    inputs.add(i)
            except BaseException:  # noqa
                logger.exception("Failed to get args from: %r", iterable)

        # Get first element, block infinitely until one appears
        try:
            input_gens = [_load_inputs(await self.q.get())]
        except RuntimeError:  # Most likely loop shutdown
            return
        else:
            self.event.clear()  # Ensure cleared in case previous cancel
            self.q.task_done()
        # Keep adding new args until the function has run successfully
        while not self.event.is_set():
            # Get all the input generators currently in the queue
            input_gens.extend(map(_load_inputs, self._empty_queue()))
            # Schedule the q.get() and save it as an attribute so it
            # can be cancelled as necessary. This needs to be scheduled
            # *before* waiting for the known inputs.
            self._getting = self._schedule_with_timeout(self.q.get())
            if input_gens:  # Load as many as possible concurrently
                await aio.gather(*input_gens)
                input_gens.clear()  # Clear processed input generators
            # Now wait for the next arguments to buffer. If this times
            # out or is cancelled, its time to run the function.
            try:
                await _load_inputs(await self._getting)
            except (aio.TimeoutError, aio.CancelledError):
                await self._run_func(inputs)
            else:
                self.q.task_done()

    async def _run_func(self, inputs: Set[T]):
        """
        Run :attr:`func` with the given set of inputs and set
        :attr:`event` once it has finished successfully.

        If an exception is raised, log it with its traceback and return
        without setting the event to prevent the buffered inputs from
        being lost.
        """
        try:
            if inputs:  # Could be empty if all empty iterators
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

    def _put(self, iterable: AsyncIterable[T]):
        """
        Helper method to put an async iterable onto the queue and clear
        the event.
        """
        self.event.clear()
        self.loop.call_soon_threadsafe(self.q.put_nowait, iterable)

    def _empty_queue(self) -> Yields[AsyncIterable[T]]:
        """Get all of the elements immediately available in the queue."""
        while True:
            try:
                yield self.q.get_nowait()
            except aio.QueueEmpty:
                break
            else:
                self.q.task_done()


_CROSS_LOOP_POOL = ThreadPoolExecutor(32)


async def ensure_aw(aw: Awaitable[T], loop: Loop) -> T:
    """
    Return an awaitable for the running loop which will ensure the
    given awaitable is evaluated in the given loop. This is useful for
    waiting on events or tasks defined in another thread's event loop.

    .. note::
        The situation in which this function is necessary should be
        avoided when possible, but sometimes it is not always possible.

    >>> e = aio.Event()
    >>> e.set()

    >>> e_loop = aio.get_event_loop()
    >>> new_loop = aio.new_event_loop()
    >>> e_loop is not new_loop
    True

    >>> def task(): return e_loop.create_task(e.wait())

    >>> run = new_loop.run_until_complete
    >>> run(task())  # ValueError: Belongs to a different loop
    Traceback (most recent call last):
    ...
    ValueError: ...

    If the target loop isn't running, it will be run directly using
    another thread:

    >>> run(ensure_aw(task(), e_loop))
    True

    If the target loop is running, it will be used as is using
    :func:`run_aw_threadsafe` internally:

    >>> stop = loop_in_thread(e_loop)

    >>> run(ensure_aw(task(), e_loop))
    True

    >>> stop()  # Cleanup the loop
    """

    main_loop = aio.get_event_loop()

    if main_loop is loop:
        return await aw

    if loop.is_running():
        return await run_aw_threadsafe(aw, loop)

    if loop.is_closed():
        raise RuntimeError("Target loop is closed!")

    def _loop_thread() -> T:
        aio.set_event_loop(loop)
        return loop.run_until_complete(aw)

    return await main_loop.run_in_executor(_CROSS_LOOP_POOL, _loop_thread)


def loop_in_thread(loop: Loop) -> Callable[[], None]:
    """
    Start the given loop in a thread, let it run indefinitely, and
    return a function which can be called to signal the loop to stop.

    >>> from time import sleep

    >>> loop = aio.get_event_loop()

    >>> stop = loop_in_thread(loop)
    >>> sleep(0.05)  # Give the loop a chance to start
    >>> loop.is_running()  # Running in background thread
    True

    >>> loop.call_soon_threadsafe(print, "called")  # doctest: +SKIP
    <Handle print('called')>

    .. testsetup::

        Run after delay to improve test case timing consistency

        >>> loop.call_soon_threadsafe(lambda: sleep(0.01) or print('called'))
        <Handle ...lambda...>

    >>> sleep(0.05)  # Give the loop a chance to process
    called

    >>> stop()  # Signal loop to stop

    >>> sleep(0.05)  # Give it a little bit of time to actually stop

    >>> loop.is_running()  # No longer running
    False
    """
    stop = aio.Event()

    async def _spin():
        aio.set_event_loop(loop)
        await stop.wait()

    Thread(target=loop.run_until_complete, args=(_spin(),), daemon=True).start()

    def _stopper(): loop.call_soon_threadsafe(stop.set)

    return _stopper


async def run_aw_threadsafe(aw: Awaitable[T], loop: Loop) -> T:
    """
    Thin wrapper around :func:`asyncio.run_coroutine_threadsafe` to
    handle any awaitable.

    .. warning::
        This does not extra handling of any event loop conflicts. Use
        :func:`ensure_aw` for that.
    """
    fut = aio.run_coroutine_threadsafe(_aw_to_coro(aw), loop)
    return await aio.wrap_future(fut)


async def _aw_to_coro(aw: Awaitable[T]) -> T:
    """Wrap a given awaitable so it appears as a coroutine."""
    return await aw


async def _obj_to_aiter(o: T) -> AsyncIterable[T]:
    """
    Wrap a given sync object so it is a single element, async iterable
    for compatibility with :attr:`BufferAsyncCalls.q`.
    """
    yield o


async def _awaitable_to_aiter(o: Awaitable[T]) -> AsyncIterable[T]:
    """
    Wrap a awaitable object so it is a single element, async iterable
    for compatibility with :attr:`BufferAsyncCalls.q`.
    """
    yield await o


class DaemonTask(aio.Task):
    """
    Custom :class:`asyncio.Task` which is meant to run forever and
    therefore doesn't warn when it is still pending at loop shutdown.
    """

    if sys.version_info <= (3, 8):

        # Ignore name arg when it isn't available for compatibility
        def __init__(self, coro, *, loop=None, name=None):
            super().__init__(coro, loop=loop)

    # Skip the __del__ defined by aio.Task which does the logging and
    # then calls super()
    __del__ = aio.Task.__base__.__del__  # noqa

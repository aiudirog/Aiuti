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
    'async_background_batcher',
    'AsyncBackgroundBatcher',
    'ensure_aw',
    'loop_in_thread',
    'run_aw_threadsafe',
    'DaemonTask',
]

import sys
import queue
import logging
import asyncio as aio
from asyncio import (
    QueueEmpty as AioQueueEmpty,
    TimeoutError as AioTimeoutError,
    AbstractEventLoop as Loop,
    run_coroutine_threadsafe as run_coro_ts,
)
from itertools import islice
from threading import Lock
from functools import partial, wraps
from concurrent.futures import ThreadPoolExecutor
from weakref import WeakKeyDictionary as WeakKeyDict, finalize
from time import sleep
from typing import (
    Any, AsyncIterable, Awaitable, Callable, Coroutine, Dict,
    MutableMapping, Generic, Iterable, Iterator, List, Optional, Protocol,
    Set, Tuple, Type, TypeVar, Union,
    overload, cast,
)

from .typing import T, Yields, AYields

E = TypeVar('E', bound=BaseException)
X = TypeVar('X')

A_contra = TypeVar('A_contra', contravariant=True)
R_co = TypeVar('R_co', covariant=True)

logger = logging.getLogger(__name__)

_DONE = object()


async def gather_excs(
    aws: Iterable[Awaitable[Any]],
    only: Type[E] = BaseException,  # type: ignore # MyPy bug
) -> AYields[E]:
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

    >>> aio.run(show_errors(gather_excs([x(), e()])))
    RuntimeError: This failed

    A filter can be applied to the specific kind of exception to look
    for. Here, no ValueErrors were raised so there is no output:

    >>> aio.run(show_errors(gather_excs([x(), e()], only=ValueError)))

    Switch it to RuntimeError or a parent class and it will show again:

    >>> aio.run(show_errors(gather_excs([x(), e()], only=RuntimeError)))
    RuntimeError: This failed

    :param aws: Awaitables to gather
    :param only:
        Optional specific type of exceptions to filter on and yield
    """
    for res in await aio.gather(*aws, return_exceptions=True):
        if isinstance(res, only):
            yield res


async def raise_first_exc(aws: Iterable[Awaitable[Any]],
                          only: Type[BaseException] = BaseException) -> None:
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

    >>> aio.run(raise_first_exc([x(), e()]))
    Traceback (most recent call last):
    ...
    RuntimeError: This failed

    A filter can be applied to the specific kind of exception to look
    for. Here, no ValueErrors were raised so there is no output:

    >>> aio.run(raise_first_exc([x(), e()], only=ValueError))

    Switch it to RuntimeError or a parent class and it will raise again:

    >>> aio.run(raise_first_exc([x(), e()], only=RuntimeError))
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

    >>> aio.run(print_user_names())
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

    def _queue_elements() -> None:
        try:
            for x in iterable:
                put(x)
        finally:
            put(_DONE)

    q: 'aio.Queue[Union[T, object]]' = aio.Queue()
    loop = aio.get_running_loop()
    put = partial(loop.call_soon_threadsafe, q.put_nowait)
    with ThreadPoolExecutor(1) as pool:
        future = loop.run_in_executor(pool, _queue_elements)
        while (i := await q.get()) is not _DONE:
            yield i  # type: ignore
        await future  # Bubble any errors


def to_sync_iter(iterable: AsyncIterable[T],
                 *,
                 loop: Optional[Loop] = None) -> Yields[T]:
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
    if loop is None:
        loop = aio.new_event_loop()

    def _set_loop_and_queue_elements(_loop: Loop) -> None:
        aio.set_event_loop(_loop)
        _loop.run_until_complete(_queue_elements())

    async def _queue_elements() -> None:
        try:
            async for x in iterable:
                put(x)
        finally:
            put(_DONE)  # type: ignore

    q: 'queue.Queue[T]' = queue.Queue()
    put = q.put_nowait
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(_set_loop_and_queue_elements, loop)
        try:
            while (i := q.get()) is not _DONE:
                yield i
        finally:
            future.result()


_CacheMap = MutableMapping[Tuple[Any, ...], Any]
_AsyncFunc = TypeVar('_AsyncFunc', bound=Callable[..., Awaitable[Any]])


@overload
def threadsafe_async_cache(
    func: None = None,
    *,
    cache: Optional[_CacheMap] = None,
) -> Callable[[_AsyncFunc], _AsyncFunc]: ...


@overload
def threadsafe_async_cache(
    func: _AsyncFunc,
    *,
    cache: Optional[_CacheMap] = None,
) -> _AsyncFunc: ...


def threadsafe_async_cache(
    func: Optional[_AsyncFunc] = None,
    *,
    cache: Optional[_CacheMap] = None,
) -> Union[Callable[[_AsyncFunc], _AsyncFunc], _AsyncFunc]:
    """
    Simple thread-safe asynchronous cache decorator which ensures that
    the decorated/wrapped function is only ever called once for a given
    set of inputs even across different event loops in multiple threads.

    To demonstrate this, we'll define an async function which sleeps
    to simulate work and then returns the square of the input:

    >>> import asyncio as aio

    >>> @threadsafe_async_cache
    ... async def square(x: int) -> int:
    ...     await aio.sleep(0.1)
    ...     print("Squaring:", x)
    ...     return x * x

    Then we'll define a function that will use a new event loop to call
    the function 10 times in parallel:

    >>> async def square_10x(x: int) -> List[int]:
    ...     return await aio.gather(*(square(x) for _ in range(10)))

    And finally we'll call that function 10 times in parallel across 10
    threads:

    >>> with ThreadPoolExecutor(max_workers=10) as pool:
    ...     results = list(pool.map(lambda x: aio.run(square_10x(x)), [2] * 10))
    Squaring: 2

    As can be seen above, the function was only called once despite
    being invoked 100 times across 10 threads and 10 event loops. This
    can be confirmed by counting the total number of results:

    >>> len([x for row in results for x in row])
    100

    Additionally, a custom mutable mapping can be provided for the cache
    to further customizize the implementation. For example, we can use
    the `lru-dict <https://pypi.org/project/lru-dict/>`__ library to
    limit the size of the cache:

    >>> from lru import LRU
    >>> cache = LRU(size=3)

    >>> @threadsafe_async_cache(cache=cache)
    ... async def double(x: int) -> int:
    ...     await aio.sleep(0.1)
    ...     print("Doubling:", x)
    ...     return x * 2

    >>> assert aio.run(double(1)) == 2
    Doubling: 1
    >>> assert aio.run(double(1)) == 2  # Cached

    >>> assert aio.run(double(2)) == 4
    Doubling: 2
    >>> assert aio.run(double(2)) == 4  # Cached

    >>> assert aio.run(double(3)) == 6
    Doubling: 3
    >>> assert aio.run(double(3)) == 6  # Cached

    >>> assert aio.run(double(4)) == 8  # Pushes 1 out of the cache
    Doubling: 4
    >>> assert aio.run(double(4)) == 8  # Cached

    >>> assert aio.run(double(1)) == 2  # No longer cached!
    Doubling: 1

    .. warning::
       The default cache is a simple dictionary and as such has no max
       size and can very easily be the source of memory leaks.
       I typically use this to wrap object methods on a per-instance
       basis during intilaztion so that the cache only lives as long
       as the object.
    """
    if func is None:
        return partial(  # type: ignore[return-value]
            threadsafe_async_cache,
            cache=cache,
        )

    # Avoid type narrowing issues related to:
    # https://github.com/python/mypy/issues/13123
    _cache: _CacheMap = cache if cache is not None else {}
    _func: _AsyncFunc = func
    del cache, func

    # 1 loop + event per input key currently caching
    events: Dict[Tuple[Any, ...], Tuple[aio.AbstractEventLoop, aio.Event]] = {}
    # Ensure thread safety while creating events
    event_making_lock = Lock()

    @wraps(_func)
    async def _wrapper(*args: Any, **kwargs: Any) -> Any:
        # Get the key from the input arguments
        key = args, frozenset(kwargs.items())

        running_loop = aio.get_running_loop()

        while True:
            # Avoid locking during this first check for performance
            try:  # to get the value from the cache
                return _cache[key]
            except KeyError:
                pass

            # Need to calculate and cache the value once

            with event_making_lock:
                try:  # verify nothing cached while waiting for lock
                    return _cache[key]
                except KeyError:
                    pass

                try:
                    # Try to get the loop + event of the loop currently
                    # caching the value
                    caching_loop, event = events[key]
                    if (caching_loop.is_closed()
                            or not caching_loop.is_running()):
                        raise KeyError  # Invalidate loop
                except KeyError:
                    # No existing event -> this task is going to cache
                    # the value and provide an event for others to wait
                    caching_loop = aio.get_running_loop()
                    event = aio.Event()
                    events[key] = caching_loop, event
                    do_caching = True
                else:
                    do_caching = False  # Need to wait for other loop

            if do_caching:  # No other task to wait for, cache the value
                try:
                    result = await _func(*args, **kwargs)
                except Exception:
                    raise  # Bubble any errors without caching
                else:
                    _cache[key] = result  # Cache for other tasks
                finally:
                    with event_making_lock:
                        # Wake up any waiting tasks
                        event.set()
                        # Allow garbage collection and/or another loop
                        # to take over caching if this failed
                        del events[key]
                return result

            # Need to wait for another task, possibly across threads

            wait_event: Awaitable[bool] = event.wait()
            if running_loop is not caching_loop:
                try:
                    wait_fut = run_coro_ts(wait_event, caching_loop)
                except RuntimeError:  # caching loop most likely closed
                    continue  # loop around and try again
                wait_event = aio.wrap_future(wait_fut)

            # Wrap anything waiting for the event in a long timeout just
            # to ensure nothing hangs completely if the original task is
            # lost and somehow never sets the event
            waiter = aio.create_task(aio.wait_for(wait_event, 60))

            # wait_for will swallow CancelledErrors if the wrapped task
            # is already finished. This can cause confusion and missed
            # timeouts when there are multiple nested wait_fors.
            # Shielding and handling the cancellation directly avoids
            # this confusion by preventing this wait_for from seeing
            # the outer CancelledError directly.
            try:
                await aio.shield(waiter)
            except aio.TimeoutError:  # Possible original task lost?
                pass  # Need to loop around and check
            except (Exception, aio.CancelledError):
                if not waiter.done():
                    waiter.cancel()
                    try:
                        await waiter
                    except aio.CancelledError:
                        pass
                raise

    return _wrapper  # type: ignore[return-value]


_BufferFunc = Callable[[Set[T]], Awaitable[None]]


@overload
def buffer_until_timeout(
    *,
    timeout: float = 1,
) -> 'Callable[[_BufferFunc[T]], BufferAsyncCalls[T]]':
    ...


@overload
def buffer_until_timeout(
    func: _BufferFunc[T],
    *,
    timeout: float = 1,
) -> 'BufferAsyncCalls[T]':
    ...


def buffer_until_timeout(
    func: Optional[_BufferFunc[T]] = None,
    *,
    timeout: float = 1,
) -> Union['BufferAsyncCalls[T]',
           'Callable[[_BufferFunc[T]], BufferAsyncCalls[T]]']:
    """
    Async function decorator/wrapper which buffers the arg passed in
    each call. After a given timeout has passed since the last call,
    the function is invoked on the event loop that the function was
    decorated nearest.

    >>> loop = aio.new_event_loop()
    >>> aio.set_event_loop(loop)

    >>> @buffer_until_timeout(timeout=0.1)
    ... async def buffer(args: Set[int]):
    ...     print("Buffered:", args)

    >>> for i in range(5):
    ...     buffer(i)

    >>> loop.run_until_complete(aio.sleep(0.5))
    Buffered: {0, 1, 2, 3, 4}

    Another function can also wait for all elements to be processed
    before continueing:

    >>> async def after_buffered():
    ...     await buffer.wait()
    ...     print("All buffered!")

    >>> for i in range(5):
    ...     buffer(i)

    >>> loop.run_until_complete(after_buffered())
    Buffered: {0, 1, 2, 3, 4}
    All buffered!

    :param func: Function to wrap
    :param timeout:
        Number of seconds to wait after the most recent call before
        executing the function.
    :return: Non-async function which will buffer the given argument
    """
    if func is None:
        return partial(buffer_until_timeout, timeout=timeout)  # type: ignore
    return wraps(func)(BufferAsyncCalls(func, timeout=timeout))  # type: ignore


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

    def __init__(self, func: _BufferFunc[T], *, timeout: float = 1):
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
        self._getting: Optional['aio.Task[AsyncIterable[T]]'] = None

    def __call__(self, _arg: T) -> None:
        """
        Place the given argument on the queue to be processed on the
        next execution of the function.
        """
        self._put(_obj_to_aiter(_arg))

    def await_(self, _arg: Awaitable[T]) -> None:
        """
        Schedule the given awaitable to be be put onto the queue after
        it has been awaited.

        >>> loop = aio.new_event_loop()
        >>> aio.set_event_loop(loop)

        >>> @buffer_until_timeout(timeout=0.1)
        ... async def buffer(args: Set[int]):
        ...     print("Buffered:", args)

        >>> async def delay(x: T) -> T:
        ...     await aio.sleep(0)
        ...     return x

        >>> for i in range(5):
        ...     buffer.await_(delay(i))

        >>> loop.run_until_complete(buffer.wait())
        Buffered: {0, 1, 2, 3, 4}
        """
        self._put(_awaitable_to_aiter(_arg))

    def map(self, _args: Iterable[T]) -> None:
        """
        Place an iterable of args onto the queue to be processed.

        >>> loop = aio.new_event_loop()
        >>> aio.set_event_loop(loop)

        >>> @buffer_until_timeout(timeout=0.1)
        ... async def buffer(args: Set[int]):
        ...     print("Buffered:", args)

        >>> buffer.map(range(5))

        >>> loop.run_until_complete(buffer.wait())
        Buffered: {0, 1, 2, 3, 4}

        """
        self._put(to_async_iter(_args))

    def amap(self, _args: AsyncIterable[T]) -> None:
        """
        Schedule an async iterable of args to be put onto the queue.

        >>> loop = aio.new_event_loop()
        >>> aio.set_event_loop(loop)

        >>> @buffer_until_timeout(timeout=0.1)
        ... async def buffer(args: Set[int]):
        ...     print("Buffered:", args)

        >>> buffer.amap(to_async_iter(range(5)))

        >>> loop.run_until_complete(buffer.wait())
        Buffered: {0, 1, 2, 3, 4}

        """
        self._put(_args)

    async def wait(self, *, cancel: bool = True) -> None:
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
        # Process all queued tasks
        # Create a task because some unknown bug on 3.7 blocks
        # infinitely sometimes if we await join directly
        # FIXME: Evaluate if the bug is possible on 3.8+?
        await self.loop.create_task(self.q.join())
        if cancel and self._getting and not self._getting.done():
            # Cancel the current q.get to avoid waiting for timeout
            # The sleep(0) forces task switching to ensure that
            # _process_queue gets at least one cycle to pull remaining
            # elements off the queue
            await aio.sleep(0)
            self._getting.cancel()
        # Wait for the function to finish processing
        await self.event.wait()

    async def wait_from_anywhere(self, *, cancel: bool = True) -> None:
        """
        Wrapper around :meth:`wait` which uses :func:`ensure_aw` to
        handle waiting from a possibly different event loop.
        """
        return await ensure_aw(self.wait(cancel=cancel), self.loop)

    async def _waiter(self) -> None:
        """
        Simple loop which tries to process the queue infinitely using
        :meth:`_process_queue`. This is spawned automatically upon
        """
        while True:
            await self._process_queue()

    async def _process_queue(self) -> None:
        """
        Internal method to retrieve all current elements from queue and
        execute the function on timeout or cancellation.
        """
        inputs: Set[T] = set()

        async def _load_inputs(iterable: AsyncIterable[T]) -> None:
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

    async def _run_func(self, inputs: Set[T]) -> None:
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

    def _schedule_with_timeout(self, coro: Awaitable[X]) -> 'aio.Task[X]':
        """
        Helper method to create a task for the given coroutine with the
        configured :attr:`timeout`.
        """
        return self.loop.create_task(aio.wait_for(coro, self.timeout))

    def _put(self, iterable: AsyncIterable[T]) -> None:
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


_BatchFunc = Callable[
    [Iterable[Tuple[str, A_contra]]],
    AsyncIterable[Tuple[str, Union[R_co, Exception]]],
]

_TaskTuple = Tuple[str, A_contra, 'aio.Future[R_co]']


class _AsyncBackgroundBatcherProto(Protocol[A_contra, R_co]):

    async def __call__(self,
                       arg: A_contra,
                       *,
                       key: Optional[str] = None) -> R_co: ...


@overload
def async_background_batcher(
    func: None = None,
    *,
    max_batch_size: int = ...,
    max_concurrent_batches: int = ...,
    batch_timeout: float = ...,
) -> Callable[
    [_BatchFunc[A_contra, R_co]],
    _AsyncBackgroundBatcherProto[A_contra, R_co],
]: ...


@overload
def async_background_batcher(
    func: _BatchFunc[A_contra, R_co],
    *,
    max_batch_size: int = ...,
    max_concurrent_batches: int = ...,
    batch_timeout: float = ...,
) -> _AsyncBackgroundBatcherProto[A_contra, R_co]: ...


def async_background_batcher(
    func: Optional[_BatchFunc[A_contra, R_co]] = None,
    *,
    max_batch_size: int = 256,
    max_concurrent_batches: int = 5,
    batch_timeout: float = 0.05,
    retention_timeout: float = 0.,
) -> Union[
    Callable[
        [_BatchFunc[A_contra, R_co]],
        _AsyncBackgroundBatcherProto[A_contra, R_co],
    ],
    _AsyncBackgroundBatcherProto[A_contra, R_co],
]:
    """
    Decorator version of :class:`AsyncBackgroundBatcher` which also
    handles automatically creating a new batcher for individual event
    loops allowing declaration outside a running event loop.

    Here is the example from :class:`AsyncBackgroundBatcher` rewritten
    using the decorator syntax:

    >>> @async_background_batcher(max_batch_size=2)
    ... async def add_1(
    ...     batch: Iterable[Tuple[str, int]],
    ... ) -> AsyncIterable[Tuple[str, int]]:
    ...     print("Starting batch:", batch)
    ...     for key, value in batch:
    ...         await aio.sleep(0.05)
    ...         yield key, value + 1
    ...     print("Finished batch:", batch)

    Now we can directly call the add_1 function before the event loop
    has started, which can be more convenient:

    >>> async def main():
    ...     return await aio.gather(
    ...         add_1(1),
    ...         add_1(2),
    ...         add_1(3),
    ...         add_1(4, key='fourth'),
    ...     )
    >>> aio.run(main())
    Starting batch: [('1', 1), ('2', 2)]
    Starting batch: [('3', 3), ('fourth', 4)]
    Finished batch: [('1', 1), ('2', 2)]
    Finished batch: [('3', 3), ('fourth', 4)]
    [2, 3, 4, 5]

    If requests for the same key come through in quick succession, the result
    will only be computed once:

    >>> async def dups():
    ...     return await aio.gather(
    ...         add_1(1), add_1(1), add_1(1),
    ...         add_1(1, key='one'), add_1(1, key='one'),
    ...     )

    >>> aio.run(dups())
    Starting batch: [('1', 1), ('one', 1)]
    Finished batch: [('1', 1), ('one', 1)]
    [2, 2, 2, 2, 2]

    This can help avoid some headaches when there is potential for multiple
    concurrent tasks to request the same resources.

    .. note::
       This method has a slight amount of extra overhead due to the
       per-argument loop batcher lookups. Initializing an
       :class:`AsyncBackgroundBatcher` directly inside a running event
       loop can avoid this, but most applications will not notice.
    """
    if func is None:
        return partial(
            async_background_batcher,  # type: ignore
            max_batch_size=max_batch_size,
            max_concurrent_batches=max_concurrent_batches,
            batch_timeout=batch_timeout,
        )

    batchers: 'WeakKeyDict[Loop, AsyncBackgroundBatcher[A_contra, R_co]]' \
        = WeakKeyDict()

    async def _wrapper(arg: A_contra,
                       *,
                       key: Optional[str] = None) -> R_co:
        loop = aio.get_running_loop()
        try:
            batcher = batchers[loop]
        except KeyError:
            batcher = batchers[loop] = AsyncBackgroundBatcher(
                cast(_BatchFunc[A_contra, R_co], func),
                max_batch_size=max_batch_size,
                max_concurrent_batches=max_concurrent_batches,
                batch_timeout=batch_timeout,
                retention_timeout=retention_timeout,
            )
        return await batcher(arg, key=key)

    for attr in ('__module__', '__name__', '__qualname__', '__doc__'):
        try:
            setattr(_wrapper, attr, getattr(func, attr))
        except AttributeError:
            pass
    return _wrapper


class AsyncBackgroundBatcher(Generic[A_contra, R_co]):
    """
    Allow single async executions of a function to be batched in the
    background and submitted together with individual results being
    properly returned to the original callers/waiters.

    A good usecase for this is single-row database lookups/mutations
    that can be submitted in bulk to reduce the number of round-trips
    to the server but are often easier to write as seperate, per-row
    tasks. Using this class for background batching allows the best of
    both worlds.

    Here is a simple example which adds one to the given input.

    First, we'll define the batch function which is asynchronous and
    takes each batch as an iterable of tuples containing each task key
    and argument. The batch function should then yield the results as
    tuples containing the task key and the result. Because the key is
    returned, results can be yielded in any order. To indicate an error
    occurred, simply yield an :class:`Exception` object instead of a
    normal result.

    >>> async def add_1(
    ...     batch: Iterable[Tuple[str, int]],
    ... ) -> AsyncIterable[Tuple[str, int]]:
    ...     print("Starting batch:", batch)
    ...     for key, value in batch:
    ...         await aio.sleep(0.05)
    ...         yield key, value + 1
    ...     print("Finished batch:", batch)

    To utilize the batched execution, first create an instance of
    :class:`AsyncBackgroundBatcher` with the batch function and any additional
    configuration kwargs. Then, the returned object can be called for
    a single execution. To pass a custom key, utilize the ``key`` kwarg:

    >>> async def main():
    ...     batched = AsyncBackgroundBatcher(add_1, max_batch_size=2)
    ...     return await aio.gather(
    ...         batched(1),
    ...         batched(2),
    ...         batched(3),
    ...         batched(4, key='fourth'),
    ...     )

    >>> aio.run(main())
    Starting batch: [('1', 1), ('2', 2)]
    Starting batch: [('3', 3), ('fourth', 4)]
    Finished batch: [('1', 1), ('2', 2)]
    Finished batch: [('3', 3), ('fourth', 4)]
    [2, 3, 4, 5]

    As can be seen above, the batches were started in order and then ran
    concurrently with the individual results returned to the original
    caller.

    .. warning::
       Awaits after the final yield may not execute as the processing of
       each batch is spun off as a background task and will not prevent
       the event loop from closing.

    If requests for the same key come through in quick succession, the result
    will only be computed once:

    >>> async def dups():
    ...     batched = AsyncBackgroundBatcher(add_1, max_batch_size=2)
    ...     return await aio.gather(
    ...         batched(1), batched(1), batched(1),
    ...         batched(1, key='one'), batched(1, key='one'),
    ...     )

    >>> aio.run(dups())
    Starting batch: [('1', 1), ('one', 1)]
    Finished batch: [('1', 1), ('one', 1)]
    [2, 2, 2, 2, 2]

    This can help avoid some headaches when there is potential for multiple
    concurrent tasks to request the same resources.

    """

    #: Async batch execution function which takes the task argument
    #: tuples and yields task result tuples
    func: _BatchFunc[A_contra, R_co]
    #: Maximum number of elements in each batch.
    #: Can be safely mutated after initialization.
    max_batch_size: int
    #: Number of seconds to wait for more elements before an
    #: incomplete batch is processed
    batch_timeout: float
    #: Number of seconds to keep results from completed batches around
    #: to avoid duplicate requests
    retention_timeout: float

    #: Queue used to buffer all tasks to be executed
    _queue: 'aio.Queue[_TaskTuple[A_contra, R_co]]'
    #: Event loop the batcher is currently running in
    _loop: Loop
    #: Semaphore used to prevent to limit the number of concurrent
    #: executions of the batch function
    _semaphore: aio.Semaphore
    #: Task running the main loop that is waiting for batches and
    #: spawning sub-tasks to process them
    _loop_task: 'DaemonTask'
    #: Cache for recently completed futures to avoid duplicate requests
    _retention_cache: 'dict[str, aio.Future[R_co]]'

    def __init__(self,
                 func: _BatchFunc[A_contra, R_co],
                 *,
                 max_batch_size: int = 256,
                 max_concurrent_batches: int = 5,
                 batch_timeout: float = 0.05,
                 retention_timeout: float = 0.):
        """
        :param func: Batch execution function
        :param max_batch_size:
            Maximum size for each batch passed to `func`
        :param max_concurrent_batches:
            Maximum number of concurrent executions of `func`
        :param batch_timeout:
            Number of seconds to wait for more elements before an
            incomplete batch is processed
        :param retention_timeout:
            Number of seconds to keep results from completed batches around
            to avoid duplicate requests
        """
        self.func = func

        self._queue = aio.Queue()
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout

        self.retention_timeout = retention_timeout
        self._retention_cache = {}

        self._semaphore = aio.Semaphore(value=max_concurrent_batches)

        self._loop = aio.get_running_loop()
        self._loop_task = self._daemon_task(
            self._processing_loop(),
            name="async-bg-batcher-processing-loop",
        )

    async def __call__(self,
                       arg: A_contra,
                       *,
                       key: Optional[str] = None) -> R_co:
        if key is None:
            key = str(arg)

        fut: 'aio.Future[R_co]'

        try:
            fut = self._retention_cache[key]
        except KeyError:
            pass
        else:
            return await fut

        fut = self._retention_cache[key] = self._loop.create_future()
        await self._queue.put((key, arg, fut))

        try:
            return await fut
        finally:
            if self.retention_timeout > 0:
                self._loop.call_later(
                    self.retention_timeout,
                    self._retention_cache.pop,
                    key,
                )
            else:
                del self._retention_cache[key]

    def _daemon_task(
        self,
        coro: Coroutine[Any, Any, Any],
        name: Optional[str] = None,
    ) -> 'DaemonTask':
        """
        Helper method to create a :class:`DaemonTask` instance in the
        current event loop.
        """
        return DaemonTask(coro, loop=self._loop, name=name)

    async def _processing_loop(self) -> None:
        while True:
            tasks = await self._get_next_batch()
            # Don't wait for the current batch to finish
            self._daemon_task(  # noqa
                self._process_batch(tasks),
                name="async-bg-batcher-process-batch",
            )

    async def _get_next_batch(self) -> List['_TaskTuple[A_contra, R_co]']:
        """
        Helper method to get the next batch of tasks to process off the
        queue. Each batch is a list of tuples containing the task key,
        argument, and future for the result.
        """
        q = self._queue
        # Wait for the first task to appear in the queue
        try:
            tasks = [await q.get()]
        except RuntimeError as e:
            if str(e).lower() == 'event loop is closed':
                return []  # Don't spam errors when shutting down
            raise

        while len(tasks) < self.max_batch_size:
            # Grab all available tasks as efficiently as possible
            # Yes - this is a micro-optimization
            try:
                tasks.extend(islice(
                    iter(q.get_nowait, object()),
                    self.max_batch_size - len(tasks),
                ))
            except AioQueueEmpty:
                pass
            else:
                continue
            # Give the publisher a little breathing room to actually
            # populate the queue before declaring it empty
            try:
                tasks.append(await aio.wait_for(q.get(), self.batch_timeout))
            except AioTimeoutError:  # No more tasks coming
                break
        return tasks

    async def _process_batch(
        self,
        tasks: List['_TaskTuple[A_contra, R_co]'],
    ) -> None:
        """
        Process a given batch of task tuples. A list of tuples of task
        key and arg will be passed to the function which will yield
        tuples of task keys and result values. If the result is an
        :class:`Exception`, the future will be marked as failed with
        that exception. Any other type of result will be treated as
        success and set as the future's result.

        If the batch function raises an error, all futures will have
        that exception set.
        """
        args = [t[:2] for t in tasks]
        futs = {k: f for k, _, f in tasks}

        try:
            async with self._semaphore:  # Limit concurrent executions
                logger.debug(
                    "Sending batch of size %d to %s",
                    len(args), self.func,
                )
                async for key, result in self.func(args):
                    fut = futs.pop(key)
                    if isinstance(result, Exception):
                        fut.set_exception(result)
                    else:
                        fut.set_result(result)
        except Exception as e:
            logger.debug("Exception while processing batch", exc_info=True)
            for fut in futs.values():
                fut.set_exception(e)
            return

        if futs:
            logger.error("Missing outputs for %d futures", len(futs))
            for key, fut in futs.items():
                fut.set_exception(ValueError(f"Missing result for {key!r}"))


_CROSS_LOOP_POOL = ThreadPoolExecutor(32)


async def ensure_aw(aw: Awaitable[T], loop: Loop) -> T:
    """
    Return an awaitable for the running loop which will ensure the
    given awaitable is evaluated in the given loop. This is useful for
    waiting on events or tasks defined in another thread's event loop.

    .. note::
        The situation in which this function is necessary should be
        avoided when possible, but sometimes it is not always possible.

    >>> e_loop = aio.new_event_loop()
    >>> aio.set_event_loop(e_loop)

    >>> e = aio.Event()
    >>> e.set()

    >>> new_loop = aio.new_event_loop()

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

    main_loop = aio.get_running_loop()

    if main_loop is loop:
        return await aw

    if loop.is_running():
        return await run_aw_threadsafe(aw, loop)

    if loop.is_closed():
        raise RuntimeError("Target loop is closed!")

    def _loop_thread() -> T:
        with _get_loop_lock(loop):
            aio.set_event_loop(loop)
            return loop.run_until_complete(aw)

    return await main_loop.run_in_executor(_CROSS_LOOP_POOL, _loop_thread)


def loop_in_thread(loop: Loop) -> Callable[[], None]:
    """
    Start the given loop in a thread, let it run indefinitely, and
    return a function which can be called to signal the loop to stop.

    >>> from time import sleep

    >>> loop = aio.new_event_loop()

    >>> stop = loop_in_thread(loop)
    >>> loop.is_running()  # Running in background thread
    True

    >>> loop.call_soon_threadsafe(print, "called")  # doctest: +SKIP
    <Handle print('called')>

    .. testsetup::

        Run after delay to improve test case timing consistency

        >>> loop.call_soon_threadsafe(lambda: sleep(0.1) or print('called'))
        <Handle ...lambda...>

    >>> sleep(0.3)  # Give the loop a chance to process
    called

    >>> stop()  # Signal loop to stop

    >>> loop.is_running()  # No longer running
    False
    """
    def _loop_thread() -> None:
        with _get_loop_lock(loop):
            aio.set_event_loop(loop)
            loop.run_forever()

    future = _CROSS_LOOP_POOL.submit(_loop_thread)

    while not loop.is_running():
        sleep(0)  # Force switching to other threads

    def _stopper() -> None:
        loop.call_soon_threadsafe(loop.stop)
        future.result()  # Wait for loop to exit and reveal errors

    return _stopper


_LOOP_LOCKS: Dict[int, Lock] = {}
_LOOP_LOCKS_CREATE_LOCK = Lock()


def _get_loop_lock(loop: aio.AbstractEventLoop) -> Lock:
    key = id(loop)
    try:  # Check to see if lock is already created
        return _LOOP_LOCKS[key]
    except KeyError:
        pass

    with _LOOP_LOCKS_CREATE_LOCK:  # Ensure atomic creation
        try:  # Handle case where another thread acquired lock first
            return _LOOP_LOCKS[key]
        except KeyError:  # FIRST! Create the lock
            lock = _LOOP_LOCKS[key] = Lock()
            # Ensure the lock is cleaned up when the loop is destroyed
            finalize(loop, _LOOP_LOCKS.pop, key, None)
            return lock


async def run_aw_threadsafe(aw: Awaitable[T], loop: Loop) -> T:
    """
    Thin wrapper around :func:`asyncio.run_coroutine_threadsafe` to
    handle any awaitable.

    .. warning::
        This does not handle event loop conflicts.
        Use :func:`ensure_aw` for that.
    """
    coro = aw if aio.iscoroutine(aw) else _aw_to_coro(aw)
    return await aio.wrap_future(run_coro_ts(coro, loop))


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


class DaemonTask(aio.Task):  # type: ignore
    """
    Custom :class:`asyncio.Task` which is meant to run forever and
    therefore doesn't warn when it is still pending at loop shutdown.
    """

    if sys.version_info < (3, 8):

        # Ignore name arg when it isn't available for compatibility
        def __init__(self, coro, *, loop=None, name=None):
            super().__init__(coro, loop=loop)

    # Skip the __del__ defined by aio.Task which does the logging and
    # then calls super()
    __del__ = aio.Task.__base__.__del__  # type: ignore

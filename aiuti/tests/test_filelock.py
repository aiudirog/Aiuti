"""
Tests adapted and expanded from:
https://github.com/benediktschmitt/py-filelock/blob
/b30bdc4fb998f5f4350257235eb51147f9e81862/test.py
"""

import gc
from pathlib import Path
from itertools import repeat, chain
from concurrent.futures import ThreadPoolExecutor

import pytest

from aiuti.filelock import FileLock
from aiuti.typing import Yields


@pytest.fixture
def lock_file(tmp_path: Path) -> Yields[str]:
    """Temporary filepath for testing the lock."""
    lock_file = tmp_path / 'file.lock'
    if lock_file.exists():
        lock_file.unlink()
    yield str(lock_file)
    if lock_file.exists():
        lock_file.unlink()


@pytest.fixture
def flock(lock_file: str) -> FileLock:
    """Default file lock for testing."""
    return FileLock(lock_file)


@pytest.fixture
def flock2(lock_file: str) -> FileLock:
    """Second file lock pointing to the same lock for testing."""
    return FileLock(lock_file)


@pytest.fixture
def frlock(lock_file: str) -> FileLock:
    """Reentrant file lock."""
    return FileLock(lock_file, reentrant=True)


@pytest.fixture
def pool() -> Yields[ThreadPoolExecutor]:
    """Thread pool with 100 maximum concurrent tasks."""
    with ThreadPoolExecutor(100) as pool:
        yield pool


def test_simple(flock: FileLock) -> None:
    """
    Asserts that the lock is locked in a context statement and that the
    return value of __enter__() is the lock.
    """
    with flock as lock:
        assert flock.is_locked
        assert flock is lock
    assert not flock.is_locked


def test_acquire_ctx(flock: FileLock) -> None:
    """Same as test_simple but using acquire_ctx."""
    with flock.acquire_ctx() as lock:
        assert flock.is_locked
        assert lock is None
    assert not flock.is_locked


def test_nested(frlock: FileLock) -> None:
    """
    Asserts that the reentrant lock is not released before the most
    outer with statement that locked the lock, is left.
    """
    with frlock as l1:
        assert frlock.is_locked
        assert frlock is l1

        with frlock as l2:
            assert frlock.is_locked
            assert frlock is l2

            with frlock as l3:
                assert frlock.is_locked
                assert frlock is l3

            assert frlock.is_locked
        assert frlock.is_locked
    assert not frlock.is_locked


def test_nested_explicit_acquire(frlock: FileLock) -> None:
    """
    The same as test_nested, but uses the acquire() and release()
    directly rather than context managers.
    """
    assert frlock.acquire()
    assert frlock.is_locked

    assert frlock.acquire()
    assert frlock.is_locked

    assert frlock.acquire()
    assert frlock.is_locked

    frlock.release()
    assert frlock.is_locked

    frlock.release()
    assert frlock.is_locked

    frlock.release()
    assert not frlock.is_locked


def test_nested_forced_release(frlock: FileLock) -> None:
    """
    Acquires the lock using a context manager and then again inside and
    releases it before leaving.
    """
    with frlock:
        assert frlock.is_locked

        frlock.acquire()
        assert frlock.is_locked

        frlock.release(force=True)
        assert not frlock.is_locked

    assert not frlock.is_locked


def test_threaded(flock: FileLock, pool: ThreadPoolExecutor) -> None:
    """
    Runs 250 threads, which need the filelock. The lock must be acquired
    by all threads and released once all are done.
    """
    def get_lock_repeatedly(count: int) -> None:
        for _ in range(count):
            with flock:
                assert flock.is_locked

    pool.map(get_lock_repeatedly, repeat(100, 250))
    assert not flock.is_locked


def test_threaded_duplicate_lock(flock: FileLock,
                                 flock2: FileLock,
                                 pool: ThreadPoolExecutor) -> None:
    """
    Runs multiple threads, which acquire the same lock file with
    different FileLock objects. When lock1 is held, lock2 must not be
    held and vice versa.
    """

    def acquire_lock1() -> None:
        for _ in range(100):
            with flock:
                assert flock.is_locked
                assert not flock2.is_locked

    def acquire_lock2() -> None:
        for _ in range(100):
            with flock2:
                assert not flock.is_locked
                assert flock2.is_locked

    def acquire_lock(which: int) -> None:
        acquire_lock1() if which == 1 else acquire_lock2()

    pool.map(acquire_lock, chain.from_iterable(repeat((1, 2), 250)))

    assert not flock.is_locked
    assert not flock2.is_locked


def test_nonblocking(flock: FileLock) -> None:
    """
    Verify lock is not acquired with blocking=False and acquire returns
    False.
    """
    flock.acquire()
    assert flock.is_locked

    assert not flock.acquire(blocking=False)
    assert flock.is_locked

    flock.release()
    assert not flock.is_locked


def test_nonblocking_multiple_locks(flock: FileLock, flock2: FileLock) -> None:
    """Same as test_nonblocking but with multiple file locks."""
    flock.acquire()
    assert flock.is_locked

    assert not flock2.acquire(blocking=False)
    assert flock.is_locked
    assert not flock2.is_locked

    flock.release()
    assert not flock.is_locked


def test_timeout(flock: FileLock) -> None:
    """Verify lock is not acquired on timeout and acquire returns False."""
    flock.acquire()
    assert flock.is_locked

    assert not flock.acquire(timeout=0.1)
    assert flock.is_locked

    flock.release()
    assert not flock.is_locked


def test_timeout_different_locks(flock: FileLock, flock2: FileLock) -> None:
    """
    Same as test_timeout but with two different locks pointing to the
    same file.
    """
    flock.acquire()
    assert flock.is_locked
    assert not flock2.is_locked

    assert not flock2.acquire(timeout=0.1)
    assert flock.is_locked
    assert not flock2.is_locked

    flock.release()
    assert not flock.is_locked
    assert not flock2.is_locked


def test_default_timeout(lock_file: str, flock: FileLock) -> None:
    """Test if the default timeout parameter works."""
    flock_to = FileLock(lock_file, timeout=0.1)

    assert flock_to.timeout == 0.1

    flock.acquire()
    assert flock.is_locked
    assert not flock_to.is_locked

    assert not flock_to.acquire()
    assert flock.is_locked
    assert not flock_to.is_locked

    flock_to.timeout = 0
    assert not flock_to.acquire()
    assert flock.is_locked
    assert not flock_to.is_locked

    flock.release()
    assert not flock.is_locked
    assert not flock_to.is_locked


def test_timeout_acquire_ctx(flock: FileLock) -> None:
    """Test TimeoutError is raised when acquire_ctx can't acquire the lock."""
    flock.acquire()
    with pytest.raises(TimeoutError):
        with flock.acquire_ctx(timeout=0.1):
            pass
    flock.release()


def test_context_exception(flock: FileLock) -> None:
    """
    Verify filelock is released if an exception is raised when used in a
    context.
    """
    try:
        with flock:
            raise BaseException
    except BaseException:  # noqa
        assert not flock.is_locked


def test_context_exception_acquire_ctx(flock: FileLock) -> None:
    """The same as test_context_exception, but uses the acquire_ctx."""
    try:
        with flock.acquire_ctx():
            raise BaseException
    except BaseException:  # noqa
        assert not flock.is_locked


def test_del(lock_file: str, flock: FileLock) -> None:
    """Verify lock is released during garbage collection."""
    tmp_flock = FileLock(lock_file)

    tmp_flock.acquire()
    assert tmp_flock.is_locked
    assert not flock.is_locked

    assert not flock.acquire(timeout=0.1)

    del tmp_flock
    gc.collect()  # Ensure gc runs, PyPy doesn't invoke it on del

    assert flock.acquire(timeout=1)
    assert flock.is_locked

    flock.release()
    assert not flock.is_locked

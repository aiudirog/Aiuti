"""
FileLock
========

Thread-safe, platform independent file lock supporting the interface
as the standard :class:`threading.Lock`.

This code is loosely adapted from the `py-filelock
<https://github.com/benediktschmitt/py-filelock/blob
/b30bdc4fb998f5f4350257235eb51147f9e81862/filelock.py>`__
module, which is unfortunately not threadsafe.

.. warning::
    Unlike py-filelock, this module does not provide a soft file lock
    to avoid any misinterpretation of the safety of using it.

Example Usage:

.. code-block:: python

    from aiuti.filelock import FileLock

    flock = FileLock('/path/to/file.lock')

    with flock:
        # Use some resource with exclusive thread and process access

"""

__all__ = [
    'BaseFileLock',
    'WindowsFileLock',
    'UnixFileLock',
    'FileLock',
]

import os
import abc
import time
import logging
import threading
import contextlib
from typing import Union, Optional, TypeVar, ClassVar, Any

from .typing import Yields


def _import_optional(name: str) -> Any:
    try:
        return __import__(name)
    except ImportError:
        pass


msvcrt = _import_optional('msvcrt')
fcntl = _import_optional('fcntl')

_logger = logging.getLogger(__name__)

PathLike = Union[str, 'os.PathLike[str]']
FileLockT = TypeVar('FileLockT', bound='BaseFileLock')


class BaseFileLock(abc.ABC):
    """Abstract base class for a file lock."""

    _thread_lock: Union[threading.Lock, threading.RLock]

    def __init__(self,
                 lock_file: PathLike,
                 timeout: float = -1,
                 reentrant: bool = False):
        """
        :param lock_file: Path to the lock file
        :param timeout:
            Optional timeout, in seconds, after which a :class:`Timeout`
            exception will be raised indicating the lock couldn't be
            acquired.
        :param reentrant:
            If True, use an :class:`~threading.RLock` internally to
            allow the same thread to acquire the lock multiple times.
        """
        # The path to the lock file.
        self._lock_file: PathLike = lock_file

        # The file descriptor for the *_lock_file* as it is returned by
        # os.open(). This is only NOT None if the lock is held.
        self._lock_file_fd: Optional[int] = None

        # The default timeout value.
        self.timeout: float = timeout

        # Boolean indicating if the lock is recursive, thread-wise
        self._reentrant: bool = reentrant

        # The internal threading lock to hold while waiting for and
        # while the file lock is held
        if self._reentrant:
            self._thread_lock = threading.RLock()
        else:
            self._thread_lock = threading.Lock()

        # For reentrant locks, the number of levels deep. When this
        # falls to zero, the file lock can be released
        self._lock_counter: int = 0

    @property
    def lock_file(self) -> PathLike:
        """The path to the lock file."""
        return self._lock_file

    @property
    def is_locked(self) -> bool:
        """True, if the object holds the file lock."""
        return self._lock_file_fd is not None

    def acquire(self,
                blocking: bool = True,
                timeout: Optional[float] = None,
                poll_interval: float = 0.05) -> bool:
        """
        Acquire the file lock.

        :param blocking:
            If True, block until the file lock is acquired or the
            timeout is exceeded. If False, return immediately if the
            lock cannot be acquired.
        :param timeout:
            Optional override for :attr:`timeout`, the maximum number of
            seconds to wait for the file lock.
        :param poll_interval:
            Number of seconds between attempts to acquire the file lock.
        :return: True if lock acquired, False otherwise
        """
        # Use the default timeout, if no timeout is provided.
        if timeout is None:
            timeout = self.timeout if blocking else -1
        else:
            blocking = blocking if timeout < 0 else True

        lid = id(self)
        fn = self._lock_file

        if not self._thread_lock.acquire(blocking, timeout):
            _logger.debug('Timeout on acquiring thread lock %s on %s', lid, fn)
            return False

        self._lock_counter += 1  # Keep lock counter synced with RLock

        if self.is_locked:
            return True

        start_time = time.time()

        def _cleanup_thread_lock() -> None:
            self._decrement_lock_counter()
            self._thread_lock.release()

        try:
            while True:
                _logger.debug('Attempting to acquire lock %s on %s', lid, fn)
                self._acquire(block=blocking and timeout < 0)

                if self.is_locked:
                    _logger.info('Lock %s acquired on %s', lid, fn)
                    break
                elif not blocking:
                    _logger.debug('Failed to acquire lock %s on %s', lid, fn)
                    _cleanup_thread_lock()
                    return False
                elif 0 <= timeout < time.time() - start_time:
                    _logger.debug('Timeout on acquiring lock %s on %s', lid, fn)
                    _cleanup_thread_lock()
                    return False
                else:
                    _logger.debug(
                        'Lock %s not acquired on %s, waiting %s seconds ...',
                        lid, fn, poll_interval,
                    )
                    time.sleep(poll_interval)
        except:  # noqa
            _logger.exception("Failed to acquire lock %s on %s", lid, fn)
            _cleanup_thread_lock()
            raise

        return True

    @contextlib.contextmanager
    def acquire_ctx(self,
                    blocking: bool = True,
                    timeout: Optional[float] = None,
                    poll_interval: float = 0.05) -> Yields[None]:
        """
        Context manager to make it more convenient to pass custom args
        to :meth:`acquire`.

        :raises TimeoutError: If :meth:`acquire` returns ``False``.
        """
        if not self.acquire(blocking, timeout, poll_interval):
            raise TimeoutError("Failed to acquire file lock:", self._lock_file)
        try:
            yield
        finally:
            self.release()

    def release(self, force: bool = False) -> None:
        """
        Release the file lock.

        .. note::
            The file lock is only completely released if the lock
            counter is zero when using reentrant locks.

        .. note::
            The lock file itself is not automatically deleted to avoid
            race conditions.

        :param force:
            If true, the lock counter is ignored and the lock is
            released in every case.
        """
        if not self.is_locked:
            return

        self._decrement_lock_counter()

        if self._lock_counter == 0 or force:
            lid = id(self)
            fn = self._lock_file

            _logger.debug('Attempting to release lock %s on %s', lid, fn)
            try:
                self._release()
            except:  # noqa
                _logger.exception("Failed to release lock %s on %s", lid, fn)
            else:
                self._lock_counter = 0
                _logger.info('Lock %s released on %s', lid, fn)

        try:
            self._thread_lock.release()
        except RuntimeError:  # not reentrant and already unlocked
            pass

    # Open mode for the file descriptor
    _FD_OPEN_MODE: ClassVar[int] = os.O_RDWR | os.O_CREAT | os.O_TRUNC

    def _acquire(self, block: bool = True) -> None:
        """
        Attempt to acquire the platform dependent lock and set
        :attr:`_lock_file_fd` to the file descriptor of the lock file.

        :param block:
            If True, attempt to block until the lock is acquired.
        """
        try:
            fd = os.open(self._lock_file, self._FD_OPEN_MODE)
        except OSError:
            return
        try:
            self._lock(fd, block)
        except (IOError, OSError):
            os.close(fd)
        else:
            self._lock_file_fd = fd

    def _release(self) -> None:
        """
        Release the platform dependent lock and clear
        :attr:`_lock_file_fd`
        """
        fd, self._lock_file_fd = self._lock_file_fd, None
        assert isinstance(fd, int)
        try:
            self._unlock(fd)
        finally:
            os.close(fd)

    @abc.abstractmethod
    def _lock(self, fd: int, block: bool = True) -> None:
        """
        Lock the platform dependent lock for the file descriptor.

        :param fd: File descriptor to acquire a file lock for
        :param block:
            If True, attempt to block until the lock is acquired.
        """

    @abc.abstractmethod
    def _unlock(self, fd: int) -> None:
        """
        Unlock the platform dependent lock for the file descriptor.

        .. note::
            This method should not remove the lock file to avoid race
            conditions.
        """

    def _decrement_lock_counter(self) -> None:
        """
        Helper method to decrement the lock counter without letting it
        drop below 0.
        """
        self._lock_counter = max(0, self._lock_counter - 1)

    def __enter__(self: FileLockT) -> FileLockT:
        self.acquire()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.release()

    def __del__(self) -> None:
        self.release(force=True)


class WindowsFileLock(BaseFileLock):
    """
    Windows specific file lock implementation which uses
    :func:`msvcrt.locking` to hard lock the lock file.
    """

    def _lock(self, fd: int, block: bool = True) -> None:
        msvcrt.locking(fd, msvcrt.LK_LOCK if block else msvcrt.LK_NBLCK, 1)

    def _unlock(self, fd: int) -> None:
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)


class UnixFileLock(BaseFileLock):
    """
    Unix specific file lock implementation which uses
    :func:`fcntl.flock` to hard lock the lock file.
    """

    def _lock(self, fd: int, block: bool = True) -> None:
        fcntl.flock(fd,  fcntl.LOCK_EX | (0 if block else fcntl.LOCK_NB))

    def _unlock(self, fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)


class _UnsupportedFileLock(BaseFileLock):

    def _lock(self, fd: int, block: bool = True) -> None:
        raise RuntimeError("Filelock isn't supported by your platform!")

    def _unlock(self, fd: int) -> None:
        raise RuntimeError("Filelock isn't supported by your platform!")


#: Alias for the lock, which should be used for the current platform.
#: On Windows, this is an alias for :class:`WindowsFileLock` and on Unix
#: for :class:`UnixFileLock`.
FileLock = _UnsupportedFileLock

if msvcrt:
    FileLock = WindowsFileLock  # type: ignore
elif fcntl:
    FileLock = UnixFileLock  # type: ignore
else:
    logging.warning("No platform specific lock available!")

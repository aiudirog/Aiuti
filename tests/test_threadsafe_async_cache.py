import asyncio as aio
from concurrent.futures import ThreadPoolExecutor

import pytest

from aiuti.asyncio import threadsafe_async_cache


def test_canceling() -> None:
    # This hangs on v0.7 due to lock.release() never running when
    # run_in_executor for lock.acquire() is cancelled

    @threadsafe_async_cache
    async def func() -> None:
        await aio.sleep(1)

    async def impatient():
        with pytest.raises(aio.TimeoutError):
            await aio.wait_for(func(), 0.0001)

    async def main() -> None:
        await aio.gather(*(impatient() for _ in range(10)))

    with ThreadPoolExecutor(4) as pool:
        for t in [pool.submit(aio.run, main()) for _ in range(10)]:
            t.result()

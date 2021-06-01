from time import sleep
import asyncio as aio

import pytest


@pytest.fixture(autouse=True)
def new_loop() -> aio.AbstractEventLoop:
    """
    Ensure that every test case gets a new loop to avoid any bleeding
    between failed test cases.
    """
    loop = aio.new_event_loop()
    aio.set_event_loop(loop)
    try:
        yield loop
    finally:
        for _ in range(5):
            if not loop.is_closed():
                try:
                    loop.close()
                except RuntimeError:  # Still running
                    sleep(0.1)
                    continue
                else:
                    break
        else:
            raise RuntimeError("Failed to shutdown event loop")

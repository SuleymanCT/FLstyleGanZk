import asyncio

from circuits.ezkl_utils import run_async


def test_run_async_with_sync_function():
    def add(a, b):
        return a + b

    assert run_async(add, 2, 3) == 5


def test_run_async_with_coroutine_function():
    async def add_async(a, b):
        await asyncio.sleep(0)
        return a + b

    assert run_async(add_async, 2, 3) == 5


def test_run_async_passes_kwargs():
    def f(a, b=0):
        return a - b

    assert run_async(f, 10, b=4) == 6


def test_run_async_propagates_exceptions():
    def boom():
        raise ValueError("kaboom")

    try:
        run_async(boom)
    except ValueError as e:
        assert "kaboom" in str(e)
    else:
        raise AssertionError("ValueError bekleniyordu")

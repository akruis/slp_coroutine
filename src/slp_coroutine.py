# Use plain old callables as coroutines using Stackless Python
# Copyright (c) 2021  Anselm Kruis
#
# This library is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License as published by the Free
# Software Foundation; either version 2.1 of the License, or (at your option)
# any later version.
#
# This library is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this library; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Suite 500, Boston, MA  02110-1335  USA.


'''
A prove of concept for adapting plain old callables to asyncio using Stackless Python

Limitations of this demo code:

   - Requires Stackless Python >= 3.7.6

'''

import sys
import stackless
import collections.abc
import functools
import types
import contextvars

__all__ = ('new_generator_coroutine', 'as_coroutinefunction', 'new_coroutine', 'await_coroutine', 'generator', 'async_generator')


class TaskletStopIteration(StopIteration):
    pass

def _run(args):
    # The args.pop() trick removes any arguments from this stack frame. This way any non-pickleable
    # object does not hurt.
    raise TaskletStopIteration(args.pop()(*args.pop(), **args.pop()))

_is_as_coroutine_tasklet = contextvars.ContextVar('is_as_coroutine_tasklet', default=False)

def _run_as_coroutine_tasklet():
    token = _is_as_coroutine_tasklet.set(True)
    try:
        return stackless.run()
    finally:
        _is_as_coroutine_tasklet.reset(token)

@types.coroutine
def new_generator_coroutine(callable_, *args, **kwargs):
    tasklet = stackless.tasklet(_run)([kwargs, args, callable_])
    del callable_
    del args
    del kwargs

    wait_for = None
    value = None
    exception = None

    while True:
        if wait_for is not None:
            assert tasklet.paused
            try:
                if exception is None:
                    value = tasklet.context_run(wait_for.send, value)
                else:
                    try:
                        value = tasklet.context_run(wait_for.throw, *exception)
                    finally:
                        exception = None
            except StopIteration as ex:
                wait_for = None
                value = ex.value
            except Exception:
                # an error
                wait_for = None
                exception = sys.exc_info()
                value = None

        if wait_for is None:
            if not tasklet.alive:
                if exception is not None:
                    try:
                        raise exception[1].with_traceback(exception[2])
                    finally:
                        exception = None
                return value

            if tasklet.paused:
                tasklet.insert()
            assert tasklet.scheduled
            tasklet.tempval = value
            value = None
            try:
                if exception is not None:
                    try:
                        tasklet.throw(*exception, pending=True)
                    finally:
                        exception = None
                
                # Usually tasklet shares the context with the current tasklet, but this is not the case,
                # if the current tasklet has no context, or if tasklet changes the context.
                # Therefore we run the stackless scheduler in the context of tasklet
                tasklet.context_run(_run_as_coroutine_tasklet)
            except TaskletStopIteration as ex:
                return ex.value
            except StopIteration as ex:
                # special case: a generator must not raise StopIteration.
                # Therefore we mask it as a StopAsyncIteration. This is consistent
                # with asynchronous generators   
                raise StopAsyncIteration() from ex
            value = tasklet.tempval
            tasklet.tempval = None

            if value is tasklet:
                # tasklet is the default return value of stackless.schedule and stackless.schedule_remove
                value = None

            # test, if tasklet called await_coroutine(...) 
            if isinstance(value, tuple) and len(value) == 2 and value[0] == "await":
                assert tasklet.paused
                wait_for = value[1]
                value = None
                continue

        try:
            value = yield value
        except GeneratorExit:
            # this exception signals a call of generator.close()
            if wait_for is not None:
                try:
                    wait_for.close()
                finally:
                    wait_for = None
            if tasklet.alive:
                tasklet.kill()
            raise
        except Exception:
            exception = sys.exc_info()
            value = None

def generator(asyncgen):
    # See Python language reference 8.8.2. The async for statement
    if not _is_as_coroutine_tasklet.get():
        raise RuntimeError("Can't call generator() from tasklet " + repr(stackless.current))
    asyncgen = type(asyncgen).__aiter__(asyncgen)
    cls = type(asyncgen)
    # first method is always __anext__
    method = cls.__anext__
    value = ()
    while True:
        try:
            value = stackless.schedule_remove(("await", method(asyncgen, *value)))
        except StopAsyncIteration:
            return
        try:
            value = ((yield value),)
            method = cls.asend
        except GeneratorExit:
            try:
                stackless.schedule_remove(("await", cls.aclose(asyncgen)))
            except StopAsyncIteration:
                pass
            raise
        except Exception:
            value = sys.exc_info()
            method = cls.athrow


async def async_generator(generator):
    value = None
    exception = None
    while True:
        try:
            if exception is None:
                try:
                    m = type(generator).send
                except AttributeError:
                    if value is not None:
                        raise
                    value = await new_generator_coroutine(type(generator).__next__, generator)
                else:
                    value = await new_generator_coroutine(m, generator, value)
            else:
                try:
                    value = await new_generator_coroutine(type(generator).throw, generator, *exception)
                finally:
                    exception = None
        except StopAsyncIteration as ex:
            if not isinstance(ex.__cause__, StopIteration):
                raise  # Not an wrapped
            if ex.__cause__.value is not None:
                raise RuntimeError("generator returned a value other than None")
            return
        try:
            value = yield value
        except GeneratorExit:
            value = None
            await new_generator_coroutine(type(generator).close, generator)
            raise
        except Exception:
            value = None
            exception = sys.exc_info()


class contextmanager:
    def __init__(self, async_contextmanager):
        self.async_contextmanager = async_contextmanager
        
    def __enter__(self):
        return await_coroutine(type(self.async_contextmanager).__aenter__(self.async_contextmanager))
    
    def __exit__(self, exc_type, exc_value, traceback):
        return await_coroutine(type(self.async_contextmanager).__aexit__(self.async_contextmanager, exc_type, exc_value, traceback))


class asynccontextmanager:
    def __init__(self, contextmanager):
        self.contextmanager = contextmanager
        
    async def __aenter__(self):
        return await new_generator_coroutine(type(self.contextmanager).__enter__(self.contextmanager))
    
    async def __aexit__(self, exc_type, exc_value, traceback):
        return await new_generator_coroutine(type(self.contextmanager).__exit__(self.contextmanager, exc_type, exc_value, traceback))



def as_coroutinefunction(callable_):
    """A decorator for a plain old callable
    """
    @functools.wraps(callable_)
    async def coro(*args, **kwargs):
        return await new_generator_coroutine(callable_, *args, **kwargs)
    return coro


def new_coroutine(callable_, *args, **kwargs):
    return as_coroutinefunction(callable_)(*args, **kwargs)


def await_coroutine(coroutine):
    """await the given coroutine"""
    if not _is_as_coroutine_tasklet.get():
        raise RuntimeError("Can't call await_coroutine from tasklet " + repr(stackless.current))
    if not isinstance(coroutine, (collections.abc.Coroutine, collections.abc.Generator)):
        raise TypeError("argument is neither a coroutine nor a generator")
    return stackless.schedule_remove(("await", coroutine))



if __name__ == '__main__':
    # demo code below
    import asyncio

    async def coroutine_function(arg):
        print("coroutine_function: start, sleeping ...")
        await asyncio.sleep(1)
        print("coroutine_function: end")
        return arg + 1
    
    def classic_function(arg):
        print("classic_function: start")
        res = await_coroutine(coroutine_function(arg + 1))
        print("classic_function: end")
        return res + 1
    
    def demo():
        print("start")
        res = asyncio.run(new_coroutine(classic_function, 0), debug=False)
        print("end, result: ", res)

    sys.exit(demo())

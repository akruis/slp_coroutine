# Use normal callables as coroutines using Stackless Python
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
Adapt normal functions, methods and other callables to asyncio using Stackless Python

Requires Stackless Python >= 3.7.6

This module provides adapters and decorators to use "normal" Python code
to write tasks of an asyncio-application. This way many libraries, which do not
support asyncio can be used with this framework.

In Python the difference between a *normal function* and a *coroutine function* is, that the coroutine
function needs a controller, which executes the coroutine step by step, whereas the normal function executes
in a single non-interruptible pass. If the coroutine needs to wait for an event (i.e. data input),
its controller can run other code, until the event happens. A normal function usually has no choice but to block.

With *Stackless Python* it is possible to execute multiple functions in parallel on a single thread.
Each execution context is called a *tasklet*. If a function has to wait for an event, it can - instead of blocking -
switch the execution to another tasklet. This switching capability makes it possible to invoke a normal callable
from a coroutine and to await coroutines from within the normal callable. This technique can be extended
to (asynchronous) generators and decorators.
'''

import sys
import collections.abc
import functools
import types
import threading
import contextlib
import stackless


__all__ = ('new_generator_coroutine', 'as_coroutinefunction', 'new_coroutine', 'await_coroutine', 'generator', 'async_generator')

_AWAIT = object()  # sentinel

class _TaskletStopIteration(StopIteration):
    """Signal the end of a tasklet and encapsulate the return value of the tasklet"""
    pass

def _run(args):
    # utility function for new_generator_coroutine()
    # The args.pop() trick removes any arguments from this stack frame. This way any non-pickleable
    # object does not hurt.
    raise _TaskletStopIteration(args.pop()(*args.pop(), **args.pop()))

# This thread local variable counts, how many times the Stackless scheduler (stackless.run()) has been
# recursively entered by new_generator_coroutine()
class _ThreadLocalRunCount(threading.local):
    runcount = 0
_new_generator_coroutine = _ThreadLocalRunCount()

@types.coroutine
def new_generator_coroutine(callable_, *args, **kwargs):
    """Run any Python callable as generator-based coroutine.

    The coroutine-function :func:`new_generator_coroutine` returns a new
    coroutine, that executes *callable_* in a new tasklet.
    *callable_* may await other coroutines or switch tasklets. Details:

    * If the tasklet yields (i.e. calls :func:`stackless.schedule`)
      the generator-coroutine returned by :func:`new_generator_coroutine` yields `tasklet.tempval`.
    * If the tasklet awaits another coroutine (calls :func:`await_coroutine`), the generator-coroutine
      returned by :func:`new_generator_coroutine` controls the awaited coroutine.
    * If you send a value or an exception to the generator-coroutine returned by :func:`new_generator_coroutine`,
      it passes the value or exception to the tasklet or, if the tasklet is awaiting a coroutine,
      to the currently awaited coroutine.
    * The returned coroutine returns the return value of ``callable(*args, **kwargs)``.
    * The returned coroutine is generator based. See :func:`types.coroutine`.

    :param Callable callable_: the callable to be executed
    :param args: positional arguments for *callable_*
    :param kwargs: keyword arguments for *callable_*
    :returns: a generator-based coroutine that executes ``callable(*args, **kwargs)``
    """
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
                rc = _new_generator_coroutine.runcount
                _new_generator_coroutine.runcount = 0
                try:
                    if exception is None:
                        value = tasklet.context_run(wait_for.send, value)
                    else:
                        try:
                            value = tasklet.context_run(wait_for.throw, *exception)
                        finally:
                            exception = None
                finally:
                    _new_generator_coroutine.runcount = rc
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
                _new_generator_coroutine.runcount += 1
                try:
                    stackless.run()
                finally:
                    _new_generator_coroutine.runcount -= 1
            except _TaskletStopIteration as ex:
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
            if isinstance(value, tuple) and len(value) == 2 and value[0] is _AWAIT:
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
    """Create a generator-iterator, that iterates over *asyncgen*

    This generator-function creates a generator-iterator, that iterates
    over the given asynchronous iterable.

    The returned generator-iterator must be called from code, that is executed
    by :func:`new_generator_coroutine`.

    :param asyncgen: an asynchronous iterable object
    :returns: a generator iterator
    :raises RuntimeError: the generator raises RuntimeError, if called from outside of
       :func:`new_generator_coroutine`.
    """
    # See Python language reference 8.8.2. The async for statement
    if not _new_generator_coroutine.runcount:
        raise RuntimeError("Can't call generator() from tasklet " + repr(stackless.current))
    asyncgen = type(asyncgen).__aiter__(asyncgen)
    cls = type(asyncgen)
    # first method is always __anext__
    method = cls.__anext__
    value = ()
    while True:
        try:
            value = stackless.schedule_remove((_AWAIT, method(asyncgen, *value)))
        except StopAsyncIteration:
            return
        try:
            value = ((yield value),)
            method = cls.asend
        except GeneratorExit:
            try:
                stackless.schedule_remove((_AWAIT, cls.aclose(asyncgen)))
            except StopAsyncIteration:
                pass
            raise
        except Exception:
            value = sys.exc_info()
            method = cls.athrow


async def async_generator(generator):
    """Create an asynchronous generator iterator, that iterates over *generator*

    This generator-function creates a asynchronous generator iterator, that iterates
    over the given generator or iterator.

    :param generator: a generator or iterator object
    :returns: an asynchronous generator iterator
    """
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
    """A context manager, that delegates to an asynchronous context manager.

    This *with* statement context manager delegates to an asynchronous context manager
    to actually manage the context.
    """
    def __init__(self, async_contextmanager):
        """
        :param async_contextmanager: the asynchronous context manager to delegate to
        """
        self.async_contextmanager = async_contextmanager

    def __enter__(self):
        return await_coroutine(type(self.async_contextmanager).__aenter__(self.async_contextmanager))

    def __exit__(self, exc_type, exc_value, traceback):
        return await_coroutine(type(self.async_contextmanager).__aexit__(self.async_contextmanager, exc_type, exc_value, traceback))

contextlib.AbstractContextManager.register(contextmanager)  # @UndefinedVariable


class asynccontextmanager:
    """An asynchronous context manager, that delegates to a *with* statement context manager.

    This asynchronous context manager delegates to a synchronous *with* statement context manager
    to actually manage the context.
    """

    def __init__(self, contextmanager):
        """
        :param contextmanager: the context manager to delegate to
        """
        self.contextmanager = contextmanager

    async def __aenter__(self):
        return await new_generator_coroutine(type(self.contextmanager).__enter__, self.contextmanager)

    async def __aexit__(self, exc_type, exc_value, traceback):
        return await new_generator_coroutine(type(self.contextmanager).__exit__, self.contextmanager, exc_type, exc_value, traceback)

contextlib.AbstractAsyncContextManager.register(asynccontextmanager)  # @UndefinedVariable


def as_coroutinefunction(callable_):
    """Create a *coroutine function* from a normal callable.

    This function is a *decorator* that can be used to define a
    *coroutine function* from any normal function, method or other
    callable.

    :param Callable callable_: the callable to be decorated
    :returns: a coroutine function
    """
    @functools.wraps(callable_)
    async def coro(*args, **kwargs):
        return await new_generator_coroutine(callable_, *args, **kwargs)
    return coro


def new_coroutine(callable_, *args, **kwargs):
    """Run any Python callable as coroutine.

    Same as :func:`new_generator_coroutine`, but return a coroutine
    created by a *async def* coroutine function.
    """
    return as_coroutinefunction(callable_)(*args, **kwargs)


def await_coroutine(coroutine):
    """await a coroutine

    A normal function (or method or other callable) may use this function
    to await an awaitable object, if the function has been directly or
    indirectly called by :func:`new_generator_coroutine`.

    :param coroutine: the coroutine to be awaited
    :type coroutine: :class:`~collections.abc.Coroutine` or :class:`~collections.abc.Generator`
    :returns: the value returned by *coroutine*
    :raises RuntimeError: if called from outside of :func:`new_generator_coroutine`.
    """
    if not _new_generator_coroutine.runcount:
        raise RuntimeError("Can't call await_coroutine from tasklet " + repr(stackless.current))
    if not isinstance(coroutine, (collections.abc.Coroutine, collections.abc.Generator)):
        raise TypeError("argument is neither a coroutine nor a generator")
    return stackless.schedule_remove((_AWAIT, coroutine))



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

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
import slp_coroutine

'''
Tests for spl_asyncio

'''

import unittest
import types
import inspect
import stackless
import contextvars
import contextlib

from slp_coroutine import *
from slp_coroutine import _new_generator_coroutine

test_context_var = contextvars.ContextVar('my_context_var')

# copied from test.test_coroutines
def run_async(coro):
    assert coro.__class__ in {types.GeneratorType, types.CoroutineType}

    buffer = []
    result = None
    while True:
        try:
            buffer.append(coro.send(None))
        except StopIteration as ex:
            result = ex.args[0] if ex.args else None
            break
    return buffer, result


class Test(unittest.TestCase):

    def setUp(self):
        super().setUp()

    def tearDown(self):
        super().tearDown()

    def test_new_coroutine_1(self):
        self.assertTrue(inspect.isfunction(new_coroutine))

    def test_new_coroutine_2(self):
        def foo():
            raise AssertionError('must not run')
        coro = new_coroutine(foo)
        self.assertTrue(inspect.iscoroutine(coro))
        self.assertIsInstance(coro, types.CoroutineType)
        coro.close()

    def test_new_coroutine_3(self):
        def foo(arg, arg2):
            return (arg, arg2)
        coro = new_coroutine(foo, 10, arg2='blubber')
        
        result = run_async(coro)
        self.assertEqual(result, ([], (10, 'blubber')))

    def test_new_coroutine_4(self):
        def foo(arg):
            raise RuntimeError('expected test exception')
        coro = new_coroutine(foo, 10)
        
        self.assertRaisesRegex(RuntimeError, 'expected test exception', coro.send, None)

    def test_new_coroutine_5(self):
        def foo(arg):
            stackless.schedule_remove()
            return arg
        coro = new_coroutine(foo, 5)
        
        result = run_async(coro)
        self.assertEqual(result, ([None], 5))

    def test_new_coroutine_6(self):
        def foo(arg):
            stackless.schedule_remove(None)
            return arg
        coro = new_coroutine(foo, 6)
        
        result = run_async(coro)
        self.assertEqual(result, ([None], 6))

    def test_new_coroutine_7(self):
        def foo(arg):
            res = stackless.schedule_remove(arg)
            return res
        coro = new_coroutine(foo, 7)
        
        self.assertEqual(coro.send(None), 7)
        with self.assertRaises(StopIteration) as cm:
            coro.send('bla')
        self.assertEqual(cm.exception.value, 'bla')

    def test_new_coroutine_8(self):
        got_TaskletExit = False
        def foo(arg):
            nonlocal got_TaskletExit
            try:
                res = stackless.schedule_remove(arg)
            except TaskletExit:
                got_TaskletExit = True
                raise
            return res
        coro = new_coroutine(foo, 8)
        
        self.assertEqual(coro.send(None), 8)
        self.assertFalse(got_TaskletExit)
        self.assertIsNone(coro.close())
        self.assertTrue(got_TaskletExit)

    def test_new_coroutine_9(self):
        def foo(arg):
            """foo docstring"""
            return arg
        coro = new_coroutine(foo, 9)
        self.assertEqual(coro.__name__, foo.__name__)
        self.assertEqual(coro.__qualname__, foo.__qualname__)
        coro.close()
        
    def test_as_coroutinefunction_1(self):
        self.assertTrue(inspect.isfunction(as_coroutinefunction))

    def test_as_coroutinefunction_2(self):
        @as_coroutinefunction
        def foo(arg: int) -> None:
            '''This is the documentation'''
            raise AssertionError('must not run')

        self.assertTrue(inspect.iscoroutinefunction(foo))
        self.assertEqual(foo.__annotations__, {'arg': int, 'return': None})
        self.assertEqual(foo.__doc__, 'This is the documentation')
        coro = foo()
        self.assertIsInstance(coro, types.CoroutineType)
        self.assertEqual(coro.__name__, foo.__name__)
        self.assertEqual(coro.__qualname__, foo.__qualname__)
        coro.close()

    def test_as_coroutinefunction_3(self):
        @as_coroutinefunction
        def foo(arg, arg2):
            return (arg, arg2)
        
        result = run_async(foo(10, arg2='blubber'))
        self.assertEqual(result, ([], (10, 'blubber')))

    def test_await_coroutine_1(self):
        @types.coroutine
        def bar(arg, arg2):
            res = yield (arg, arg2)
            return res
        def foo(arg, arg2):
            return await_coroutine(bar(arg, arg2=arg2))

        arg = (11,)
        coro = new_coroutine(foo, arg, 'blubber')
        result = run_async(coro)
        self.assertEqual(result, ([(arg, 'blubber')], None))
        self.assertIs(result[0][0][0], arg)

    def test_await_coroutine_2(self):
        @types.coroutine
        def bar(arg):
            res = yield (arg)
            return res
        def foo(arg):
            return await_coroutine(bar(arg))

        arg = (12,)
        arg2 = NotImplemented
        coro = new_coroutine(foo, arg)
        result = coro.send(None)
        self.assertIs(result, arg)
        with self.assertRaises(StopIteration) as cm:
            coro.send(arg2)
        self.assertEqual(cm.exception.value, arg2)

    def test_await_coroutine_3(self):
        @types.coroutine
        def bar(arg):
            try:
                res = yield (arg)
            except RuntimeError as e:
                res = ('got exception', e)
            return res
        def foo(arg):
            return await_coroutine(bar(arg))

        arg = (12,)
        coro = new_coroutine(foo, arg)
        result = coro.send(None)
        self.assertIs(result, arg)
        with self.assertRaises(StopIteration) as cm:
            coro.throw(RuntimeError('expected exception'))
        self.assertEqual(cm.exception.value[0], 'got exception')
        with self.assertRaisesRegex(RuntimeError, 'expected exception'):
            raise cm.exception.value[1]

    def test_await_coroutine_4(self):
        @types.coroutine
        def bar(arg):
            raise RuntimeError('immediate exception')
            yield  # make this function a generator function

        def foo(arg):
            return await_coroutine(bar(arg))

        arg = (12,)
        coro = new_coroutine(foo, arg)
        self.assertRaisesRegex(RuntimeError, 'immediate exception', coro.send, None)

    def test_generator_1(self):
        async def bar(*args):
            for v in args:
                yield v

        def foo(*args):
            res = []
            for v in generator(bar(*args)):
                res.append(v)
            return tuple(res)
        
        arg = (21, 22, 23)
        coro = new_coroutine(foo, *arg)
        result = run_async(coro)
        self.assertEqual(result, ([], arg))

    def test_generator_2(self):
        got_GeneratorExit = False

        async def bar(*args):
            nonlocal got_GeneratorExit
            for v in args:
                try:
                    yield v
                except GeneratorExit:
                    got_GeneratorExit = True
                    raise

        def foo(*args):
            g = generator(bar(*args))
            res = next(g)
            g.close()
            g.close()
            return res
        
        arg = (24, 25)
        coro = new_coroutine(foo, *arg)
        result = run_async(coro)
        self.assertEqual(result, ([], 24))
        self.assertTrue(got_GeneratorExit)

    def test_generator_3(self):
        async def bar(*args):
            raise RuntimeError('this is a test')
            yield

        def foo(*args):
            return tuple(generator(bar(*args)))
        
        arg = (24, 25)
        coro = new_coroutine(foo, *arg)
        self.assertRaisesRegex(RuntimeError, 'this is a test', run_async, coro)

    def test_generator_4(self):
        result = []

        async def bar():
            v = 0
            while True:
                try:
                    v = yield v
                    result.append(v)
                except RuntimeError as e:
                    result.append(e)
                    raise

        def foo():
            g = generator(bar())
            result.append(next(g))
            result.append(g.send(26))
            g.throw(RuntimeError('test4'))  # raises
        
        coro = new_coroutine(foo)
        with self.assertRaisesRegex(RuntimeError, 'test4') as cm:
            run_async(coro)
        self.assertEqual(result, [0, 26, 26, cm.exception])

    def test_async_generator_1(self):
        def gen():
            yield 1
            yield 2
            
        async def coro():
            res = []
            async for i in async_generator(gen()):
                res.append(i)
            return res 

        result = run_async(coro())
        self.assertEqual(result, ([], [1, 2]))

    def test_async_generator_2(self):
        def gen():
            yield 1
            yield 2
            
        async def coro():
            res = []
            ag = async_generator(gen())
            async for i in ag:
                res.append(i)
                break
            await ag.aclose()
            return res 

        result = run_async(coro())
        self.assertEqual(result, ([], [1]))

    def test_async_generator_3(self):
        def gen():
            raise RuntimeError('this is a test')
            yield

        async def coro():
            res = []
            async for i in async_generator(gen()):
                res.append(i)
            return res 
        
        self.assertRaisesRegex(RuntimeError, 'this is a test', run_async, coro())

    def test_async_generator_4(self):
        result = []

        def gen():
            v = 0
            while True:
                try:
                    v = yield v
                    result.append(v)
                except RuntimeError as e:
                    result.append(e)
                    raise

        async def coro():
            ag = async_generator(gen())
            result.append(await ag.__anext__())
            result.append(await ag.asend(36))
            await ag.athrow(RuntimeError('agtest4'))  # raises
        
        with self.assertRaisesRegex(RuntimeError, 'agtest4') as cm:
            run_async(coro())
        self.assertEqual(result, [0, 36, 36, cm.exception])

    def test_contextmanager_1(self):
        @contextlib.asynccontextmanager
        async def ascmgr():
            yield 1

        context_manager = slp_coroutine.contextmanager(ascmgr())
        self.assertIsInstance(context_manager, contextlib.AbstractContextManager)
        self.assertTrue(callable(context_manager.__enter__))
        self.assertTrue(callable(context_manager.__exit__))

    def test_contextmanager_2(self):

        steps = []
        @contextlib.asynccontextmanager
        async def ascmgr():
            steps.append(1)
            try:
                yield 2
            finally:
                steps.append(3)

        def func():
            context_manager = slp_coroutine.contextmanager(ascmgr())
            with context_manager as value:
                self.assertListEqual(steps, [1])
                steps.append((value,4))

        result = run_async(new_coroutine(func))
        self.assertListEqual(steps, [1, (2,4), 3])
        self.assertEqual(result, ([], None))

    def test_asynccontextmanager_1(self):
        @contextlib.contextmanager
        def cmgr():
            yield 1

        context_manager = slp_coroutine.asynccontextmanager(cmgr())
        self.assertIsInstance(context_manager, contextlib.AbstractAsyncContextManager)
        self.assertTrue(callable(context_manager.__aenter__))
        self.assertTrue(callable(context_manager.__aexit__))

    def test_asynccontextmanager_2(self):
        steps = []
        @contextlib.contextmanager
        def cmgr():
            steps.append(1)
            try:
                yield 2
            finally:
                steps.append(3)

        async def func():
            context_manager = slp_coroutine.asynccontextmanager(cmgr())
            async with context_manager as value:
                self.assertListEqual(steps, [1])
                steps.append((value,4))

        result = run_async(func())
        self.assertListEqual(steps, [1, (2,4), 3])
        self.assertEqual(result, ([], None))

    def test_context_1(self):

        @types.coroutine
        def bar():
            return _new_generator_coroutine.runcount, test_context_var.get()
            yield

        def foo():
            token = test_context_var.set('test_context_1_foo')
            try:
                return await_coroutine(bar()), _new_generator_coroutine.runcount
            finally:
                test_context_var.reset(token)

        self.assertFalse(_new_generator_coroutine.runcount)
        sentinel = object()
        self.assertIs(test_context_var.get(sentinel), sentinel)
        coro = new_coroutine(foo)
        result = run_async(coro)
        self.assertEqual(result, ([], ((0, 'test_context_1_foo'), 1)))


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()

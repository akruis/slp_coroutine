'''
A demo for pickling asyncio.Task and for implementing a task using classic functions

Limitations of this demo code:

   - Requires Stackless Python 3.7, because this Python implementation can pickle coroutine objects.
   - Uses the pure python task implementation asyncio.tasks._PyTask 

Copyright (C) 2021  Anselm Kruis

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

import sys
import asyncio
import pickle
import pickletools  # @UnusedImport
import copyreg
import contextvars
import io
import cvpickle  # available at https://github.com/akruis/cvpickle or PyPi
from slp_coroutine import await_coroutine, new_coroutine

# Extend pickling support a bit. Not perfect, just enough for the demo
#
# pickling for objects of type asyncio.tasks._PyTask
# In order to support pickling of the C implementation of asyncio.Task, more effort is required.
# Probably we need an extension module
def _reduce_asyncio_tasks__PyTask(task):
    # attributes of _PyTask
    # _loop                 # from _PyFuture
    # _source_traceback     # init with  format_helpers.extract_stack(sys._getframe(1))  # _PyFuture
    # _log_destroy_pending  # not always present
    # _must_cancel
    # _fut_waiter
    # _coro
    # _context
    return (object.__new__, (asyncio.tasks._PyTask,), task.__dict__)
copyreg.pickle(asyncio.tasks._PyTask, _reduce_asyncio_tasks__PyTask)

class PicklerWithExternalObjects(pickle._Pickler):
    def __init__(self, file, protocol=None, *, fix_imports=True, external_map=None):
        super().__init__(file, protocol, fix_imports=fix_imports)
        self.inverted_external_map = { id(v): k for k,v in external_map.items() }

    def persistent_id(self, obj):
        if self.inverted_external_map:
            return self.inverted_external_map.get(id(obj))
        return None

    @classmethod
    def dumps(cls, obj, protocol=None, *, fix_imports=True, external_map=None):
        f = io.BytesIO()
        cls(f, protocol, fix_imports=fix_imports, external_map=external_map).dump(obj)
        res = f.getvalue()
        assert isinstance(res, (bytes, bytearray))
        return res

class UnPicklerWithExternalObjects(pickle.Unpickler):
    def __init__(self, file, *, fix_imports=True, encoding="ASCII", errors="strict", external_map=None):
        super().__init__(file, fix_imports=fix_imports, encoding=encoding, errors=errors)
        self.external_map = external_map

    def persistent_load(self, pid):
        # This method is invoked whenever a persistent ID is encountered.
        # Here, pid is the tuple returned by DBPickler.
        if self.external_map is None:
            raise pickle.UnpicklingError("external_map not set")
        try:
            return self.external_map[pid]
        except KeyError:
            # Always raises an error if you cannot return the correct object.
            # Otherwise, the unpickler will think None is the object referenced
            # by the persistent ID.
            raise pickle.UnpicklingError("unsupported persistent object")

    @classmethod
    def loads(cls, s, *, fix_imports=True, encoding="ASCII", errors="strict", external_map=None):
        if isinstance(s, str):
            raise TypeError("Can't load pickle from unicode string")
        file = io.BytesIO(s)
        return cls(file, fix_imports=fix_imports, encoding=encoding, errors=errors, external_map=external_map).load()

# A portable extension of asyncio.Loop. Adds the method "create_py_task(self, coro)".
# It is required, because asyncio.Task (the C-implementation) can't be pickled
class ExtendedEventLoopPolicy(asyncio.DefaultEventLoopPolicy):
    def __init__(self):
        super().__init__()
        loop = super().new_event_loop()
        loop.close()
        class ExtendedEventLoop(type(loop)):
            def create_py_task(self, coro):
                """Same as loop.create_task() but returns a _PyTask object"""
                self._check_closed()
                task = asyncio.tasks._PyTask(coro, loop=self)
                if task._source_traceback:
                    del task._source_traceback[-1]
                return task

        self.extended_event_loop_factory = ExtendedEventLoop

    def new_event_loop(self):
        return self.extended_event_loop_factory()


# Task Controller 

class _TaskControllerDoneFutureSurrogate(object):
    '''A surrogate of a asyncio.Future, that is done.
    
    Instances of this class are used to inject the result of an
    API call (i.e. TaskController.serialize()) into the unpickled task.
    '''
    def __init__(self, result, exception=None):
        self._result = result
        self._exception = exception

    def done(self):
        return True

    def result(self):
        if self._exception is not None:
            raise self._exception
        return self._result


class TaskController(object):
    """Controls a single task

    The controller is responsible for
    - starting
    - serializing
    - de-serializing

    The controller works asynchronously so it can be used in an asyncio program.
    The TaskController is stored in the context variable `current_task_controller`.
    """
    current_task_controller = contextvars.ContextVar('current_task_controller')
        
    # default values for instance vars
    _controller_task = None  # not really required, just used in assertions
    _task = None
    _ignore_cancel = False
    _result = None
    _exception = None

    @classmethod
    async def serialize(cls, do_stop):
        """Serialize a task.
        
        This class method must be awaited from the task to be serialized.
        """
        self = cls.current_task_controller.get()  # get the TaskController from the context
        loop = asyncio.get_running_loop()
        # switch task to make the current task inactive
        return await loop.create_py_task(self._serialize_impl(do_stop, asyncio.current_task()))

    async def _serialize_impl(self, do_stop, task):
        assert task is self._task
        assert self._controller_task._fut_waiter is task
        assert task._fut_waiter is asyncio.current_task()
        loop = asyncio.get_running_loop()
        callbacks = task._callbacks[:]
        try:
            for c in callbacks:
                task.remove_done_callback(c[0])
            external_map={'EventLoop': loop,
                          'CurrentFuture': task._fut_waiter,  # see assert
                          'TaskController': self,
                          'GLOBAL asyncio.tasks._PyTask': asyncio.tasks._PyTask }
            p = PicklerWithExternalObjects.dumps(task, external_map=external_map)
        finally:
            for c in callbacks:
                task.add_done_callback(c[0], context=c[1])
        if do_stop:
            self._result = p
            self._ignore_cancel = True
            task.cancel()
        return p

    async def start_task(self, corofunc, *args, **kwargs):
        """Start a task.
        """
        loop = asyncio.get_running_loop()
        token = self.current_task_controller.set(self)
        self._controller_task = asyncio.current_task()
        try:
            self._task = loop.create_py_task(corofunc(*args, **kwargs))
        finally:
            self.current_task_controller.reset(token)
        return await self._run_task()
            
    async def continue_task(self, pickle_bytes, send=None, throw=None):
        """Continue a pickled task
        """
        assert self._task is None
        loop = asyncio.get_running_loop()
        self._controller_task = asyncio.current_task()

        token = self.current_task_controller.set(self)
        try:
            done_fut = _TaskControllerDoneFutureSurrogate(send, throw)
            external_map = {'EventLoop': loop,
                            'CurrentFuture': done_fut,
                            'TaskController': self,
                            'GLOBAL asyncio.tasks._PyTask': asyncio.tasks._PyTask}
            task = UnPicklerWithExternalObjects.loads(pickle_bytes, external_map=external_map)
            assert task._fut_waiter is done_fut
            task._fut_waiter = None             #  call_soon sets this value to self._controller_task
            asyncio.tasks._register_task(task)  # unpickling does not register it
            loop.call_soon(task._Task__step, context=task._context)
            # now task is a valid task, scheduled and ready to run 
            self._task = task
        finally:
            self.current_task_controller.reset(token)
        return await self._run_task()

    async def _run_task(self):
        '''Backend for `start_task` and `continue_task`'''
        try:
            return await self._task
        except asyncio.CancelledError:
            if self._ignore_cancel:
                if self._exception is not None:
                    raise self._exception
                return self._result
            raise  # rethrow

cvpickle.register_contextvar(TaskController.current_task_controller, __name__, 'TaskController.current_task_controller')


#  Task Definition

async def task_main_function(do_stop):
    print("task_main_function: start")
    res = await task_sub_function(do_stop)
    print("task_main_function: end")
    return res

async def task_sub_function(do_stop):
    print("task_sub_function: start")
    res = await TaskController.serialize(do_stop)
    print("task_sub_function: end")
    return res

def classic_task_main_function(do_stop):
    print("classic_task_main_function: start")
    res = classic_task_sub_function(do_stop)
    print("classic_task_main_function: end")
    return res

def classic_task_sub_function(do_stop):
    print("classic_task_sub_function: start")
    res = await_coroutine(TaskController.serialize(do_stop))
    print("classic_task_sub_function: end")
    return res


# Main


async def amain(*args):
    # A simple example using a task defined by coroutine "task_main_function()"
    controller = TaskController()
    res = await controller.start_task(task_main_function, True)
    print('Task 1 done, expected result is bytes (a serialized task). Got a result of type:', type(res).__name__)

    # uncomment, if you like assembly code
    # res = pickletools.optimize(res) ; pickletools.dis(res)

    controller2 = TaskController()
    res = await controller2.continue_task(res, "value_to_be_returned_by_TaskController.serialize()")
    print('Task 2 done, expected result is str. Got a result of type:', type(res).__name__)
    assert res == "value_to_be_returned_by_TaskController.serialize()"

    print('');
    print('Now the same example using a task defined by a classic function "classic_task_main_function()" ' +\
          'converted to a coroutine with new_coroutine().')

    controller3 = TaskController()
    res = await controller3.start_task(new_coroutine, classic_task_main_function, True)
    print('Task 3 done, expected result is bytes (a serialized task). Got a result of type:', type(res).__name__)

    # res = pickletools.optimize(res) ; pickletools.dis(res)

    controller4 = TaskController()
    res = await controller4.continue_task(res, "another_value_to_be_returned_by_TaskController.serialize()")
    print('Task 4 done, expected result is str. Got a result of type:', type(res).__name__)
    assert res == "another_value_to_be_returned_by_TaskController.serialize()"



def main(*args):
    # import logging; logging.basicConfig(level=logging.DEBUG)
    print("demo start")
    print()
    asyncio.set_event_loop_policy(ExtendedEventLoopPolicy())
    asyncio.run(amain(*args), debug=True)
    print()
    print("demo end")


if __name__ == '__main__':
    sys.exit(main(*sys.argv))
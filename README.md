# slp_coroutine - Use Plain Old Functions as Coroutines

Using Stackless Python tasklets it is possible to decorate any plain old
Python callable as a C-Python coroutine, that can be used with the
asyncio framework. It is also possible to "async" call a coroutine
from a callable decorated as coroutine.

This Python module all sort of utility functions to

 * create a coroutine from a callable
 * call a coroutine from a callable
 * create an asynchronous generator/iterator form a normal generator/iterator
 * create a normal generator/iterator form an asynchronous normal generator/iterator
 * create an asynchronous context manager from a normal context manager
 * create a normal context manager from an asynchronous context manager
 
This code requires Stackless Python 3.7.6 or any later version.

Documentation: https://slp-coroutine.readthedocs.io/

Bug Tracker: https://github.com/akruis/slp_coroutine/issues

## Changelog

 * 0.0.1: Initial release
 * 0.0.2: Add the documentation to the source archive


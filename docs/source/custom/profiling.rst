submit PR to contributing section; modify docs on how to do profiling (cProfile, how to run, what to expect, how to read it)

#%%

Profiling
=========

ISOFIT has been successfully profiled using the built-in Python module [cProfile](https://docs.python.org/3/library/profile.html).

How to Use
----------

.. code::

  python -m cProfile [-o output_file] [-s sort_order] (-m module | myscript.py)


Outputs and Analysis
--------------------

Once a profiling data file is produced, there are a variety of ways to explore the results.

.. image:: images/example_profiler.png


Further Reading
---------------

Read the Docs: https://docs.python.org/3/library/profile.html
In-depth guide: https://www.machinelearningplus.com/python/cprofile-how-to-profile-your-python-code/

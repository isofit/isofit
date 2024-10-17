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

.. image:: images/example_profile_0.png

The above image uses [Snakeviz](https://jiffyclub.github.io/snakeviz/) to visualize the profiler results. This is the icicle view which shows the root at the top and all calls made by that function below it.

Clicking on a block will set that block as the root. This allows you to explore what functions and modules are calling others.

The time shown on each block is the cumulative time spent in that function. Children functions may have greater cumulative times than the parent, but that is because those functions were called elsewhere during the run.

.. image:: images/example_profile_1.png

The lower half of the visualizer includes the result values. There are six columns:

- ncalls: The number of calls to this function
- tottime: The total time spent in the given function (and excluding time made in calls to sub-functions)
- percall: The quotient of `tottime` divided by `ncalls`
- cumtime: The cumulative time spent in this and all subfunctions (from invocation till exit). This figure is accurate even for recursive functions.
- percall: The quotient of `cumtime` divided by primitive calls
- filename:lineno(function): Provides the respective data of each function

Depending on what information you are seeking, these columns may be sorted high/low. This example uses the high sort of `tottime` to analyze what functions take the most total time over the run.

ISOFIT Profiling Examples
-------------------------

ISOFIT comes with a few examples of generating profiling results found under `$(isofit path examples)/profiling_cube`. This example is based on the `$(isofit path examples)/image_cube/small_chunk`.

`$(isofit path examples)/profiling_cube` provides results for two different interpolation methods:

- RegularGrid (rg)
- Multilinear Grid (mlg)

In terms of speed, mlg >> nds > rg. The example will run each interpolation method 5 times. The user may then compile the results for comparison.

Further Reading
---------------

Read the Docs: https://docs.python.org/3/library/profile.html
In-depth guide: https://www.machinelearningplus.com/python/cprofile-how-to-profile-your-python-code/

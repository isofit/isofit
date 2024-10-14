"""
This module circumvents the ray package while maintaining ray-like syntax in the code.
Only the exact ISOFIT use cases of Ray are implemented here. If new uses of Ray are
implemented, those uses/functions will have to be defined here as well.

To enable, set the environment variable `ISOFIT_DEBUG` to "1". For example:

$ export ISOFIT_DEBUG=1
$ python isofit ...

Additionally, you may pass it as a temporary environment variable via:

$ ISOFIT_DEBUG=1 python isofit ...
"""

import logging
from types import FunctionType

Logger = logging.getLogger(__file__)


class Remote:
    def __init__(self, obj, *args, **kwargs):
        self.obj = obj
        self.args = args
        self.kwargs = kwargs

    def __getattr__(self, key):
        """
        Returns a Remote object on the key being requested. This enables
        ray.remote(Class).func.remote()
        """
        return Remote(getattr(self.obj, key))

    def remote(self, *args, **kwargs):
        return Remote(self.obj, *args, **kwargs)

    def get(self):
        return self.obj(*self.args, **self.kwargs)

    def __repr__(self):
        return f"<Remote({self.obj})>"


def __getattr__(key):
    """
    Reports any call to Ray that is not emulated
    """
    Logger.error(f"Unsupported operation: {key!r}")
    return lambda *a, **kw: None


def remote(*args, **kwargs):
    def wrap(obj):
        return Remote(obj)

    if len(args) == 1 and isinstance(args[0], (FunctionType, object)):
        return wrap(*args)
    return wrap


def init(*args, **kwargs):
    Logger.debug("Ray has been disabled for this run")


def get(jobs):
    if hasattr(jobs, "__iter__"):
        return [job.get() for job in jobs]
    else:
        return jobs.get()


def wait(jobs, num_returns=1, **kwargs):
    if hasattr(jobs, "__iter__"):
        if num_returns + 1 < len(jobs):
            return jobs[:num_returns], jobs[num_returns:]
        return jobs[:num_returns], []
    else:
        return jobs


def put(obj):
    return obj


def shutdown(*args, **kwargs):
    pass


class util:
    class ActorPool:
        def __init__(self, actors):
            """
            Emulates https://docs.ray.io/en/latest/_modules/ray/util/actor_pool.html

            Parameters
            ----------
            actors: list
                List of Remote objects to call
            """
            # Only need one actor function
            self.actors = Remote(actors[0].get())

        def map_unordered(self, func, iterable):
            return [func(self.actors, item).get() for item in iterable]

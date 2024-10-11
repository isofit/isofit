"""
Ray Wrapper module to circumvent the ray package while maintaining ray-like
syntax in the code. Only the exact ISOFIT use cases of Ray are wrapped here.
If new uses of Ray are implemented, those uses/functions will have to be wrapped
here as well.

To enable, set the environment variable `ISOFIT_DEBUG` to any value before
runtime. For example:
$ export ISOFIT_DEBUG=1
$ python isofit.py ...

Additionally, you may pass it as a temporary environment variable via:
$ ISOFIT_DEBUG=1 python isofit.py ...
"""

import logging
from types import FunctionType

Logger = logging.getLogger("isofit/wrappers/ray")


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
    print(f"__getattr__({key})")
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
            self.actors = [Remote(actor.get()) for actor in actors]

        def map_unordered(self, func, iterable):
            return [func(*pair).get() for pair in zip(self.actors, iterable)]

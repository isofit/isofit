"""
Ray Wrapper module to circumvent the ray package while maintaining ray-like
syntax in the code. Only the exact ISOFIT use cases of Ray are wrapped here.
If new uses of Ray are implemented, those uses/functions will have to be wrapped
here as well.
"""
import logging
import os

import ray

Logger = logging.getLogger("isofit/wrappers/ray")
DEBUG = os.environ.get("ISOFIT_DEBUG")


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


def remote(obj):
    return Remote(obj)


def init(self, *args, **kwargs):
    Logger.debug("Ray has been disabled for this run")


def get(jobs):
    if hasattr(jobs, "__iter__"):
        return [job.get() for job in jobs]
    else:
        return jobs.get()


def put(obj):
    return obj


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

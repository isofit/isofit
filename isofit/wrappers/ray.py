"""
Ray Wrapper module to circumvent the ray package while maintaining ray-like
syntax in the code. Only the exact ISOFIT use cases of Ray are wrapped here.
If new uses of Ray are implemented, those uses/functions will have to be wrapped
here as well.
"""
import logging

import ray

Logger = logging.getLogger("isofit/wrappers/ray.py")
DEBUG = False


class Remote:
    def __init__(self, obj, *args, **kwargs):
        self.obj = obj
        self.args = args
        self.kwargs = kwargs

    def __getattr__(self, key):
        return Remote(getattr(self.obj, key))

    def remote(self, *args, **kwargs):
        return Remote(self.obj, *args, **kwargs)

    def get(self):
        return self.obj(*self.args, **self.kwargs)

    def __repr__(self):
        return f"<Remote({self.obj})>"


class Ray:
    def __getattribute__(self, key):
        """
        __getattribute__ intercepts every attr lookup call
        """
        if DEBUG or key == "init":
            return object.__getattribute__(self, key)
        else:
            return getattr(ray, key)

    def init(self, *args, debug=False, **kwargs):
        global DEBUG
        DEBUG = debug
        if not DEBUG:
            ray.init(*args, **kwargs)
        else:
            Logger.debug("Ray has been disabled for this run.")

    @staticmethod
    def remote(obj):
        return Remote(obj)

    @staticmethod
    def get(jobs):
        if hasattr(jobs, "__iter__"):
            return [job.get() for job in jobs]
        else:
            return jobs.get()

    @staticmethod
    def put(obj):
        return obj

    class util:
        def __getattr__(self, key):
            """
            __getattr__ will check the class first then fallback to this if
            self.key doesn't exist.
            """
            return getattr(ray.util, key)

        class ActorPool:
            def __init__(self, actors):
                self.actors = [Remote(actor.get()) for actor in actors]

            def map_unordered(self, func, iterable):
                return [func(*pair).get() for pair in zip(self.actors, iterable)]

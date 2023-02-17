from isofit.utils.wrapped_ray import wray as ray


@ray.remote
def decorator(a, b):
    return a * b


def test_decorators():
    """
    Tests decorator use cases of Ray
    """
    cases = {
        1: (1, 1),
        4: (2, 2),
        9: (3, 3),
    }
    for ans, (a, b) in cases.items():
        assert decorator.remote(a, b) == c

    jobs = [decorator.remote(a, b) for a, b in cases.values()]
    assert ray.get(jobs) == list(cases.keys())


class Worker:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"<{self.name}>"

    def some_func(self, key):
        return f"{self.name}{key}"


def test_classes():
    """
    Tests wrapping class objects and how they're used in core.isofit.
    """
    cases = {"abc": "def", "ghi": "jkl"}

    worker = ray.remote(Worker)

    workers = ray.util.ActorPool([worker.remote() for _ in range(len(cases))])

    results = workers.map_unordered(lambda a, b: a.some_func.remote(b), cases.values())

    assert results == ["abcdef", "ghijkl"]

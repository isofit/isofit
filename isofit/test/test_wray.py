from isofit.wrappers import ray


@ray.remote(num_cpus=1)
def decorator(a, b):
    return a * b


@ray.remote()
def decorator_nocpu(a, b):
    return a * b


def test_decorators():
    """
    Tests decorator use cases of Ray
    """
    assert decorator.__module__ == "isofit.wrappers.ray"

    cases = {
        1: (1, 1),
        4: (2, 2),
        9: (3, 3),
    }
    for ans, (a, b) in cases.items():
        res = ray.get(decorator.remote(a, b))
        assert res == ans, f"Failed {a}*{b}, got {res} expected {ans}"

    jobs = [decorator.remote(a, b) for a, b in cases.values()]
    assert ray.get(jobs) == list(cases.keys())

    jobs = [decorator_nocpu.remote(a, b) for a, b in cases.values()]
    assert ray.get(jobs) == list(cases.keys())


class Worker:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"<{self.name}>"

    def some_func(self, key):
        return f"{self.name}{key}"


def test_classes(name="test", n=4):
    """
    Tests wrapping class objects and how they're used in core.isofit.
    """
    assert "isofit.wrappers.ray" in str(ray)

    name_id = ray.put(name)
    worker = ray.remote()(Worker)
    workers = ray.util.ActorPool([worker.remote(name_id) for _ in range(n)])

    results = workers.map_unordered(lambda a, b: a.some_func.remote(b), range(n))

    assert list(results) == [f"{name}{i}" for i in range(n)]

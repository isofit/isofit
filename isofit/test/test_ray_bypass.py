from isofit.debug import ray_bypass as ray


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
    assert decorator.__module__ == "isofit.debug.ray_bypass"

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


def test_classes(name="test", w=4, n=10):
    """
    Tests wrapping class objects and how they're used in core.isofit.
    """
    assert "isofit.debug.ray_bypass" in str(ray)

    name_id = ray.put(name)
    worker = ray.remote()(Worker)
    workers = ray.util.ActorPool([worker.remote(name_id) for _ in range(w)])

    results = workers.map_unordered(lambda a, b: a.some_func.remote(b), range(n))

    assert list(results) == [f"{name}{i}" for i in range(n)]


def test_options(name="test", w=4, n=10):
    """
    Tests `.options()` scheduling directives, used by the torch backend to
    request one actor per GPU: `Worker.options(num_gpus=1).remote(...)`.
    Resource requests are meaningless without a scheduler, so they are ignored
    while the ActorPool call chain keeps working exactly as in test_classes.
    """
    name_id = ray.put(name)
    worker = ray.remote()(Worker)
    workers = ray.util.ActorPool(
        [worker.options(num_gpus=1, num_cpus=1).remote(name_id) for _ in range(w)]
    )

    results = workers.map_unordered(lambda a, b: a.some_func.remote(b), range(n))

    assert list(results) == [f"{name}{i}" for i in range(n)]


def test_options_with_no_kwargs():
    """Bare .options() must also be a no-op passthrough."""
    worker = ray.remote()(Worker)
    workers = ray.util.ActorPool([worker.options().remote("bare")])

    results = workers.map_unordered(lambda a, b: a.some_func.remote(b), ["!"])

    assert list(results) == ["bare!"]


def test_options_returns_chainable_remote():
    """options() must not consume the handle; .remote() still follows it."""
    worker = ray.remote()(Worker)
    assert isinstance(worker.options(num_gpus=1), ray.Remote)
    assert isinstance(worker.options(num_gpus=1).remote("x"), ray.Remote)

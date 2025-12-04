from pathlib import Path

import pytest

from isofit import ray
from isofit.data import env


def normalize(paths):
    return {key: Path(path).as_posix().replace("D:", "") for key, path in paths.items()}


@pytest.fixture(scope="session")
def changedDirs():
    env.reset()

    # Changed dirs
    dirs = {key: f"/abc/{key}" for key in env._dirs}

    # Default keys
    keys = {key: env[key] for key in env._keys}

    return normalize({**dirs, **keys})


@pytest.fixture(scope="session")
def changedKeys():
    env.reset()

    # Default dirs
    dirs = {key: env[key] for key in env._dirs}

    # Changed keys
    keys = {key: f"/xyz/{key}" for key in env._keys}

    return normalize({**dirs, **keys})


@pytest.fixture(scope="session")
def changedBoth(changedDirs, changedKeys):
    both = {}

    for key in env._dirs:
        both[key] = changedDirs[key]

    for key in env._keys:
        both[key] = changedKeys[key]

    return both


def test_changePath(changedDirs):
    env.reset()

    for key in env._dirs:
        env.changePath(key, changedDirs[key])

    assert normalize(dict(env)) == changedDirs


def test_changeKey(changedKeys):
    env.reset()

    for key in env._keys:
        env.changeKey(key, changedKeys[key])

    assert normalize(dict(env)) == changedKeys


def test_changeBase(changedDirs):
    env.reset()

    env.changeBase("/abc")

    assert normalize(dict(env)) == changedDirs


@ray.remote
def check_ini():
    from isofit.data import env

    return env.ini


def test_custom_ini(tmp_path: Path):
    ini = tmp_path / "custom_test.ini"
    ini.write_text(f"[DEFAULT]\ndata = {tmp_path}")

    env.load(ini)
    remote = ray.get(check_ini.remote())

    assert env.ini == remote, "The custom ini was not loaded in a child ray process"


def test_raise_path_errors():
    # Make sure an exception is not raised
    env.path("data", "fake.file")

    # Now make sure it is raised
    env.changeKey("raise_path_errors", True)
    with pytest.raises(FileNotFoundError):
        env.path("data", "fake.file")

import pytest

from isofit.data import env


@pytest.fixture(scope="session")
def changedDirs():
    env.reset()

    # Changed dirs
    dirs = {key: f"/abc/{key}" for key in env._dirs}

    # Default keys
    keys = {key: env[key] for key in env._keys}

    return {**dirs, **keys}


@pytest.fixture(scope="session")
def changedKeys():
    env.reset()

    # Default dirs
    dirs = {key: env[key] for key in env._dirs}

    # Changed keys
    keys = {key: f"/xyz/{key}" for key in env._keys}

    return {**dirs, **keys}


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

    assert dict(env) == changedDirs


def test_changeKey(changedKeys):
    env.reset()

    for key in env._keys:
        env.changeKey(key, changedKeys[key])

    assert dict(env) == changedKeys


def test_changeBase(changedDirs):
    env.reset()

    env.changeBase("/abc")

    assert dict(env) == changedDirs

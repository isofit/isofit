from configparser import ConfigParser
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from isofit.core import env


def test_getattr_existing_key():
    assert env.data == env.CONFIG["DEFAULT"]["data"]


def test_changePath():
    expected = {}
    for key in env.KEYS:
        expected[key] = (val := f"/test/{key}")
        env.changePath(key, val)

    assert env.CONFIG["DEFAULT"] == expected


@patch("builtins.open")
def test_loadEnv_file_exists(mock_open):
    path = "test.ini"
    data = {key: f"/test/{key}" for key in env.KEYS}

    with patch("pathlib.Path.exists", return_value=True), patch.object(
        env.CONFIG, "read", return_value=None
    ):
        env.load(path)

    assert dict(env.CONFIG["DEFAULT"]) == data
    assert env.INI == Path(path)


@patch("builtins.open")
def test_loadEnv_file_not_exists(mock_open):
    path = "test.ini"
    with patch("pathlib.Path.exists", return_value=False):
        env.load(path)

    # load changes INI for the remainder of the session
    assert env.INI == Path(path)


@patch("builtins.open")
def test_saveEnv(mock_open):
    path = "test.ini"
    with patch.object(env.CONFIG, "write"), patch("pathlib.Path.mkdir"):
        env.save(path)

        # save changes INI for the remainder of the session
        assert env.INI == Path(path)

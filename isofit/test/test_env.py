from configparser import ConfigParser
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from isofit.core import env


def test_getattr_existing_key():
    assert env.data == env.CONFIG["DEFAULT"]["data"]


def test_changePath():
    key = "data"
    val = "/abc"
    env.changePath(key, val)

    assert env.CONFIG["DEFAULT"][key] == val


@patch("builtins.open", new_callable=mock_open)
def test_loadEnv_file_exists(mock_open):
    path = "/abc.ini"
    data = {f"/abc/{key}" for key in env.KEYS}

    with patch("pathlib.Path.exists", return_value=True), patch.object(
        env.CONFIG, "read", return_value=None
    ):
        env.load(path)

    mock_open.assert_called_once_with(path, "r")
    assert dict(env.CONFIG["DEFAULT"]) == data
    assert env.INI == Path(path)


@patch("builtins.open", new_callable=mock_open)
def test_loadEnv_file_not_exists(mock_open):
    path = "/abc.ini"
    with patch("pathlib.Path.exists", return_value=False):
        env.load(path)

    # load changes INI for the remainder of the session
    assert env.INI == Path(path)


@patch("builtins.open", new_callable=mock_open)
def test_saveEnv(mock_open):
    path = "/abc.ini"
    with patch.object(env.CONFIG, "write") as mock_write, patch(
        "pathlib.Path.parent.mkdir"
    ) as mock_mkdir:
        env.save(path)

        mock_mkdir.assert_called_once()

        # save changes INI for the remainder of the session
        assert env.INI == Path(path)

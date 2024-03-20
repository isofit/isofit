from pathlib import Path
from unittest.mock import mock_open, patch

import pytest
import toml

from isofit.core import env


def test_getattr_existing_key():
    assert env.data == str(env.HOME / ".isofit/data")


def test_changePath():
    key = "data"
    val = "/abc"
    env.changePath(key, val)

    assert env.DATA[key] == val


@patch("builtins.open", new_callable=mock_open)
def test_loadEnv_file_exists(mock_open):
    path = "/abc.toml"
    data = {f"/abc/{key}" for key in env.KEYS}

    with patch("pathlib.Path.exists", return_value=True), patch(
        "toml.load", return_value=data
    ):
        env.loadEnv(path)

    mock_open.assert_called_once_with(path, "r")
    assert env.DATA == data
    assert env.TOML == Path(path)


@patch("builtins.open", new_callable=mock_open)
def test_loadEnv_file_not_exists(mock_open):
    path = "/abc.toml"
    with patch("pathlib.Path.exists", return_value=False):
        env.loadEnv(path)

    # loadEnv changes TOML for the remainder of the session
    assert env.TOML == Path(path)


@patch("builtins.open", new_callable=mock_open)
def test_saveEnv(mock_open):
    path = "/test/save_env_test.toml"
    with patch("pathlib.Path.parent.mkdir") as mock_mkdir:
        env.saveEnv(path)

        mock_mkdir.assert_called_once()

        # saveEnv changes TOML for the remainder of the session
        assert env.TOML == Path(path)

"""``pytest`` configuration"""


def pytest_collection_modifyitems(items, config):
    """Modify collected tests.

    ``pytest`` does not offer a mechanism for selecting only tests lacking a
    marker. Instead, detect tests without a marker, and quietly add an
    ``unmarked`` marker that can be used to select these tests.
    """

    # Detecting an unmarked test is actually a bit difficult. Any marker
    # generated with '@pytest.mark' counts as a marked test, including
    # '@pytest.mark.parametrize()'. To get around this, we set
    # 'addopts = --strict-markers' in 'pytest.ini' to force all markers to be
    # registered, then we can consider any marker that is not registered to be
    # a bulitin marker, and thus consider any test not marked with a custom
    # marker to be 'unmarked'.

    # Names of markers that are explicitly listed in the config file. I had to
    # read the '$ pytest' source code to figure out how to do this. This is how
    # '$ pytest --markers' works - the 'pytest' library does not provide any
    # abstraction for retrieving this information.
    registered_markers = [m.split(":")[0] for m in config.getini("markers")]

    for item in items:
        if not any(m.name in registered_markers for m in item.iter_markers()):
            item.add_marker("unmarked")

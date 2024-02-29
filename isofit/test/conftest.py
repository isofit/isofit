"""``pytest`` configuration"""


def pytest_collection_modifyitems(items, config):
    """Modify collected tests.

    ``pytest`` does not offer a mechanism for selecting only tests lacking a
    marker. Instead, detect tests without a marker, and quietly add an
    ``unmarked`` marker that can be used to select these tests.
    """

    # Detecting an unmarked test is actually a bit difficult. Any marker generated with
    # '@pytest.mark' counts as a marked test, including '@pytest.mark.parametrize()'.
    # To get around this, we set 'addopts = --strict-markers' in 'pytest.ini' to
    # force all markers to be registered, then we can consider any marker that is
    # not registered to be a bulitin marker, and thus consider any test not marked
    # with a custom marker to be 'unmarked'.

    # Get registered markers from the 'pytest' configuration.
    registered_markers = config.getini("markers")

    for item in items:
        for marker in item.iter_markers():
            if marker.name in registered_markers:
                continue
        else:
            item.add_marker("unmarked")

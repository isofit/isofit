"""``pytest`` configuration"""


def pytest_collection_modifyitems(items, config):
    """Modify collected tests.

    ``pytest`` does not offer a mechanism for selecting only tests lacking a
    marker. Instead, detect tests without a marker, and quietly add an
    ``unmarked`` marker that can be used to select these tests.
    """

    for item in items:
        if not any(item.iter_markers()):
            item.add_marker("unmarked")

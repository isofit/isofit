Documentation
=============

Documentation is an ongoing effort for the isofit codebase.  Your contributions are greatly
appreciated.  In general, we prefer the use of Google Doc Strings, and the use of Python 3.6+
typing specification, where possible.  Good models for how documentation should be updated
are the isofit/utils/apply_oe.py and isofit/core/common.py files.

We use sphinx-autodoc to build the documentation automatically.  If no major code structures
are changed, documentation will update automatically via githooks and be available at
https://isofit.readthedocs.io/en/latest/index.html

However, if there are major structural changes, the source rst files will need to be rebuilt.
This can be done by ::

    cd docs/
    rm source/isofit*.rst
    make build_docs

You can also build a local copy of the documentation by running::

    cd docs/
    make html
    open build/html/index.html

However, if you do a local build of the documentation, **do not commit the
contents of docs/build**. These files do not need to be hosted on the repository, are
not tightly compressed, and will change frequently.
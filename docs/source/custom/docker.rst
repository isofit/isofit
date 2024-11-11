ISOFIT Docker
=============
The official container images are located on Dockerhub under `jammont/isofit <https://hub.docker.com/r/jammont/isofit>`_.
They come pre-configured with everything a user may need to get started using ISOFIT and executing its examples.

Tags
----
- ``latest`` - Most recent released ISOFIT version
- ``3.x.x`` - Specific release builds

Images are currently built for ``amd64`` and ``arm64`` architectures.


Getting Started
---------------

Start by pulling the image:

.. code-block::

    $ docker pull jammont/isofit:latest

This image is default tagged as ``jammont/isofit``. As such, to use it run:

.. code-block::

    $ docker run -it --rm --shm-size=16gb jammont/isofit bash

A breakdown of the above command:

.. list-table::
    :widths: 20 80
    :header-rows: 1

    * - Flag
      - Purpose
    * - ``-it``
      - Run in ``[i]nteractive`` and ``[t]erminal`` modes. Any commands after ``jammont/isofit`` will be executed from the container
    * - ``--rm``
      - Removes the container once finished. If you intend to modify the container between sessions, remove this flag. View existing containers via ``docker ps``
    * - ``--shm-size=16gb``
      - Expands the shared memory space for the container. This is used by Ray and, if not set, may have significant performance impacts. The larger the better, this may need to be tweaked to your system.
    * - ``jammont/isofit``
      - The ``tag`` name of the image. If you built your own ISOFIT image using a different tag, be sure to replace this string. Defaults to ``jammont/isofit:latest``, or pull a specific version like ``jammont/isofit:3.2.1``
    * - ``bash``
      - The command to run for ``-it``. Using ``bash`` here will invoke a bash instance for the user. If you wanted to launch a specific script without entering the container, you could replace this command, such as ``isofit -v`` to get the installed ISOFIT version.

Given that the above command invokes a bash shell, the user will be placed inside the container.
From here you may use ISOFIT via the commandline or make further edits.
If ``--rm`` is enabled, exiting the container will delete it along with any changes made.

You may need to attach volumes to the container at runtime in order to access files external to the container. For instance:

.. code-block::

    $ docker run -v /host/path:/container/path jammont/isofit

This will provide ``/host/path`` available inside of the container as ``/container/path``.
As such, any changes made inside the container to this path or its contents will be reflected on the host.

Additional examples for executing ISOFIT interactively:

.. code-block::

    # Print version
    $ docker run --rm -it jammont/isofit isofit --version

    # Execute an example
    $ docker run --rm -it --shm-size=16gb jammont/isofit bash examples/20151026_SantaMonica/run.sh
    $ docker run --rm -it --shm-size=16gb jammont/isofit bash examples/image_cube/small/analytical.sh


Jupyter
-------
The default run command for the container is to start up a Jupyterlab server on internal port 8888.
To connect to this port and make it accessible via the browser, pass the ``-p [host]:8888`` parameter:

.. code-block::

    $ docker run --rm --shm-size=16gb -p 8888:8888 jammont/isofit

This will start up the Jupyterlab server on port ``8888``. Navigate to ``127.0.0.1:8888`` in a web browser to start using the server.

To shutdown, hit CTRL-C in the running terminal.


Building the Image
------------------

The ISOFIT container can be manually built by pulling the repository and running ``docker build --tag [name] .`` in the repository's root (where the ``Dockerfile`` is located).
For example, to build a specific branch of ISOFIT, the following may be performed:

.. code-block::

    git clone https://github.com/isofit/isofit.git
    cd isofit
    git checkout [branch]
    docker build --tag [branch] .

This will build a container and tag it as ``[branch]``. This tag can anything as it is local to your device.

Once the container is built, refer to the ``Getting Started`` section for next steps.
Replace the default ``jammont/isofit`` tag with the tag you chose for your newly built image.

ISOFIT Docker
=============

The ISOFIT team provides a prebuilt Docker image that can be used for any purpose.

This image uses [micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html) to provide two virtual environments:

- `isofit` - The default environment that has ISOFIT installed alongside all of its dependencies.
- `test` - The testing environment that does not include ISOFIT but does include its dependencies.
- `nodeps` - An additional testing environment that does not include ISOFIT or its dependencies. This is primarily used for testing the installation process of ISOFIT.

All three environments use Python 3.10. To activate an environment at container runtime, provide the flag `-e ENV_NAME=[env name]`.

Additionally, 6S and sRTMnet are preinstalled under `/6sv-2.1` and `/sRTMnet_v100` respectively. ISOFIT is installed under `/isofit`.

How to Use
----------

Start by pulling the image:

.. code-block:: bash
  docker pull jammont/isofit:latest

This image is default tagged as `jammont/isofit`. As such, to use it run:

.. code-block:: bash
  $ docker run -it --rm jammont/isofit bash

- `-it` - Run in `interactive` and `terminal` modes, which will build a container and place the user inside of it.
- `--rm` - Removes the container once finished. If you intend to modify the container between sessions, remove this flag.
- `jammont/isofit` - The `tag` name of the image. If you built your own ISOFIT image using a different tag, be sure to replace this string.
- `bash` - The command to run for `-it`. Using `bash` here will invoke a bash instance for the user. If you wanted to launch a specific script without entering the container, you could replace this command eg. `python /some/path/to/a/script.py`

After running the above command, you will placed inside the container as the root user. From here you may proceed to use ISOFIT as you need.

You may need to attach volumes to the container at runtime in order to access files external to the container. For instance:

.. code-block:: bash
  docker run -it -v /host/path:/container/path --rm jammont/isofit bash

This will provide `/host/path` available inside of the container as `/container/path`. As such, any changes made inside the container to this path or its contents will be reflected on the host.

Building the Image
------------------

The ISOFIT container can be manually built by pulling the repository and running `docker build --tag [name] .` in the repository's root.
For example, to build a specific branch of ISOFIT, the following may be performed:

.. code-block:: bash
  git clone https://github.com/isofit/isofit.git
  cd isofit
  git checkout [branch]
  docker build --tag [branch] .

This will build a container and tag it as [branch]. This tag can anything as it is local to your device.

Once the container is built, refer to the `How to Use` section for next steps.

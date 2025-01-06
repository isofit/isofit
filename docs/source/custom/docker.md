These are the official container images for the [ISOFIT](https://github.com/isofit/isofit) package. They come pre-configured with everything a user may need to get started using ISOFIT and executing its examples.

Tags
----
- `latest` - Most recent released ISOFIT version
- `3.x.x` - Specific release builds

Getting Started
---------------
Start by pulling the image:

```bash
$ docker pull jammont/isofit:latest
```

This image is default tagged as `jammont/isofit`. As such, to use it run:

```bash
$ docker run -it --rm --shm-size=16gb jammont/isofit bash
```

| Flag              | Reason |
| ----              | ------ |
| `-it`             | Run in `[i]nteractive` and `[t]erminal` modes, which will build a container and place the user inside of it. |
| `--rm`            | Removes the container once finished. If you intend to modify the container between sessions, remove this flag. |
| `--shm-size=16gb` | Expands the shared memory space for the container. This is used by Ray and, if not set, may have significant performance impacts. The larger the better, this may need to be tweaked to your system. |
| `jammont/isofit`  | The `tag` name of the image. If you built your own ISOFIT image using a different tag, be sure to replace this string. |
| `bash`            | The command to run for `-it`. Using `bash` here will invoke a bash instance for the user. If you wanted to launch a specific script without entering the container, you could replace this command. |

After running the above command, you will placed inside the container as the root user. From here you may proceed to use ISOFIT as you need.

You may need to attach volumes to the container at runtime in order to access files external to the container. For instance:

```
$ docker run --rm -it -v /host/path:/container/path jammont/isofit bash
```

This will provide `/host/path` available inside of the container as `/container/path`. As such, any changes made inside the container to this path or its contents will be reflected on the host.

Additional examples for executing ISOFIT interactively:

```bash
# Print version
$ docker run --rm -it jammont/isofit isofit --version

# Execute an example
$ docker run --rm -it --shm-size=16gb jammont/isofit bash examples/20151026_SantaMonica/run.sh
$ docker run --rm -it --shm-size=16gb jammont/isofit bash examples/image_cube/small/analytical.sh
```

Jupyter
-------
The default run command for the container is to start up a Jupyterlab server on internal port 8888.
To connect to this port and make it accessible via the browser, pass the `-p [host]:8888` parameter:

```
$ docker run --rm --shm-size=16gb -p 8888:8888 jammont/isofit
```

This will start up the Jupyterlab server on port `8888`. Navigate to `127.0.0.1:8888` in a web browser to start using the server.

To shutdown, hit CTRL-C in the running terminal.

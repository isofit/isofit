# Environments

ISOFIT maintains multiple Conda environment YAML files to reduce the total dependencies of the package. The table below describes the purpose for each file.


File         | Purpose
-------------|--------
`isofit.yml` | Basic required packages to get started with ISOFIT
`docker.yml` | Extra utilities for the docker images

These can be installed together using [mamba](https://mamba.readthedocs.io/en/latest) by using multiple `-f`, `--file` flags when creating a new environment. For example, our Dockerfile installs two files using micromamba:

```bash
$ micromamba create  --name isofit python=3.10
$ micromamba install --name isofit --file ISOFIT/recipe/isofit.yml \
                                   --file ISOFIT/recipe/docker.yml
```

# Docker

There are two types of docker files officially supported:

File                     | Purpose
-------------------------|--------
`Dockerfile`             | Default image that contains all extra ISOFIT dependencies
`recipe/slim.Dockerfile` | Code-only slim version. For advanced users, will require mounting dependencies

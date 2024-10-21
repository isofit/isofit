FROM rayproject/ray:latest-py310

USER root
RUN apt-get update &&\
    apt-get install --no-install-recommends -y \
      gfortran \
      make \
      unzip

USER ray
WORKDIR /home/ray

# Copy and install ISOFIT
COPY --chown=ray:users . isofit/
RUN conda config --prepend channels conda-forge &&\
    conda update --all --yes &&\
    conda create --name isofit --clone base &&\
    conda install --name base --solver=classic conda-libmamba-solver nb_conda_kernels jupyterlab &&\
    conda env update --name isofit --solver=libmamba --file isofit/recipe/environment_isofit_basic.yml &&\
    conda install --name isofit --solver=libmamba ipykernel &&\
    anaconda3/envs/isofit/bin/pip install --no-deps -e isofit &&\
    echo "conda activate isofit" >> ~/.bashrc &&\
    echo "alias mi=conda install --solver=libmamba"

ENV PATH anaconda3/envs/isofit/bin:$PATH

# Install ISOFIT extra files
RUN isofit -b . download all &&\
    isofit build

# Explicitly set the shell to bash so the Jupyter server defaults to it
ENV SHELL /bin/bash

# Ray Dashboard port
EXPOSE 8265

# Start the Jupyterlab server
EXPOSE 8888
CMD isofit/scripts/startJupyter.sh

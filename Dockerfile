FROM rayproject/ray:2.4.0-py310-aarch64

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
RUN conda install --name base --solver=classic conda-libmamba-solver &&\
    conda env update --name base --solver=libmamba --file isofit/recipe/unpinned.yml &&\
    conda install --name base jupyter &&\
    pip install --no-deps -e isofit

# Install 6S
RUN mkdir 6sv-2.1 &&\
    cd 6sv-2.1 &&\
    wget https://github.com/ashiklom/isofit/releases/download/6sv-mirror/6sv-2.1.tar
RUN cd 6sv-2.1 &&\
    tar -xf 6sv-2.1.tar &&\
    rm 6sv-2.1.tar &&\
    sed -i Makefile -e 's/FFLAGS.*/& -std=legacy/' &&\
    make
ENV SIXS_DIR="/home/ray/6sv-2.1"

# Install sRTMnet
RUN mkdir sRTMnet_v100 &&\
    cd sRTMnet_v100 &&\
    wget https://zenodo.org/record/4096627/files/sRTMnet_v100.zip
RUN cd sRTMnet_v100 &&\
    unzip sRTMnet_v100.zip &&\
    rm sRTMnet_v100.zip
ENV EMULATOR_PATH="/home/ray/sRTMnet_v100/sRTMnet_v100"

# Some ISOFIT examples require this env var to be present but does not need to be installed
ENV MODTRAN_DIR=""

# Start a Jupyter server
CMD jupyter notebook --ip 0.0.0.0 --no-browser --allow-root


# FROM alpine:3.14 AS build

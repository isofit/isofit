FROM rayproject/ray:2.4.0-py310-aarch64

USER root
# sudo apt-get update && sudo apt-get install --no-install-recommends -y gfortran
RUN apt-get update &&\
    apt-get install --no-install-recommends -y \
      gfortran \
      make \
      unzip

USER ray
WORKDIR /home/ray

# Copy and install ISOFIT
COPY --chown=ray:users . isofit/
RUN conda create --name isofit --clone base &&\
    conda install --name base --solver=classic conda-libmamba-solver nb_conda_kernels jupyter jupyterthemes &&\
    conda env update --name isofit --solver=libmamba --file isofit/recipe/unpinned.yml &&\
    conda install --name isofit ipykernel &&\
    anaconda3/envs/isofit/bin/pip install --no-deps -e isofit &&\
    echo "conda activate isofit" >> ~/.bashrc
    echo "LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1:$LD_PRELOAD" >> ~/.bashrc

# Install 6S
RUN mkdir 6sv-2.1 &&\
    cd 6sv-2.1 &&\
    wget https://github.com/ashiklom/isofit/releases/download/6sv-mirror/6sv-2.1.tar &&\
    tar -xf 6sv-2.1.tar &&\
    rm 6sv-2.1.tar &&\
    sed -i Makefile -e 's/FFLAGS.*/& -std=legacy/' &&\
    make
ENV SIXS_DIR="/home/ray/6sv-2.1"

# Install sRTMnet
RUN mkdir sRTMnet_v100 &&\
    cd sRTMnet_v100 &&\
    wget https://zenodo.org/record/4096627/files/sRTMnet_v100.zip &&\
    unzip sRTMnet_v100.zip &&\
    rm sRTMnet_v100.zip
ENV EMULATOR_PATH="/home/ray/sRTMnet_v100/sRTMnet_v100"

# Some ISOFIT examples require this env var to be present but does not need to be installed
ENV MODTRAN_DIR=""

# Start a Jupyter server
EXPOSE 8888
CMD jupyter notebook --ip 0.0.0.0 --no-browser --allow-root

# FROM alpine:3.14 AS build

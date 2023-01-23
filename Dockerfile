FROM mambaorg/micromamba:latest
USER root

WORKDIR /

RUN apt-get update &&\
    apt-get install --no-install-recommends -y \
      build-essential \
      ca-certificates \
      curl            \
      git             \
      gfortran        \
      unzip

# Install 6S
RUN mkdir /6sv-2.1 &&\
    cd /6sv-2.1 &&\
    curl -SLO https://github.com/ashiklom/isofit/releases/download/6sv-mirror/6sv-2.1.tar &&\
    tar -xf 6sv-2.1.tar &&\
    rm 6sv-2.1.tar &&\
    sed -i Makefile -e 's/FFLAGS.*/& -std=legacy/' &&\
    make
ENV SIXS_DIR /6sv-2.1

# Install sRTMnet
RUN mkdir /sRTMnet_v100 &&\
    cd /sRTMnet_v100 &&\
    curl -SLO https://zenodo.org/record/4096627/files/sRTMnet_v100.zip &&\
    unzip sRTMnet_v100.zip &&\
    rm sRTMnet_v100.zip
ENV EMULATOR_PATH /sRTMnet_v100/sRTMnet_v100

# Prebuild the virtual environments
# Default environment is isofit. To activate the different one, use: docker run -e ENV_NAME=[env name]
RUN micromamba create -y -n isofit -c conda-forge python=3.10 && \
    micromamba create -y -n test   -c conda-forge python=3.10 && \
    micromamba create -y -n nodeps -c conda-forge python=3.10

# Auto activate environments
ARG MAMBA_DOCKERFILE_ACTIVATE=1

COPY . /isofit

# Setup the test environment
ENV ENV_NAME test
RUN micromamba install --name test --yes --file /isofit/recipe/environment_isofit_basic.yml &&\
    micromamba install --name test --yes --channel conda-forge pip &&\
    micromamba clean --all --yes &&\
    pip install ray ndsplines xxhash --upgrade

# Install ISOFIT
ENV ENV_NAME isofit
RUN micromamba install --name isofit --yes --file /isofit/recipe/environment_isofit_basic.yml &&\
    micromamba install --name isofit --yes --channel conda-forge pip &&\
    micromamba clean --all --yes &&\
    pip install ray ndsplines xxhash --upgrade &&\
    pip install --editable isofit

# Run test examples at startup if not in -it mode
CMD echo "Example: image_cube" &&\
    cd /isofit/examples/image_cube/ && bash run_example_cube.sh &&\
    echo "Example: 20151026_SantaMonica" &&\
    cd /isofit/examples/20151026_SantaMonica/ && bash run_examples.sh &&\
    echo "Example: 20171108_Pasadena" &&\
    cd /isofit/examples/20171108_Pasadena/ && bash run_example_modtran.sh

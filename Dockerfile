FROM --platform=$BUILDPLATFORM mambaorg/micromamba

USER root
RUN apt-get update &&\
    apt-get install --no-install-recommends -y \
      gfortran \
      make \
      nano

USER mambauser
WORKDIR /home/mambauser

# Copy and install the ISOFIT environment
COPY --chown=mambauser:mambauser . isofit/
RUN micromamba config prepend channels conda-forge &&\
    micromamba update --all --yes &&\
    micromamba create --name isofit python=3.10 &&\
    micromamba install --name base nb_conda_kernels jupyterlab &&\
    micromamba install --name isofit --file isofit/recipe/environment_isofit_basic.yml &&\
    micromamba install --name isofit ipykernel &&\
    echo "micromamba activate isofit" >> ~/.bashrc

ENV PATH=/opt/conda/envs/isofit/bin:$PATH

# Install ISOFIT and extra files
RUN pip install -e isofit &&\
    isofit -b . download all &&\
    isofit build

# Explicitly set the shell to bash so the Jupyter server defaults to it
ENV SHELL=["/bin/bash", "-l", "-c"]

# Ray Dashboard port
EXPOSE 8265

# Start the Jupyterlab server
EXPOSE 8888
CMD isofit/scripts/startJupyter.sh

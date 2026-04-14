FROM --platform=$BUILDPLATFORM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

USER root
RUN apt-get update &&\
    apt-get install -y --no-install-suggests --no-install-recommends \
      g++ \
      gfortran \
      make \
      libgsl-dev \
      libnetcdf-dev &&\
    rm -rf /var/lib/apt/lists/*

WORKDIR /root

# Copy and install the ISOFIT environment
COPY . ISOFIT/
RUN cd ISOFIT &&\
    uv sync --extra docker --extra test &&\
    uv run isofit -b .. download all &&\
    uv run isofit build &&\
    echo "source /root/ISOFIT/.venv/bin/activate" >> ~/.bashrc &&\
    echo "alias launch=\"jupyter-lab --ip 0.0.0.0 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''\"" >> ~/.bashrc

# Jupyter needs this to access the terminal
ENV SHELL="/bin/bash"

# Ray Dashboard port
EXPOSE 8265

# Start the Jupyterlab server
EXPOSE 8888

CMD ["ISOFIT/.venv/bin/jupyter-lab", "--ip", "0.0.0.0", "--no-browser", "--allow-root", "--ServerApp.token=''", "--ServerApp.password=''"]

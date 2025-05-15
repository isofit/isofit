FROM --platform=$BUILDPLATFORM python:3.10-slim

USER root
RUN apt-get update &&\
    apt-get install --no-install-recommends -y \
      gfortran \
      make \
      nano &&\
    rm -rf /var/lib/apt/lists/*

WORKDIR /root

# Copy and install the ISOFIT environment
COPY . ISOFIT/
RUN pip install --upgrade pip && pip install -e "ISOFIT[docker]" jupyterlab &&\
    python -m ipykernel install --user --name isofit &&\
    echo "alias launch=\"jupyter-lab --ip 0.0.0.0 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''\"" >> ~/.bashrc

# Jupyter needs this to access the terminal
ENV SHELL="/bin/bash"

# Ray Dashboard port
EXPOSE 8265

# Start the Jupyterlab server
EXPOSE 8888

CMD ["jupyter-lab", "--ip", "0.0.0.0", "--no-browser", "--allow-root", "--ServerApp.token=''", "--ServerApp.password=''"]

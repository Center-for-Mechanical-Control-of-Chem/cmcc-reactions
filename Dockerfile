FROM condaforge/miniforge3:latest
ENV DEBIAN_FRONTEND=noninteractive PATH=/opt/conda/bin:$PATH

RUN printf '%s\n' \
    '#!/bin/bash' \
    'conda run --no-capture-output -n cmcc-rxn jupyter "$@"' \
    > /usr/bin/jupyter && \
    chmod +x /usr/bin/jupyter

ARG CACHEBUST
COPY environment.yml environment_mace.yml environment_uma.yml cli.py ./

RUN apt-get update && apt-get -y install --no-install-recommends git gcc g++ && \
    mamba env create -f environment.yml && \
    mamba env create -f environment_mace.yml && \
    mamba env create -f environment_uma.yml && \
    mamba clean --all -afy && \
    find /opt/conda -follow -type f -name '*.a' -delete && \
    find /opt/conda -follow -type f -name '*.pyc' -delete && \
    find /opt/conda -follow -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true

ENTRYPOINT ["python", "/cli.py"]
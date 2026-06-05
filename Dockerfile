FROM continuumio/miniconda3:latest

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get -y install git gcc g++ && \
    rm -rf /var/lib/apt/lists/*

ARG CACHEBUST

COPY environment.yml .
COPY environment_mace.yml .
COPY environment_uma.yml .
COPY cli.py .

RUN conda env create -f environment.yml && \
    conda env create -f environment_mace.yml && \
    conda env create -f environment_uma.yml && \
    conda clean --all -afy

ENV PATH=/opt/conda/bin:$PATH

RUN printf '%s\n' \
    '#!/bin/bash' \
    'conda run --no-capture-output -n cmcc-rxn jupyter "$@"' \
    > /usr/bin/jupyter && \
    chmod +x /usr/bin/jupyter


ENTRYPOINT ["python", "cli.py"]
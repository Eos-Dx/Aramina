FROM python:3.13-slim-bookworm

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /opt

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        build-essential \
        ocl-icd-libopencl1 \
    && rm -rf /var/lib/apt/lists/*

COPY XRD-preprocessing /opt/XRD-preprocessing

# Linux pyFAI is deliberately pinned: it is the validated integration runtime.
RUN python -m pip install --upgrade pip \
    && python -m pip install pyFAI==2026.5.0 fabio==2025.10.0 pyopencl==2026.1.2 \
    && python -m pip install /opt/XRD-preprocessing

COPY Aramis /opt/Aramis
COPY run_training_docker.sh /opt/aramis-bundle/run_training_docker.sh
COPY run_prediction_examples_docker.sh /opt/aramis-bundle/run_prediction_examples_docker.sh
COPY run_prediction_docker.sh /opt/aramis-bundle/run_prediction_docker.sh

RUN python -m pip install --no-deps /opt/Aramis \
    && chmod +x /opt/aramis-bundle/run_training_docker.sh \
    && chmod +x /opt/aramis-bundle/run_prediction_examples_docker.sh \
    && chmod +x /opt/aramis-bundle/run_prediction_docker.sh

WORKDIR /opt/Aramis

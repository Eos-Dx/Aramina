FROM python:3.13-slim-bookworm

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /opt

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ocl-icd-libopencl1 \
    && rm -rf /var/lib/apt/lists/*

COPY XRD-preprocessing /opt/XRD-preprocessing

# Deliberately exclude notebook and visualization-only dependencies. The bundle
# contains the fixed product preprocessing, training, and prediction runtime.
RUN python -m pip install --upgrade pip \
    && python -m pip install \
        numpy==2.4.1 \
        pandas==3.0.5 \
        scipy==1.18.0 \
        scikit-learn==1.9.0 \
        joblib==1.5.3 \
        h5py==3.15.1 \
        pyyaml==6.0.3 \
        pyFAI==2026.5.0 \
        fabio==2025.10.0 \
        pyopencl==2026.1.2 \
    && python -m pip install --no-deps /opt/XRD-preprocessing

COPY Aramis /opt/Aramis
COPY run_training_docker.sh /opt/aramis-bundle/run_training_docker.sh
COPY run_prediction_examples_docker.sh /opt/aramis-bundle/run_prediction_examples_docker.sh
COPY run_prediction_docker.sh /opt/aramis-bundle/run_prediction_docker.sh

RUN python -m pip install --no-deps /opt/Aramis \
    && chmod +x /opt/aramis-bundle/run_training_docker.sh \
    && chmod +x /opt/aramis-bundle/run_prediction_examples_docker.sh \
    && chmod +x /opt/aramis-bundle/run_prediction_docker.sh

WORKDIR /opt/Aramis

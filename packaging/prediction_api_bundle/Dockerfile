FROM python:3.13-slim-bookworm

ARG MODEL_ID

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ARAMIS_MODEL_PATH=/opt/Aramis/models/${MODEL_ID}/model.joblib

WORKDIR /opt/Aramis

RUN apt-get update \
    && apt-get install --yes --no-install-recommends build-essential git ocl-icd-libopencl1 \
    && rm -rf /var/lib/apt/lists/*

COPY . /opt/Aramis

RUN python -m pip install --upgrade pip \
    && python -m pip install pyFAI==2026.5.0 fabio==2025.10.0 pyopencl==2026.1.2 \
    && python -m pip install "/opt/Aramis[api]"

CMD ["uvicorn", "aramis.prediction_api:app", "--host", "0.0.0.0", "--port", "8000"]

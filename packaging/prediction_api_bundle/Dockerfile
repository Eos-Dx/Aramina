FROM python:3.13-slim-bookworm

ARG MODEL_ID

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ARAMINA_MODEL_PATH=/opt/Aramina/models/${MODEL_ID}/model.joblib

WORKDIR /opt/Aramina

RUN apt-get update \
    && apt-get install --yes --no-install-recommends build-essential git ocl-icd-libopencl1 \
    && rm -rf /var/lib/apt/lists/*

COPY . /opt/Aramina

# This runtime deliberately excludes notebook, visualization, and training-only
# dependencies such as marimo. The model and preprocessing contracts are frozen.
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
        fastapi==0.139.2 \
        uvicorn==0.51.0 \
        python-multipart==0.0.32 \
    && python -m pip install --no-deps \
        "git+https://github.com/Eos-Dx/XRD-preprocessing.git@88dcaa277c5a0d4be2ab637bc5827a14bd106bea" \
    && python -m pip install --no-deps /opt/Aramina

CMD ["uvicorn", "aramina.prediction_api:app", "--host", "0.0.0.0", "--port", "8000"]

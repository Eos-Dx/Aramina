ARG BASE_IMAGE=eosdx/aramis-training:0.2.11-beta-arm64
FROM ${BASE_IMAGE}

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /opt/aramis-demo

RUN python -m pip install --upgrade pip \
    && python -m pip install reportlab==4.4.3 streamlit==1.47.1

COPY demo/platform /opt/aramis-demo

CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]

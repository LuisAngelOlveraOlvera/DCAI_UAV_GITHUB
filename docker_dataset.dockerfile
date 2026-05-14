# syntax=docker/dockerfile:1

# ============================================================
# DATASET_TESIS Docker image
# Python: 3.12.7
# Targets:
#   final-cpu -> CPU-only PyTorch
#   final-gpu -> CUDA 12.1 PyTorch wheels, for use with --gpus all
# ============================================================

ARG PYTHON_VERSION=3.12.7

FROM python:${PYTHON_VERSION}-slim-bookworm AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    YOLO_CONFIG_DIR=/tmp/Ultralytics

WORKDIR /dataset

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

COPY requirements.txt /tmp/requirements.txt

# Keep PyTorch outside requirements.txt so CPU/GPU targets can install the
# correct wheel set while sharing the same project dependencies.

FROM base AS final-cpu

RUN python -m pip install \
    --trusted-host download.pytorch.org \
    --trusted-host download-r2.pytorch.org \
    --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1


RUN python -m pip install -r /tmp/requirements.txt

ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g ${GROUP_ID} appuser \
    && useradd -m -u ${USER_ID} -g appuser appuser \
    && mkdir -p /tmp/Ultralytics \
    && chown -R appuser:appuser /dataset /tmp/Ultralytics

COPY --chown=appuser:appuser . /dataset

USER appuser

RUN python -c "import sys, torch; print(sys.version); print(f'PyTorch: {torch.__version__}; CUDA available: {torch.cuda.is_available()}')"

CMD ["python", "scripts/01_check_db_status.py"]

FROM base AS final-gpu

RUN python -m pip install \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

RUN python -m pip install -r /tmp/requirements.txt

ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g ${GROUP_ID} appuser \
    && useradd -m -u ${USER_ID} -g appuser appuser \
    && mkdir -p /tmp/Ultralytics \
    && chown -R appuser:appuser /dataset /tmp/Ultralytics

COPY --chown=appuser:appuser . /dataset

USER appuser

RUN python -c "import sys, torch; print(sys.version); print(f'PyTorch: {torch.__version__}; CUDA wheel: {torch.version.cuda}; CUDA available at build: {torch.cuda.is_available()}')"

CMD ["python", "scripts/01_check_db_status.py"]

# Default target: CPU, because it builds on the widest set of machines.
FROM final-cpu

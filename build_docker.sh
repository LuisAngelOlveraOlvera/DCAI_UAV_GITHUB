#!/usr/bin/env bash
set -euo pipefail

# Build helper for Linux/macOS/WSL.
# Usage: ./build_docker.sh [gpu|cpu|auto]

MODE="${1:-auto}"
IMAGE_NAME="validador-imagen"
PYTHON_VERSION="3.12.7"

echo "=================================="
echo " BUILD DOCKER DATASET_TESIS"
echo "=================================="

detect_gpu() {
    if command -v nvidia-smi > /dev/null 2>&1 && nvidia-smi > /dev/null 2>&1; then
        echo "GPU NVIDIA detectada"
        nvidia-smi --query-gpu=name --format=csv,noheader | head -1
        return 0
    fi

    echo "No se detecto GPU compatible"
    return 1
}

if [ "$MODE" = "auto" ]; then
    echo ""
    echo "Detectando hardware..."
    if detect_gpu; then
        MODE="gpu"
    else
        MODE="cpu"
    fi
fi

if [ "$MODE" != "gpu" ] && [ "$MODE" != "cpu" ]; then
    echo "Modo no valido: $MODE"
    echo "Uso: $0 [gpu|cpu|auto]"
    exit 1
fi

echo ""
echo "Modo seleccionado: $MODE"
echo "Python: $PYTHON_VERSION"
echo ""

if ! docker info > /dev/null 2>&1; then
    echo "Docker no esta disponible. Inicia Docker Desktop/daemon y vuelve a ejecutar este script."
    exit 1
fi

echo "Construyendo imagen ${MODE}..."
docker build \
    -f docker_dataset.dockerfile \
    --build-arg PYTHON_VERSION="${PYTHON_VERSION}" \
    --build-arg USER_ID="$(id -u)" \
    --build-arg GROUP_ID="$(id -g)" \
    --target "final-${MODE}" \
    -t "${IMAGE_NAME}:${MODE}" \
    -t "${IMAGE_NAME}:latest" .

echo ""
echo "Imagen construida:"
echo "  ${IMAGE_NAME}:${MODE}"
echo "  ${IMAGE_NAME}:latest"
echo ""

if [ "$MODE" = "gpu" ]; then
    echo "Ejemplo:"
    echo "  docker run --gpus all --rm -v \"\$(pwd)\":/dataset ${IMAGE_NAME} python scripts/16_evaluar_coco_persona.py"
else
    echo "Ejemplo:"
    echo "  docker run --rm -v \"\$(pwd)\":/dataset ${IMAGE_NAME} python scripts/01_check_db_status.py"
fi
echo "Preparar dataset Kaggle:"
echo "  docker run --rm -v \"\$(pwd)\":/dataset ${IMAGE_NAME} python scripts/00_dataset_setup.py"

echo ""
echo "=================================="
echo " BUILD COMPLETADO"
echo "=================================="

# Build helper for Windows PowerShell.
# Usage: .\build_docker.ps1 [gpu|cpu|auto]

param(
    [ValidateSet("gpu", "cpu", "auto")]
    [string]$Mode = "auto"
)

$IMAGE_NAME = "validador-imagen"
$PYTHON_VERSION = "3.12.7"

Write-Host "==================================" -ForegroundColor Cyan
Write-Host " BUILD DOCKER DATASET_TESIS" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan

function Test-GPU {
    try {
        $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
        if ($nvidiaSmi) {
            $result = nvidia-smi --query-gpu=name --format=csv,noheader 2>&1
            if ($LASTEXITCODE -eq 0) {
                Write-Host "GPU NVIDIA detectada" -ForegroundColor Green
                Write-Host "  $($result | Select-Object -First 1)" -ForegroundColor Gray
                return $true
            }
        }
    }
    catch {
        # Ignore GPU probing errors and fall back to CPU.
    }

    Write-Host "No se detecto GPU compatible" -ForegroundColor Yellow
    return $false
}

if ($Mode -eq "auto") {
    Write-Host ""
    Write-Host "Detectando hardware..." -ForegroundColor Cyan
    if (Test-GPU) {
        $Mode = "gpu"
    }
    else {
        $Mode = "cpu"
    }
}

Write-Host ""
Write-Host "Modo seleccionado: $Mode" -ForegroundColor White
Write-Host "Python: $PYTHON_VERSION" -ForegroundColor White
Write-Host ""

docker info *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Docker no esta disponible. Abre Docker Desktop y vuelve a ejecutar este script." -ForegroundColor Red
    exit 1
}

Write-Host "Construyendo imagen $Mode..." -ForegroundColor Cyan
docker build `
    -f docker_dataset.dockerfile `
    --build-arg PYTHON_VERSION=$PYTHON_VERSION `
    --build-arg USER_ID=1000 `
    --build-arg GROUP_ID=1000 `
    --target "final-$Mode" `
    -t "${IMAGE_NAME}:$Mode" `
    -t "${IMAGE_NAME}:latest" .

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "Imagen construida:" -ForegroundColor Green
Write-Host "  ${IMAGE_NAME}:$Mode" -ForegroundColor Gray
Write-Host "  ${IMAGE_NAME}:latest" -ForegroundColor Gray
Write-Host ""

if ($Mode -eq "gpu") {
    Write-Host "Ejemplo:" -ForegroundColor Cyan
    Write-Host "  docker run --gpus all --rm -v `"`${PWD}:/dataset`" $IMAGE_NAME python scripts/16_evaluar_coco_persona.py" -ForegroundColor Gray
}
else {
    Write-Host "Ejemplo:" -ForegroundColor Cyan
    Write-Host "  docker run --rm -v `"`${PWD}:/dataset`" $IMAGE_NAME python scripts/01_check_db_status.py" -ForegroundColor Gray
}
Write-Host "Preparar dataset Kaggle:" -ForegroundColor Cyan
Write-Host "  docker run --rm -v `"`${PWD}:/dataset`" $IMAGE_NAME python scripts/00_dataset_setup.py" -ForegroundColor Gray

Write-Host ""
Write-Host "==================================" -ForegroundColor Cyan
Write-Host " BUILD COMPLETADO" -ForegroundColor Green
Write-Host "==================================" -ForegroundColor Cyan

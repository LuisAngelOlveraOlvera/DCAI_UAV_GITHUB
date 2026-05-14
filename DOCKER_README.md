# Guía de Uso Docker - DATASET_TESIS

Documentación completa para trabajar con Docker en el proyecto de validación DCAI.

---

## Tabla de Contenidos

1. [Instalación](#-instalación)
2. [Build Inteligente GPU/CPU](#-build-inteligente-gpucpu)
3. [Preparacion del Dataset Kaggle](#preparacion-del-dataset-kaggle)
4. [Ciclo de Trabajo](#-ciclo-de-trabajo)
5. [Comandos por Script](#-comandos-por-script)
6. [Depuración](#-depuración)
7. [Troubleshooting](#-troubleshooting)

---

## Instalación

### Linux/Ubuntu

```bash
# 1. Actualizar e instalar dependencias
sudo apt-get update
sudo apt-get install ca-certificates curl gnupg

# 2. Añadir llave oficial de Docker
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# 3. Configurar repositorio
echo \
  "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# 4. Instalar Docker Engine
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# 5. Dar permisos a tu usuario (evita usar 'sudo' siempre)
sudo usermod -aG docker $USER

#  IMPORTANTE: Cierra sesión y vuelve a entrar para que los permisos se apliquen

### ELIMINAR CHACÉ, MEMPROA E IMAGEN 
docker system prune -a --volumes BORRA TODA LA IMAGEN Y VOLEMN
docker builder prune -a -- SIN BORRAR LA IMAGNE

# Borra solo una imagen específica
docker rmi nombre-de-la-imagen

# Ejemplo para tu caso (si quisieras borrarla):
docker rmi validador-imagen

Contenedores e Historial: Podrías tener contenedores detenidos o volúmenes de datos que sí ocupan espacio en el disco. Para ver todo el "ruido" acumulado, usa: docker system df


```

### Windows 10/11

1. **Descargar**: [Docker Desktop para Windows](https://www.docker.com/products/docker-desktop/)
2. **Instalar**: Ejecutar el instalador
3. **Habilitar WSL 2**: Durante instalación, aceptar usar WSL 2 (más rápido)
4. **Reiniciar**: Windows pedirá reiniciar para activar virtualización
5. **Verificar**: Abrir Docker Desktop, debe aparecer ballena verde en barra de tareas

### GPU Support (Opcional - NVIDIA)

**Linux:**
```bash
# 1. Configurar repositorio y GPG key
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 2. Actualizar e instalar
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# 3. Configurar Docker runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 4. Verificar
nvidia-smi
docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

**Windows:**
- Docker Desktop WSL 2 soporta GPUs automáticamente si tienes drivers NVIDIA actualizados

---

## Build Inteligente GPU/CPU

### Primera Vez - Build Automático

**Linux/Mac:**
```bash
cd ~/Documents/TESIS/DATASET_TESIS
chmod +x build_docker.sh
./build_docker.sh auto
```

**Windows (PowerShell como Administrador):**
```powershell
cd C:\Users\TuUsuario\Documents\TESIS\DATASET_TESIS
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
.\build_docker.ps1 auto
```

El script detectará automáticamente si tienes GPU y construirá la imagen óptima.

### Build Manual (Avanzado)

```bash
# Solo GPU (CUDA 12.1, Python 3.12.7)
./build_docker.sh gpu
docker build -f docker_dataset.dockerfile --build-arg PYTHON_VERSION=3.12.7 --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) --target final-gpu -t validador-imagen:gpu .

# Solo CPU (más ligera)
./build_docker.sh cpu
docker build -f docker_dataset.dockerfile --build-arg PYTHON_VERSION=3.12.7 --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) --target final-cpu -t validador-imagen:cpu .
```

### Reconstruir (si cambias Dockerfile)

```bash
# Con caché (rápido)
./build_docker.sh auto

# Sin caché (forzar todo)
docker build --no-cache -f docker_dataset.dockerfile --build-arg PYTHON_VERSION=3.12.7 --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -t validador-imagen .
```

---

## Preparacion del Dataset Kaggle

El repositorio incluye `scripts/00_dataset_setup.py`. La imagen Docker instala las
dependencias desde `requirements.txt`, incluyendo `kagglehub`, `requests` y
`urllib3`, por lo que el setup puede ejecutarse localmente o dentro del
contenedor.

### Local

```bash
python scripts/00_dataset_setup.py
```

### Docker Linux/Mac

```bash
docker run --rm -v "$(pwd)":/dataset validador-imagen python scripts/00_dataset_setup.py
```

### Docker Windows PowerShell

```powershell
docker run --rm -v "${PWD}:/dataset" validador-imagen python scripts/00_dataset_setup.py
```

Si Kaggle solicita credenciales, coloca `kaggle.json` en `~/.kaggle` en local o
monta esa carpeta en el contenedor. Si tu red requiere desactivar verificacion
SSL, agrega `-e KAGGLE_INSECURE_SSL=1`.

Estructura relativa esperada:

```text
DATASET_KAGGLE\evaluation_videos\VIDEO_1
DATASET_KAGGLE\evaluation_videos\VIDEO_2

DATASET_KAGGLE\EVALUATION\COCO_TEST\images
DATASET_KAGGLE\EVALUATION\COCO_TEST\labels
DATASET_KAGGLE\EVALUATION\IRINA\images
DATASET_KAGGLE\EVALUATION\IRINA\labels
DATASET_KAGGLE\EVALUATION\MANIPAL_UAV\images
DATASET_KAGGLE\EVALUATION\MANIPAL_UAV\labels
DATASET_KAGGLE\EVALUATION\NTUT\images
DATASET_KAGGLE\EVALUATION\NTUT\labels
DATASET_KAGGLE\EVALUATION\UAQ_MSUAV_TEST\images
DATASET_KAGGLE\EVALUATION\UAQ_MSUAV_TEST\labels
DATASET_KAGGLE\EVALUATION\VISDRONE\images
DATASET_KAGGLE\EVALUATION\VISDRONE\labels

DATASET_KAGGLE\PASCAL_VOC_UAQ_MSUAV\images
DATASET_KAGGLE\PASCAL_VOC_UAQ_MSUAV\labels
```

Esta estructura coincide con `README.md`: las primeras etapas del repo leen
`DATASET_KAGGLE/PASCAL_VOC_UAQ_MSUAV/images` y `labels`, y las evaluaciones
externas leen `DATASET_KAGGLE/EVALUATION`.

---

## Ciclo de Trabajo

### Flujo Típico

```
1. Construir la imagen con `./build_docker.sh auto` o `.\build_docker.ps1 auto`
2. Preparar `DATASET_KAGGLE` con `python scripts/00_dataset_setup.py`
3. Ejecutar `scripts/01_check_db_status.py`
4. Correr el pipeline o el script Docker correspondiente
5. Revisar resultados en `exports/`, `logs/` y `runs/`
```

### Verificación Inicial (SIEMPRE PRIMERO)

```bash
# Linux/Mac
docker run --rm -v "$(pwd)":/dataset validador-imagen python scripts/01_check_db_status.py

# Windows PowerShell
docker run --rm -v "${PWD}:/dataset" validador-imagen python scripts/01_check_db_status.py
```

 **IMPORTANTE**: Este script inspecciona el estado de `analysis_metadata.sqlite` y `dataset_master.sqlite` antes de correr el pipeline.

---

## Flujo recomendado

### Core pipeline

0. Preparar estructura Kaggle si `DATASET_KAGGLE/PASCAL_VOC_UAQ_MSUAV` no existe:
```bash
docker run --rm -v "$(pwd)":/dataset validador-imagen python scripts/00_dataset_setup.py
```

1. Chequeo rapido del estado de bases:
```bash
docker run --rm -v "$(pwd)":/dataset validador-imagen python scripts/01_check_db_status.py
```

2. Ingesta analitica:
```bash
docker run --rm -v "$(pwd)":/dataset validador-imagen python scripts/02_analysis_etl.py
Salida: analysis_metadata.sqlite
IMPORTANTE: SE TIENE QUE VOLVER A EJECUTAR TERMINANDO 02_analysis_etl.py
docker run --rm -v "$(pwd)":/dataset validador-imagen python scripts/01_check_db_status.py
```

3. Reporte de duplicados Hamming por fuente (opcional):
```bash
docker run --rm -v "$(pwd)":/dataset validador-imagen python scripts/03_hamming_source_report.py
Salida: hamming_source_report.csv
```

4. ETL maestro (dataset fisico y anotaciones):
```bash
docker run --rm -v "$(pwd)":/dataset validador-imagen python scripts/04_etl_ingestion.py
IMPORTANTE: SE TIENE QUE VOLVER A EJECUTAR TERMINANDO 02_analysis_etl.py
docker run --rm -v "$(pwd)":/dataset validador-imagen python scripts/01_check_db_status.py
```

5. Validacion de integridad del dataset maestro:
```bash
docker run --rm -v "$(pwd)":/dataset validador-imagen python scripts/05_validar_integridad.py
```

6. Exportacion multi-formato de anotaciones (opcional):
```bash
docker run --rm -v "$(pwd)":/dataset validador-imagen python scripts/06_exportar_anotacion.py
Salida: Carpetas con diferentes formatos de frameworks
```

7. Juez one-pass (inferencia + scoring + QA semantico):
```bash
docker run --gpus all --rm -v "$(pwd)":/dataset validador-imagen python scripts/07_judge_scoring.py
Salida: dataset_master.sqlite
```

8. Sync QA (sin reinferencia, copia `label_issue` a `dataset_master.sqlite`):
```bash
docker run --rm -v "$(pwd)":/dataset validador-imagen python scripts/08_qa_audit.py
Salida: audit_details.csv
```

9. Inspeccion visual de bad labels / poor alignment (opcional):
```bash
docker run --gpus all --rm -v "$(pwd)":/dataset validador-imagen python scripts/09_analysis_bad_labels.py
Salida: Reporte interactivo de defectos bad labels, dedup_phash, poor_alignment, weak
```

10. Validacion humana del auditor (opcional, por fases):
```bash
docker run --rm -v "$(pwd)":/dataset validador-imagen python scripts/10_human_validation_audit.py prepare
docker run --rm -v "$(pwd)":/dataset validador-imagen python scripts/10_human_validation_audit.py review --annotator-id A1
docker run --rm -v "$(pwd)":/dataset validador-imagen python scripts/10_human_validation_audit.py score --expected-annotators 3
salida: \human_validation_consensus.csv, \human_validation_pairwise_kappa.csv, \human_validation_metrics_summary.csv, \human_validation_metrics_per_class.csv, \human_validation_metrics_breakdown.csv
```

11. Simulacion DoE (tabla de impacto) a diferentes niveles de Phash:
```bash
docker run --rm -v "$(pwd)":/dataset validador-imagen python scripts/11_scenario_simulator.py
Salida: reporte_analisis_escenarios.csv
```

12. Reporte analitico de base / QA / Judge (opcional):
```bash
docker run --rm -v "$(pwd)":/dataset validador-imagen python scripts/12_db_analysis_report.py
Salida: db_analysis_report.csv, \exports\db_threshold_breakdown.csv
```

13. Generacion fisica de datasets DoE:
```bash
docker run --rm -v "$(pwd)":/dataset validador-imagen python scripts/13_dataset_generation.py
```

14. Validacion fisica de escenarios:
```bash
docker run --rm -v "$(pwd)":/dataset validador-imagen python scripts/14_validation_report.py
```

15. Entrenamiento:
```bash
docker run --gpus all --rm -v "$(pwd)":/dataset validador-imagen python scripts/15_training.py
Salida - ejecuta los entrenamientos, es un menu interactivo
```

16. Evaluacion:
```bash
docker run --gpus all --rm -v "$(pwd)":/dataset validador-imagen python scripts/16_evaluar_coco_persona.py
```

### Seleccion R0-R3 y confirmacion R4

17. Seleccion reducida R0-R3:
```bash
docker run --rm -v "$(pwd)":/dataset validador-imagen python scripts/17_doe_pipeline_reduced.py
```

18. Entrenamiento confirmatorio multi-seed:
```bash
docker run --gpus all --rm -v "$(pwd)":/dataset validador-imagen python scripts/18_train_seed_confirm.py
IMPORTANTE: LOS R4_FINALISTS DE LA LINEA 40 - 48 SE TIENEN QUE COLOCAR DE FORMA MANUAL BASADO EN LOS RESULTAODS DE 17_doe_pipeline_reduced.py
```

19. Seleccion final R4:
```bash
docker run --gpus all --rm -v "$(pwd)":/dataset validador-imagen python scripts/19_doe_r4_final.py
IMPORTANTE: LOS R4_FINALISTS DE LA LINEA 48 - 54 SE TIENEN QUE COLOCAR DE FORMA MANUAL BASADO EN LOS RESULTAODS DE 17_doe_pipeline_reduced.py
```

20. Graficas de momentos por seed:
```bash
docker run --rm -v "$(pwd)":/dataset validador-imagen python scripts/20_plot_r4_seed_moments.py
```

21. Estadistica confirmatoria R4:
```bash
docker run --rm -v "$(pwd)":/dataset validador-imagen python scripts/21_r4_confirmatory_stats.py
```

22. Tukey HSD + Shapiro-Wilk:
```bash
docker run --rm -v "$(pwd)":/dataset validador-imagen python scripts/22_r4_tukey_shapiro.py
```

23. Evaluacion por distancia:
```bash
docker run --rm -v "$(pwd)":/dataset validador-imagen python scripts/23_evaluar_coco_distancia.py
```

24. Analisis de degradacion por distancia:
```bash
docker run --rm -v "$(pwd)":/dataset validador-imagen python scripts/24_distance_degradation_r4.py
```

## Validacion humana del auditor

Flujo recomendado para validar el modelo juez con anotadores humanos:

1. Preparar la muestra humana y exportar plantillas:
```bash
docker run --rm -v "$(pwd)":/dataset validador-imagen python scripts/10_human_validation_audit.py prepare
```

Comportamiento por defecto de `prepare`:

- incluye todos los errores detectados por el juez (`missing_label`, `bad_label`, `poor_alignment`)
- agrega `250` casos `ok` estratificados
- deja una muestra lista para validacion humana con ~`586` imagenes en el estado actual de la BD

Si quieres volver al muestreo mixto anterior:
```bash
docker run --rm -v "$(pwd)":/dataset validador-imagen python scripts/10_human_validation_audit.py prepare --strategy mixed_stratified --sample-size 600
```

2. Revisar imagen por imagen como anotador A1:
```bash
docker run --rm -v "$(pwd)":/dataset validador-imagen python scripts/10_human_validation_audit.py review --annotator-id A1
```

3. Revisar imagen por imagen como anotador A2:
```bash
docker run --rm -v "$(pwd)":/dataset validador-imagen python scripts/10_human_validation_audit.py review --annotator-id A2
```

4. Revisar imagen por imagen como anotador A3:
```bash
docker run --rm -v "$(pwd)":/dataset validador-imagen python scripts/10_human_validation_audit.py review --annotator-id A3
```

Teclas del modo interactivo:

- `1`: `ok`
- `2`: `missing_label`
- `3`: `bad_label`
- `4`: `poor_alignment`
- `5`: `ambiguous`
- `s` o `espacio`: saltar muestra
- `q` o `Esc`: guardar y salir

5. Calcular consenso humano y estadisticas finales:
```bash
docker run --rm -v "$(pwd)":/dataset validador-imagen python scripts/10_human_validation_audit.py score --expected-annotators 3
```

Salidas principales:

- `exports/human_validation_audit/human_validation_sample_master.csv`
- `exports/human_validation_audit/human_validation_annotator_A*.csv`
- `exports/human_validation_audit/human_validation_consensus.csv`
- `exports/human_validation_audit/human_validation_metrics_summary.csv`
- `exports/human_validation_audit/human_validation_pairwise_kappa.csv`

## Entrenamiento fijo (protocolo)

- Modelo: `yolo11n.pt`
- `epochs=100`
- `batch=8`
- `imgsz=640`
- `workers=4`
- `seed=0` (recomendado para reproducibilidad)
- `patience=15` (early stopping)

## Notas

- `13_dataset_generation.py` usa deduplicacion por similitud (Hamming), no por hash exacto.
- `13_dataset_generation.py` excluye ruido semantico (`missing_label`, `bad_label`, `poor_alignment`) antes de construir datasets.
- `07_judge_scoring.py` ejecuta una sola pasada del juez y guarda `judge_score` + `label_issue` en `analysis_metadata.sqlite`.
- `08_qa_audit.py` ya no infiere; solo sincroniza resultados QA a `dataset_master.sqlite` y valida `% unknown`.
- `11_scenario_simulator.py` y `12_db_analysis_report.py` usan `analysis_images.label_issue` cuando existe.
- `12_db_analysis_report.py` calcula `Injected` sobre `Train Base` (igual que `13_dataset_generation.py`), no sobre el total limpio.
- `config_analysis.py` mantiene compatibilidad con scripts legacy via `use_pascal` y `use_propio`.

## Como ejecutar training seed

Eso corre con defaults:

```text
--model n
--epochs 100
--seeds 7 42 123 999
```

Ahora, combinaciones utiles:

```bash
docker run --gpus all --rm -v "$(pwd)":/dataset validador-imagen python .\scripts\18_train_seed_confirm.py --dry_run
```
Muestra el plan sin entrenar.

```bash
docker run --gpus all --rm -v "$(pwd)":/dataset validador-imagen python .\scripts\18_train_seed_confirm.py --model n --epochs 100 --seeds 7 42 123 999
```
Ejecucion completa explicita.

```bash
docker run --gpus all --rm -v "$(pwd)":/dataset validador-imagen python .\scripts\18_train_seed_confirm.py --model n --epochs 50 --seeds 42
```
Una sola seed, mas rapido.

```bash
docker run --gpus all --rm -v "$(pwd)":/dataset validador-imagen python .\scripts\18_train_seed_confirm.py --model n --epochs 50 --seeds 42 123
```
Confirmacion ligera con dos seeds.

```bash
docker run --gpus all --rm -v "$(pwd)":/dataset validador-imagen python .\scripts\18_train_seed_confirm.py --model s --epochs 100 --seeds 7 42 123 999
```
Version con yolo11s.pt.

```bash
docker run --gpus all --rm -v "$(pwd)":/dataset validador-imagen python .\scripts\18_train_seed_confirm.py --model m --epochs 100 --seeds 42 123
```
Mas pesado; el script baja batch a 4 si detecta m.

```bash
docker run --gpus all --rm -v "$(pwd)":/dataset validador-imagen python .\scripts\18_train_seed_confirm.py --only_missing
```
Solo corre los run_dir que todavia no existan.

```bash
docker run --gpus all --rm -v "$(pwd)":/dataset validador-imagen python .\scripts\18_train_seed_confirm.py --only_missing --model n --epochs 100 --seeds 7 42 123 999
```
Muy util si se interrumpio una corrida.

```bash
docker run --gpus all --rm -v "$(pwd)":/dataset validador-imagen python .\scripts\18_train_seed_confirm.py --dry_run --only_missing --model n --epochs 100 --seeds 42 123
```

---

## Comandos por Script

### Linux/Mac (Bash/ZSH)

| Paso | Script | Comando |
|------|--------|---------|
| `00` | `00_dataset_setup.py` | `docker run --rm -v "$(pwd)":/dataset validador-imagen python scripts/00_dataset_setup.py` |
| `01` | `01_check_db_status.py` | `docker run --rm -v "$(pwd)":/dataset validador-imagen python scripts/01_check_db_status.py` |
| `02` | `02_analysis_etl.py` | `docker run --rm -v "$(pwd)":/dataset validador-imagen python scripts/02_analysis_etl.py` |
| `03` | `03_hamming_source_report.py` | `docker run --rm -v "$(pwd)":/dataset validador-imagen python scripts/03_hamming_source_report.py` |
| `04` | `04_etl_ingestion.py` | `docker run --rm -v "$(pwd)":/dataset validador-imagen python scripts/04_etl_ingestion.py` |
| `05` | `05_validar_integridad.py` | `docker run --rm -v "$(pwd)":/dataset validador-imagen python scripts/05_validar_integridad.py` |
| `06` | `06_exportar_anotacion.py` | `docker run --rm -v "$(pwd)":/dataset validador-imagen python scripts/06_exportar_anotacion.py` |
| `07` | `07_judge_scoring.py` | `docker run --gpus all --rm -v "$(pwd)":/dataset validador-imagen python scripts/07_judge_scoring.py` |
| `08` | `08_qa_audit.py` | `docker run --rm -v "$(pwd)":/dataset validador-imagen python scripts/08_qa_audit.py` |
| `09` | `09_analysis_bad_labels.py` | `docker run --gpus all --rm -v "$(pwd)":/dataset validador-imagen python scripts/09_analysis_bad_labels.py` |
| `10` | `10_human_validation_audit.py` | `docker run --rm -v "$(pwd)":/dataset validador-imagen python scripts/10_human_validation_audit.py prepare` |
| `11` | `11_scenario_simulator.py` | `docker run --rm -v "$(pwd)":/dataset validador-imagen python scripts/11_scenario_simulator.py` |
| `12` | `12_db_analysis_report.py` | `docker run --rm -v "$(pwd)":/dataset validador-imagen python scripts/12_db_analysis_report.py` |
| `13` | `13_dataset_generation.py` | `docker run --rm -v "$(pwd)":/dataset validador-imagen python scripts/13_dataset_generation.py` |
| `14` | `14_validation_report.py` | `docker run --rm -v "$(pwd)":/dataset validador-imagen python scripts/14_validation_report.py` |
| `15` | `15_training.py` | `docker run --gpus all --rm -v "$(pwd)":/dataset validador-imagen python scripts/15_training.py --epochs 100` |
| `16` | `16_evaluar_coco_persona.py` | `docker run --gpus all --rm -v "$(pwd)":/dataset validador-imagen python scripts/16_evaluar_coco_persona.py` |
| `17` | `17_doe_pipeline_reduced.py` | `docker run --rm -v "$(pwd)":/dataset validador-imagen python scripts/17_doe_pipeline_reduced.py` |
| `18` | `18_train_seed_confirm.py` | `docker run --gpus all --rm -v "$(pwd)":/dataset validador-imagen python scripts/18_train_seed_confirm.py` |
| `19` | `19_doe_r4_final.py` | `docker run --gpus all --rm -v "$(pwd)":/dataset validador-imagen python scripts/19_doe_r4_final.py` |
| `20` | `20_plot_r4_seed_moments.py` | `docker run --rm -v "$(pwd)":/dataset validador-imagen python scripts/20_plot_r4_seed_moments.py` |
| `21` | `21_r4_confirmatory_stats.py` | `docker run --rm -v "$(pwd)":/dataset validador-imagen python scripts/21_r4_confirmatory_stats.py` |
| `22` | `22_r4_tukey_shapiro.py` | `docker run --rm -v "$(pwd)":/dataset validador-imagen python scripts/22_r4_tukey_shapiro.py` |
| `23` | `23_evaluar_coco_distancia.py` | `docker run --rm -v "$(pwd)":/dataset validador-imagen python scripts/23_evaluar_coco_distancia.py` |
| `24` | `24_distance_degradation_r4.py` | `docker run --rm -v "$(pwd)":/dataset validador-imagen python scripts/24_distance_degradation_r4.py` |

### Windows (PowerShell)

| Paso | Script | Comando |
|------|--------|---------|
| `00` | `00_dataset_setup.py` | `docker run --rm -v "${PWD}:/dataset" validador-imagen python scripts/00_dataset_setup.py` |
| `01` | `01_check_db_status.py` | `docker run --rm -v "${PWD}:/dataset" validador-imagen python scripts/01_check_db_status.py` |
| `02` | `02_analysis_etl.py` | `docker run --rm -v "${PWD}:/dataset" validador-imagen python scripts/02_analysis_etl.py` |
| `03` | `03_hamming_source_report.py` | `docker run --rm -v "${PWD}:/dataset" validador-imagen python scripts/03_hamming_source_report.py` |
| `04` | `04_etl_ingestion.py` | `docker run --rm -v "${PWD}:/dataset" validador-imagen python scripts/04_etl_ingestion.py` |
| `05` | `05_validar_integridad.py` | `docker run --rm -v "${PWD}:/dataset" validador-imagen python scripts/05_validar_integridad.py` |
| `06` | `06_exportar_anotacion.py` | `docker run --rm -v "${PWD}:/dataset" validador-imagen python scripts/06_exportar_anotacion.py` |
| `07` | `07_judge_scoring.py` | `docker run --gpus all --rm -v "${PWD}:/dataset" validador-imagen python scripts/07_judge_scoring.py` |
| `08` | `08_qa_audit.py` | `docker run --rm -v "${PWD}:/dataset" validador-imagen python scripts/08_qa_audit.py` |
| `09` | `09_analysis_bad_labels.py` | `docker run --gpus all --rm -v "${PWD}:/dataset" validador-imagen python scripts/09_analysis_bad_labels.py` |
| `10` | `10_human_validation_audit.py` | `docker run --rm -v "${PWD}:/dataset" validador-imagen python scripts/10_human_validation_audit.py prepare` |
| `11` | `11_scenario_simulator.py` | `docker run --rm -v "${PWD}:/dataset" validador-imagen python scripts/11_scenario_simulator.py` |
| `12` | `12_db_analysis_report.py` | `docker run --rm -v "${PWD}:/dataset" validador-imagen python scripts/12_db_analysis_report.py` |
| `13` | `13_dataset_generation.py` | `docker run --rm -v "${PWD}:/dataset" validador-imagen python scripts/13_dataset_generation.py` |
| `14` | `14_validation_report.py` | `docker run --rm -v "${PWD}:/dataset" validador-imagen python scripts/14_validation_report.py` |
| `15` | `15_training.py` | `docker run --gpus all --rm -v "${PWD}:/dataset" validador-imagen python scripts/15_training.py --epochs 100` |
| `16` | `16_evaluar_coco_persona.py` | `docker run --gpus all --rm -v "${PWD}:/dataset" validador-imagen python scripts/16_evaluar_coco_persona.py` |
| `17` | `17_doe_pipeline_reduced.py` | `docker run --rm -v "${PWD}:/dataset" validador-imagen python scripts/17_doe_pipeline_reduced.py` |
| `18` | `18_train_seed_confirm.py` | `docker run --gpus all --rm -v "${PWD}:/dataset" validador-imagen python scripts/18_train_seed_confirm.py` |
| `19` | `19_doe_r4_final.py` | `docker run --gpus all --rm -v "${PWD}:/dataset" validador-imagen python scripts/19_doe_r4_final.py` |
| `20` | `20_plot_r4_seed_moments.py` | `docker run --rm -v "${PWD}:/dataset" validador-imagen python scripts/20_plot_r4_seed_moments.py` |
| `21` | `21_r4_confirmatory_stats.py` | `docker run --rm -v "${PWD}:/dataset" validador-imagen python scripts/21_r4_confirmatory_stats.py` |
| `22` | `22_r4_tukey_shapiro.py` | `docker run --rm -v "${PWD}:/dataset" validador-imagen python scripts/22_r4_tukey_shapiro.py` |
| `23` | `23_evaluar_coco_distancia.py` | `docker run --rm -v "${PWD}:/dataset" validador-imagen python scripts/23_evaluar_coco_distancia.py` |
| `24` | `24_distance_degradation_r4.py` | `docker run --rm -v "${PWD}:/dataset" validador-imagen python scripts/24_distance_degradation_r4.py` |

### Notas sobre los Comandos

- **`--rm`**: Borra el contenedor automáticamente al terminar
- **`-v "$(pwd)":/dataset`**: Monta tu carpeta actual dentro del contenedor
- **`--gpus all`**: Usa GPU (solo para entrenamiento/evaluación)
- **Sin GPU**: Omite `--gpus all`, el script detecta CPU automáticamente

---

## Depuración

### Entrar al Contenedor (Explorar)

```bash
# Linux/Mac
docker run -it --rm -v "$(pwd)":/dataset validador-imagen /bin/bash

# Windows PowerShell
docker run -it --rm -v "${PWD}:/dataset" validador-imagen /bin/bash

# Dentro del contenedor puedes:
ls -la                    # Ver archivos
python scripts/01_check_db_status.py
python scripts/04_etl_ingestion.py
exit                      # Salir
```

### Ver Contenedores Activos

```bash
docker ps                 # Ver contenedores corriendo
docker ps -a              # Ver todos (incluso detenidos)
```

### Detener Contenedores

```bash
# Detener uno específico
docker stop <CONTAINER_ID>

# Detener todos
docker stop $(docker ps -q)
```

### Ver Logs

```bash
# Logs de un contenedor específico
docker logs <CONTAINER_ID>

# Seguir logs en tiempo real
docker logs -f <CONTAINER_ID>
```

### Limpiar Sistema

```bash
# Limpiar contenedores detenidos, redes, imágenes huérfanas
docker system prune

# Limpiar TODO (incluye imágenes sin usar)
docker system prune -a

# Limpiar volúmenes huérfanos
docker volume prune
```

### Ver Imágenes

```bash
docker images             # Listar imágenes
docker rmi validador-imagen  # Borrar imagen específica
docker rmi -f validador-imagen  # Forzar borrado
```

---

## Troubleshooting

### 1. Carpetas Protegidas (no puedo borrar exports/logs/runs)

**Problema**: Docker crea carpetas como root y no las puedes editar/borrar.

**Solución Rápida (Linux/Mac)**:
```bash
chmod +x fix_permission.sh
./fix_permission.sh
```

**Solución Manual (Linux/Mac)**:
```bash
sudo chown -R $(id -u):$(id -g) exports logs runs dataset_master.sqlite
sudo chmod -R u+rwX exports logs runs
```

**Solución Windows (PowerShell como Admin)**:
```powershell
icacls exports /grant:r "${env:USERNAME}:(OI)(CI)F" /T
icacls logs /grant:r "${env:USERNAME}:(OI)(CI)F" /T
icacls runs /grant:r "${env:USERNAME}:(OI)(CI)F" /T
```

**Prevención**: Reconstruye con `./build_docker.sh auto` para que use tu USER_ID

### 2. Error: "could not select device driver"

**Causa**: NVIDIA Container Toolkit no instalado.

**Solución**:
```bash
# Instalar toolkit (ver sección de instalación GPU)
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 3. Error: "No such file or directory" al montar volumen

**Causa**: Estás en la carpeta incorrecta.

**Solución**:
```bash
# Verificar dónde estás
pwd

# Debe ser algo como:
# /home/usuario/Documents/TESIS/DATASET_TESIS
# o C:\Users\Usuario\Documents\TESIS\DATASET_TESIS

# Navegar a la carpeta correcta
cd ~/Documents/TESIS/DATASET_TESIS  # Linux/Mac
cd C:\Users\TuUsuario\Documents\TESIS\DATASET_TESIS  # Windows
```

### 4. Docker muy lento en Windows

**Causas Comunes**:
- WSL 2 no activado
- Antivirus bloqueando
- Disco compartido lento

**Soluciones**:
1. Verificar WSL 2: `wsl --status`
2. Excluir carpeta Docker del antivirus
3. Mover proyecto a WSL: `\\wsl$\Ubuntu\home\usuario\`

### 5. "Out of Memory" durante entrenamiento

**Solución**:
```python
# En 15_training.py, reducir batch size
python scripts/15_training.py --batch 4 --imgsz 320

# O usar CPU
python scripts/15_training.py --device cpu --batch 2
```

### 6. Imagen muy grande (>5GB)

**Verificar**:
```bash
docker images validador-imagen
```

**Limpiar capas antiguas**:
```bash
docker builder prune
docker system prune -a
```

### 7. Scripts modificados no se ven en Docker

**Causa**: Dockerfile usa `COPY . /dataset` en tiempo de build, no runtime.

**Solución**: Los scripts se montan con `-v`, así que los cambios se ven inmediatamente. Si aún no funciona:
```bash
# Verificar que el volumen esté bien montado
docker run --rm -v "$(pwd)":/dataset validador-imagen ls -la scripts/
```

---

## Verificación Final

Después de configurar todo, ejecuta esta secuencia para verificar:

```bash
# 1. Build
./build_docker.sh auto

# 2. Verificar estado
docker run --rm -v "$(pwd)":/dataset validador-imagen python scripts/01_check_db_status.py

# 3. Ingesta analitica
docker run --rm -v "$(pwd)":/dataset validador-imagen python scripts/02_analysis_etl.py

# 4. ETL maestro
docker run --rm -v "$(pwd)":/dataset validador-imagen python scripts/04_etl_ingestion.py

# 5. Validar integridad
docker run --rm -v "$(pwd)":/dataset validador-imagen python scripts/05_validar_integridad.py

# Si todo funciona, verás:
# análisis y master DB detectadas
# Logs en logs/
# Carpetas editables por tu usuario
```

---

## Tips Pro

1. **Alias útiles** (añadir a `~/.bashrc` o `~/.zshrc`):
```bash
alias drun='docker run --rm -v "$(pwd)":/dataset validador-imagen'
alias dgpu='docker run --gpus all --rm -v "$(pwd)":/dataset validador-imagen'
alias dsh='docker run -it --rm -v "$(pwd)":/dataset validador-imagen /bin/bash'

# Uso:
drun python scripts/05_validar_integridad.py
dgpu python scripts/15_training.py --epochs 100
dsh  # Entrar al contenedor
```

2. **Hot reload** (para desarrollo rápido):
```bash
# Los cambios en scripts/ se ven inmediatamente
# No necesitas reconstruir la imagen
```

3. **Multi-terminal**:
```bash
# Terminal 1: Entrenar
dgpu python scripts/15_training.py

# Terminal 2: Monitorear
watch -n 1 'ls -lh runs/train/exp/weights/'
```

4. **Backups antes de cambios grandes**:
```bash
tar -czf backup_$(date +%Y%m%d).tar.gz dataset_master.sqlite exports/ logs/
```

---

##  Recursos Adicionales

- [Docker Docs](https://docs.docker.com/)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)
- [Ultralytics YOLO](https://docs.ultralytics.com/)
- [WSL 2 Setup](https://learn.microsoft.com/en-us/windows/wsl/install)

---

## Soporte

Si encuentras problemas:
1. Verifica que estés en la carpeta correcta: `pwd`
2. Verifica que la imagen esté construida: `docker images`
3. Revisa los logs: `docker logs <container_id>`
4. Consulta esta guía en la sección Troubleshooting
5. Reconstruye desde cero: `docker rmi -f validador-imagen && ./build_docker.sh auto`

---

**Última actualización**: Diciembre 2025  
**Versión**: 2.0 (GPU/CPU Inteligente con permisos correctos)

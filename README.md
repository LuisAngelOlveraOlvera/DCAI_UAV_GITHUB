# Tesis DCAI - Deteccion de Personas

Repositorio para validacion de metodologia Data-Centric AI (DCAI) en deteccion de personas con Ultralytics YOLO.

## Estructura

```text
DATASET_TESIS/
|-- DATASET_KAGGLE/
|   |-- PASCAL_VOC_UAQ_MSUAV/
|   |   |-- images/          # Imagenes base para las primeras etapas
|   |   `-- labels/          # Etiquetas YOLO base
|   |-- EVALUATION/
|   |   |-- COCO_TEST/
|   |   |-- IRINA/
|   |   |-- MANIPAL_UAV/
|   |   |-- NTUT/
|   |   |-- UAQ_MSUAV_TEST/
|   |   `-- VISDRONE/
|   `-- evaluation_videos/
|       |-- VIDEO_1/
|       `-- VIDEO_2/
|-- scripts/
|   |-- 00_dataset_setup.py
|   |-- config.py
|   |-- config_analysis.py
|   |-- 01_check_db_status.py
|   |-- 02_analysis_etl.py
|   |-- 04_etl_ingestion.py
|   |-- 07_judge_scoring.py
|   |-- 13_dataset_generation.py
|   |-- 15_training.py
|   |-- 16_evaluar_coco_persona.py
|   `-- 19_doe_r4_final.py
|-- exports/                 # Datasets y reportes generados
|-- logs/                    # Logs de ejecucion
|-- dataset_master.sqlite
`-- analysis_metadata.sqlite
```

## Preparacion del dataset Kaggle

El dataset fuente se descarga desde Kaggle con `scripts/00_dataset_setup.py`:

```bash
python scripts/00_dataset_setup.py
```

El script usa `kagglehub`, `os`, `shutil` y `Path` para descargar el dataset
`luisngeld/person-detection-uav-pascal-voc-uaq-msuav` y normalizarlo dentro de
`DATASET_KAGGLE`. Si tu red requiere desactivar verificacion SSL para Kaggle,
ejecuta con `KAGGLE_INSECURE_SSL=1`.

Rutas relativas esperadas despues de preparar el dataset:

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

`scripts/config.py` toma las imagenes y labels base desde
`DATASET_KAGGLE/PASCAL_VOC_UAQ_MSUAV/`, y los scripts de evaluacion externa usan
`DATASET_KAGGLE/EVALUATION/`.

## Higiene de Git y Docker

El dataset, modelos, videos, logs y resultados de entrenamiento son artefactos locales.
No deben versionarse ni copiarse al contexto de build de Docker. Para eso se usan dos
archivos distintos:

- `.gitignore`: evita que Git agregue datasets, pesos, bases SQLite, logs, exports,
  runs, videos, comprimidos y ambientes virtuales al historial.
- `.dockerignore`: evita que `docker build` envie `.git`, datasets, modelos, logs,
  resultados y caches al daemon de Docker.

Esto es importante porque el repositorio puede parecer muy grande aunque el codigo sea
pequeno. En este proyecto se detecto un caso donde `.git/objects` crecio a ~16.5 GB y
`DATASET_KAGGLE/` ocupaba ~16.3 GB.

Diagnostico rapido en Windows/PowerShell:

```powershell
git count-objects -vH

Get-ChildItem -Force |
  Sort-Object Length -Descending |
  Select-Object Mode, Length, Name

Get-ChildItem -Force .git\objects -Recurse -File |
  Measure-Object Length -Sum
```

Limpieza de objetos Git no alcanzables:

```powershell
git gc --prune=now
git count-objects -vH
```

Si Windows no permite borrar un archivo viejo en `.git/objects/pack`, cierra procesos
que puedan bloquearlo, por ejemplo OneDrive, VS Code, Docker Desktop, Explorer o el
antivirus. Despues repite `git gc --prune=now` o borra manualmente el pack viejo solo
si ya confirmaste que Git quedo funcional.

## Convenciones DoE

- `DEDUP`: pHash con distancia de Hamming `<= 10` (`config.HAMMING_THRESHOLD`).
- `Judge`: auditoria semantica por score con `tau=0.35` (Relax) y `tau=0.65` (Strict).
- `Noise`: smart noise injection (oversampling) en `train` con `{0, 15, 40}%`.
- `TL`: transferencia desde COCO para YOLO11n.
- Control `N1_Z_COCO`: es control de referencia, no genera dataset fisico.

## Escenarios activos (config_analysis.py)

- Controles: `N1_Z_COCO`, `N2_B_Raw_0`
- Hybrid ablacion limpia: `N3_H_Raw_0`, `N4_H_pH_0`, `N5_H_JR_0`, `N6_H_pH_JR_0`, `N7_H_pH_JS_0`
- Hybrid con ruido: `N8_H_pH_JR_15`, `N9_H_pH_JR_40`, `N10_H_pH_JS_40`
- Proprietary: `N11_P_Raw_0`, `N12_P_pH_JR_0`, `N13_P_pH_JR_15`, `N14_P_pH_JR_40`

## Matriz trazable (Ref -> escenario)

- Archivo: `TRACEABILITY_MATRIX_DOE.csv`
- Mapea referencias de paper (`R0-1`..`R3-4`) a escenarios implementados (`N1`..`N14`).
- Incluye campos de control: dominio, deduplicacion pHash, `tau` del Judge, `noise %`, TL y si construye dataset fisico.

## Flujo recomendado

### Paso previo para scripts con ventanas interactivas en Docker

Algunos scripts abren ventanas interactivas con OpenCV/Qt o Matplotlib, por ejemplo
`cv2.imshow()`, `cv2.waitKey()` o `plt.show()`. Docker no tiene pantalla propia, por lo
que antes de ejecutarlos dentro del contenedor hay que redirigir la salida grafica hacia
el sistema host.

Este paso aplica especialmente a `scripts/09_analysis_bad_labels.py`.

En Windows:

- Instalar y abrir XLaunch/VcXsrv antes de ejecutar el contenedor.
- Configurar XLaunch con:
- `Multiple windows`
- `Start no client`
- `Disable access control` activado
- Ejecutar el contenedor con `DISPLAY=host.docker.internal:0.0` y `QT_X11_NO_MITSHM=1`.

```powershell
docker run --rm -it `
  -v "${PWD}:/dataset" `
  -e DISPLAY=host.docker.internal:0.0 `
  -e QT_X11_NO_MITSHM=1 `
  validador-imagen `
  python scripts/09_analysis_bad_labels.py
```

En Linux/Ubuntu:

- Permitir acceso local de Docker al servidor X con `xhost +local:docker`.
- Montar el socket X11 `/tmp/.X11-unix` y pasar `DISPLAY=$DISPLAY` al contenedor.

```bash
xhost +local:docker

docker run --rm -it \
  -v "$(pwd)":/dataset \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY=$DISPLAY \
  validador-imagen python scripts/09_analysis_bad_labels.py
```

### Core pipeline

0. Preparar estructura Kaggle si `DATASET_KAGGLE/PASCAL_VOC_UAQ_MSUAV` no existe:
```bash
python scripts/00_dataset_setup.py
```

1. Chequeo rapido del estado de bases:
```bash
python scripts/01_check_db_status.py
```

2. Ingesta analitica:
```bash
python scripts/02_analysis_etl.py
Salida: analysis_metadata.sqlite
IMPORTANTE: SE TIENE QUE VOLVER A EJECUTAR TERMINANDO 02_analysis_etl.py
python scripts/01_check_db_status.py
```

3. Reporte de duplicados Hamming por fuente (opcional):
```bash
python scripts/03_hamming_source_report.py
Salida: hamming_source_report.csv
```

4. ETL maestro (dataset fisico y anotaciones):
```bash
python scripts/04_etl_ingestion.py
IMPORTANTE: SE TIENE QUE VOLVER A EJECUTAR TERMINANDO 02_analysis_etl.py
python scripts/01_check_db_status.py
```

5. Validacion de integridad del dataset maestro:
```bash
python scripts/05_validar_integridad.py
```

6. Exportacion multi-formato de anotaciones (opcional):
```bash
python scripts/06_exportar_anotacion.py
Salida: Carpetas con diferentes formatos de frameworks
```

7. Juez one-pass (inferencia + scoring + QA semantico):
```bash
python scripts/07_judge_scoring.py
Salida: dataset_master.sqlite
```

8. Sync QA (sin reinferencia, copia `label_issue` a `dataset_master.sqlite`):
```bash
python scripts/08_qa_audit.py
Salida: audit_details.csv
```

9. Inspeccion visual de bad labels / poor alignment (opcional):
```bash
python scripts/09_analysis_bad_labels.py
Salida: Reporte interactivo de defectos bad labels, dedup_phash, poor_alignment, weak
```

10. Validacion humana del auditor (opcional, por fases):
```bash
python scripts/10_human_validation_audit.py prepare
python scripts/10_human_validation_audit.py review --annotator-id A1
python scripts/10_human_validation_audit.py score --expected-annotators 3
salida: \human_validation_consensus.csv, \human_validation_pairwise_kappa.csv, \human_validation_metrics_summary.csv,  \human_validation_metrics_per_class.csv, \human_validation_metrics_breakdown.csv
```

11. Simulacion DoE (tabla de impacto) a diferentes niveles de Phash:
```bash
python scripts/11_scenario_simulator.py
Salida: reporte_analisis_escenarios.csv
```

12. Reporte analitico de base / QA / Judge (opcional):
```bash
python scripts/12_db_analysis_report.py
Salida: db_analysis_report.csv, \exports\db_threshold_breakdown.csv
```

13. Generacion fisica de datasets DoE:
```bash
python scripts/13_dataset_generation.py
```

Para construir solo escenarios especificos:
```bash
python scripts/13_dataset_generation.py --scenarios N2
python scripts/13_dataset_generation.py --scenarios N2_B_Raw_0
python scripts/13_dataset_generation.py --scenarios N2_B_Raw_0 N3_H_Raw_0
```

14. Validacion fisica de escenarios:
```bash
python scripts/14_validation_report.py
```

15. Entrenamiento:
```bash
python scripts/15_training.py --datasets all --model n --epochs 100 --yes
Salida - ejecuta los entrenamientos secuencialmente sin menu interactivo.

Para usar el menu interactivo:

python scripts/15_training.py
```

16. Evaluacion:
```bash
python scripts/16_evaluar_coco_persona.py --datasets all --weights all

python scripts/16_evaluar_coco_persona.py --list-datasets
python scripts/16_evaluar_coco_persona.py --list-weights


Para elegir datasets de evaluacion y pesos por menu:

python scripts/16_evaluar_coco_persona.py

Para evaluar solo algunos datasets con pesos especificos:

python scripts/16_evaluar_coco_persona.py --datasets COCO_TEST VISDRONE --weights N1_YOLO11n N2_B_Raw_0_yolo11n_e100
```

Nota: `16_evaluar_coco_persona.py` genera YAML temporales con rutas absolutas calculadas
en runtime. Esto evita que Ultralytics reinterprete rutas relativas bajo su
`datasets_dir` interno, por ejemplo `/dataset/datasets` dentro de Docker.

### Seleccion R0-R3 y confirmacion R4

17. Seleccion reducida R0-R3:
```bash
python scripts/17_doe_pipeline_reduced.py
```

18. Entrenamiento confirmatorio multi-seed:
```bash
python scripts/18_train_seed_confirm.py --only_missing --model n --epochs 100 --seeds 7 42 123 999
IMPORTANTE: LOS R4_FINALISTS DE LA LÍNEA 40 - 48 SE TIENEN QUE COLOCAR DE FORMA MANUAL BASADO EN LOS RESULTAODS DE 17_doe_pipeline_reduced.py
```

19. Seleccion final R4:
```bash
python scripts/19_doe_r4_final.py

# Listar datasets disponibles para la evaluacion R4
python scripts/19_doe_r4_final.py --list-datasets

# Evaluar solo algunos datasets especificos
python scripts/19_doe_r4_final.py --datasets VISDRONE NTUT COCO_TEST

# Tambien acepta indices del menu o 'all'
python scripts/19_doe_r4_final.py --datasets 1 5 2
python scripts/19_doe_r4_final.py --datasets all

Sin argumentos, si ejecutas desde terminal interactiva, el script muestra un menu para
elegir los datasets donde se evalua R4.
IMPORTANTE: LOS R4_FINALISTS DE LA LÍNEA 48 - 54 SE TIENEN QUE COLOCAR DE FORMA MANUAL BASADO EN LOS RESULTAODS DE 17_doe_pipeline_reduced.py
```

20. Graficas de momentos por seed:
```bash
python scripts/20_plot_r4_seed_moments.py
```

21. Estadistica confirmatoria R4:
```bash
python scripts/21_r4_confirmatory_stats.py
```

22. Tukey HSD + Shapiro-Wilk:
```bash
python scripts/22_r4_tukey_shapiro.py
```

23. Evaluación por distancia:
```bash
python scripts/23_evaluar_coco_distancia.py

# Listar escenarios disponibles
python scripts/23_evaluar_coco_distancia.py --list-scenarios

# Evaluar solo algunos escenarios por alias N#
python scripts/23_evaluar_coco_distancia.py --scenarios N1 N7 N11

# Evaluar por nombre completo, indices o 'all'
python scripts/23_evaluar_coco_distancia.py --scenarios N1_YOLO11n N2_B_Raw_0_yolo11n_e100
python scripts/23_evaluar_coco_distancia.py --scenarios 1 2 7
python scripts/23_evaluar_coco_distancia.py --scenarios all

Sin argumentos, si ejecutas desde terminal interactiva, el script muestra un menu para
elegir los escenarios que entran a la evaluacion por distancia.
```

24. Analisis de degradacion por distancia:
```bash
python scripts/24_distance_degradation_r4.py
```

25. Análisis del database sqlite:
```bash

python scripts/24_distance_degradation_r4.py

# Modo básico (ambas DBs en el mismo directorio)
python scripts/db_explore.py

# Rutas explícitas + exportar HTML y CSV
python scripts/db_explore.py --db ./data/analysis_metadata.sqlite ./data/dataset_master.sqlite --export

# Solo análisis cruzado
python scripts/db_explore.py --cross
```

## Validacion humana del auditor

Flujo recomendado para validar el modelo juez con anotadores humanos:

1. Preparar la muestra humana y exportar plantillas:
```bash
python scripts/10_human_validation_audit.py prepare
```

Comportamiento por defecto de `prepare`:

- incluye todos los errores detectados por el juez (`missing_label`, `bad_label`, `poor_alignment`)
- agrega `250` casos `ok` estratificados
- deja una muestra lista para validacion humana con ~`586` imagenes en el estado actual de la BD

Si quieres volver al muestreo mixto anterior:
```bash
python scripts/10_human_validation_audit.py prepare --strategy mixed_stratified --sample-size 600
```

Debug rapido con 10 imagenes:

```bash
python scripts/10_human_validation_audit.py prepare --strategy mixed_stratified --sample-size 10 --annotators 3
python scripts/10_human_validation_audit.py review --annotator-id A1
python scripts/10_human_validation_audit.py review --annotator-id A2
python scripts/10_human_validation_audit.py review --annotator-id A3
python scripts/10_human_validation_audit.py score --expected-annotators 3
```

Nota: este modo debug solo limita la muestra para pruebas rapidas. Para correr la
validacion completa, usa el flujo normal con `prepare` sin `--strategy mixed_stratified
--sample-size 10`.

2. Revisar imagen por imagen como anotador A1:
```bash
python scripts/10_human_validation_audit.py review --annotator-id A1
```

3. Revisar imagen por imagen como anotador A2:
```bash
python scripts/10_human_validation_audit.py review --annotator-id A2
```

4. Revisar imagen por imagen como anotador A3:
```bash
python scripts/10_human_validation_audit.py review --annotator-id A3
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
python scripts/10_human_validation_audit.py score --expected-annotators 3
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

### Pesos YOLO locales

Ultralytics descarga pesos desde GitHub cuando recibe un nombre como `yolo11n.pt` o
`yolo11x.pt` y no encuentra el archivo local. En Docker o redes corporativas esa
descarga puede fallar. Para evitarlo:

- `scripts/15_training.py` usa primero `runs/train/N1_YOLO11n/weights/best.pt` cuando
  se selecciona `--model n`.
- `scripts/07_judge_scoring.py` busca el juez en `yolo11x.pt` dentro de la raiz del
  proyecto montado (`/dataset/yolo11x.pt` en Docker).
- Si se usan `yolo11s.pt` o `yolo11m.pt`, coloca esos archivos en la raiz del proyecto
  antes de ejecutar el entrenamiento.

## Notas

- `13_dataset_generation.py` usa deduplicacion por similitud (Hamming), no por hash exacto.
- `13_dataset_generation.py` excluye ruido semantico (`missing_label`, `bad_label`, `poor_alignment`) antes de construir datasets.
- `07_judge_scoring.py` ejecuta una sola pasada del juez y guarda `judge_score` + `label_issue` en `analysis_metadata.sqlite`.
- `08_qa_audit.py` ya no infiere; solo sincroniza resultados QA a `dataset_master.sqlite` y valida `% unknown`.
- `11_scenario_simulator.py` y `12_db_analysis_report.py` usan `analysis_images.label_issue` cuando existe.
- `12_db_analysis_report.py` calcula `Injected` sobre `Train Base` (igual que `13_dataset_generation.py`), no sobre el total limpio.
- `config_analysis.py` mantiene compatibilidad con scripts legacy via `use_pascal` y `use_propio`.


## Cómo ejecutar training seed
Eso corre con defaults:

--model n
--epochs 100
--seeds 7 42 123 999
--batch automatico si se omite
Ahora, combinaciones útiles:

python .\scripts\18_train_seed_confirm.py --dry_run
Muestra el plan sin entrenar.

python .\scripts\18_train_seed_confirm.py --model n --epochs 100 --seeds 7 42 123 999
Ejecución completa explícita.

python .\scripts\18_train_seed_confirm.py --model 1 --epochs 100 --seeds 7 42 --batch 8
Modelo numerico equivalente a `n`; batch manual.

python .\scripts\18_train_seed_confirm.py --model n --epochs 50 --seeds 42
Una sola seed, más rápido.

python .\scripts\18_train_seed_confirm.py --model n --epochs 50 --seeds 42 123
Confirmación ligera con dos seeds.

python .\scripts\18_train_seed_confirm.py --model s --epochs 100 --seeds 7 42 123 999
Versión con yolo11s.pt.

python .\scripts\18_train_seed_confirm.py --model m --epochs 100 --seeds 42 123
Más pesado; el script baja batch a 4 si detecta m.

python .\scripts\18_train_seed_confirm.py --only_missing
Solo corre los run_dir que todavía no existan.

python .\scripts\18_train_seed_confirm.py --only_missing --model n --epochs 100 --seeds 7 42 123 999
Muy útil si se interrumpió una corrida.

python .\scripts\18_train_seed_confirm.py --dry_run --only_missing --model n --epochs 100 --seeds 42 123

Notas:

- El script usa GPU solo si CUDA esta visible; si Docker se ejecuta sin `--gpus all`,
  cae a `device=cpu`.
- Para `--model n`, primero intenta usar `runs/train/N1_YOLO11n/weights/best.pt` y evita
  descargar `yolo11n.pt` desde GitHub.
- `--model` acepta `n/s/m`, `1/2/3` o nombres `.pt` soportados.

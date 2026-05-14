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

14. Validacion fisica de escenarios:
```bash
python scripts/14_validation_report.py
```

15. Entrenamiento:
```bash
python scripts/15_training.py
Salida - ejecuta los entrenamientos, es un menú interactivo
```

16. Evaluacion:
```bash
python scripts/16_evaluar_coco_persona.py
```

### Seleccion R0-R3 y confirmacion R4

17. Seleccion reducida R0-R3:
```bash
python scripts/17_doe_pipeline_reduced.py
```

18. Entrenamiento confirmatorio multi-seed:
```bash
python scripts/18_train_seed_confirm.py
IMPORTANTE: LOS R4_FINALISTS DE LA LÍNEA 40 - 48 SE TIENEN QUE COLOCAR DE FORMA MANUAL BASADO EN LOS RESULTAODS DE 17_doe_pipeline_reduced.py
```

19. Seleccion final R4:
```bash
python scripts/19_doe_r4_final.py
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
```

24. Analisis de degradacion por distancia:
```bash
python scripts/24_distance_degradation_r4.py
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
Ahora, combinaciones útiles:

python .\scripts\18_train_seed_confirm.py --dry_run
Muestra el plan sin entrenar.

python .\scripts\18_train_seed_confirm.py --model n --epochs 100 --seeds 7 42 123 999
Ejecución completa explícita.

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

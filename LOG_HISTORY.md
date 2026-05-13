# Log History

## 2026-05-11

### Renumeracion canonica de scripts

- Se renombraron los scripts del pipeline para seguir el orden operativo acordado:
- `check_db_status.py` -> `01_check_db_status.py`
- `02_5_hamming_source_report.py` -> `03_hamming_source_report.py`
- `03_etl_ingestion.py` -> `04_etl_ingestion.py`
- `01_validar_integridad.py` -> `05_validar_integridad.py`
- `12_exportar_anotacion.py` -> `06_exportar_anotacion.py`
- `04_judge_scoring.py` -> `07_judge_scoring.py`
- `05_qa_audit.py` -> `08_qa_audit.py`
- `13_analysis_bad_labels.py` -> `09_analysis_bad_labels.py`
- `19_human_validation_audit.py` -> `10_human_validation_audit.py`
- `06_scenario_simulator.py` -> `11_scenario_simulator.py`
- `08_5_db_analysis_report.py` -> `12_db_analysis_report.py`
- `07_dataset_generation.py` -> `13_dataset_generation.py`
- `08_validation_report.py` -> `14_validation_report.py`
- `09_training.py` -> `15_training.py`
- `11_evaluar_coco_persona.py` -> `16_evaluar_coco_persona.py`
- `14_doe_pipeline_reduced.py` -> `17_doe_pipeline_reduced.py`
- `10_train_seed_confirm.py` -> `18_train_seed_confirm.py`
- `15_doe_r4_final.py` -> `19_doe_r4_final.py`
- `17_plot_r4_seed_moments.py` -> `20_plot_r4_seed_moments.py`
- `18_r4_confirmatory_stats.py` -> `21_r4_confirmatory_stats.py`
- `20_r4_tukey_shapiro.py` -> `22_r4_tukey_shapiro.py`
- `16_distance_degradation_r4.py` -> `23_distance_degradation_r4.py`

### Documentacion y referencias internas

- `TEMPORAL/README.md` actualizado con la nueva numeracion y el flujo completo del pipeline.
- `TEMPORAL/DOCKER_README.md` actualizado para usar los nombres nuevos en comandos `docker run`.
- Referencias internas de ayuda, loaders y mensajes `Ejecuta primero` alineadas con la nueva numeracion.

## 2026-02-13

### One-pass Judge + QA sync (sin doble inferencia)

- `scripts/07_judge_scoring.py` refactorizado a flujo one-pass:
- una sola inferencia YOLO11x por imagen representante (`is_dedup_keep=1`);
- calcula y persiste `judge_score`, `label_issue`, `judge_has_pred`, `judge_pred_count`, `judge_max_iou`;
- mantiene reanudacion por `processed` y propaga resultados a duplicados del mismo `dedup_cluster`.
- `scripts/08_qa_audit.py` convertido a paso de sincronizacion:
- no ejecuta inferencia;
- copia `label_issue` desde `analysis_metadata.sqlite` hacia `dataset_master.sqlite`;
- exporta `exports/audit_details.csv`;
- agrega guardrail `QA_MAX_UNKNOWN_PCT` (default 5.0), fallando si `% unknown` supera el limite.

### Correcciones de consistencia en reportes

- `scripts/11_scenario_simulator.py` corregido para usar `analysis_images.label_issue` como fuente primaria QA (evita conflictos de merge con `dataset_master`).
- `scripts/11_scenario_simulator.py` ahora imprime distribucion QA al inicio para verificacion trazable.
- `scripts/12_db_analysis_report.py` corregido para usar `analysis_images.label_issue` cuando existe (fix al warning `QA merge skipped: 'label_issue'`).
- `scripts/12_db_analysis_report.py` ahora separa rechazo del juez en:
- `Judge BadLabel`, `Judge PoorAlign`, `Judge WeakValid`.
- `scripts/12_db_analysis_report.py` ajustado para calcular `Injected` sobre `Train Base` (`SPLIT_RATIOS[0]`), alineado con `scripts/13_dataset_generation.py` y `scripts/14_validation_report.py`.
- `scripts/12_db_analysis_report.py` agrega columna `Train Base` para trazabilidad del calculo de inyeccion.

### DoE estructurado (N1-N14)

- `scripts/config_analysis.py` migrado a esquema DoE estructurado con escenarios:
- `N1_Z_COCO` (control metadata-only, sin construccion de dataset fisico)
- `N2_B_Raw_0` a `N14_P_pH_JR_40`
- Se agregaron campos de escenario: `id`, `model`, `domain`, `sources`, `dedup`, `judge_thresh`, `noise_rate`, `transfer_learning`, `build_dataset`.
- Se agrego compatibilidad legacy: `use_pascal` y `use_propio`.

### Generacion de datasets

- `scripts/13_dataset_generation.py` reescrito para consumir el nuevo esquema de escenarios.
- Deduplicacion actualizada a pHash por distancia de Hamming (`<= config.HAMMING_THRESHOLD`, default 10).
- Soporte de compatibilidad para escenarios legacy (`sources` o `use_pascal/use_propio`).
- `N1_Z_COCO` se omite correctamente en construccion (`build_dataset=False`).
- Noise injection se mantiene solo en `train`.

### Consistencia metodologica previa

- `scripts/08_qa_audit.py`: fallback de `IOU_THRESHOLD` a `0.5` si no existe en `config.py`; comentarios alineados al criterio real por imagen (max IoU GT-pred).
- `scripts/04_etl_ingestion.py`: redundancia actualizada a criterio de Hamming y documentacion de contrato YOLO normalizado para QA.
- `scripts/config.py`: agregado `HAMMING_THRESHOLD = 10`.

### Alineacion DoE completa y trazabilidad

- `scripts/13_dataset_generation.py`: ahora excluye ruido semantico (`missing_label`, `bad_label`, `poor_alignment`) antes de dedup/filtrado Judge/construccion.
- `scripts/11_scenario_simulator.py`: reescrito para escenarios N1-N14 y logica alineada con metodologia (fuente -> filtro semantico -> pHash Hamming<=10 -> Judge -> Smart Noise).
- `scripts/14_validation_report.py`: migrado de validacion heredada (M1-M4) a validacion dinamica de escenarios DoE activos (`build_dataset=True`).
- `scripts/12_db_analysis_report.py`: reescrito con la misma logica DoE para evitar discrepancias entre reportes.
- `TRACEABILITY_MATRIX_DOE.csv`: agregado mapeo oficial `R0-1..R3-4` -> `N1..N14` con parametros metodologicos (pHash, tau, noise, TL).

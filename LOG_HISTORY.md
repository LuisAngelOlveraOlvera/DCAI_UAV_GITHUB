# Log History

## 2026-05-17

### Menus seleccionables y registro SQLite para pasos 17-24

- `scripts/19_doe_r4_final.py` ahora permite elegir los datasets donde se evalua R4:
- menu interactivo si hay TTY;
- `--datasets` para ejecucion no interactiva por nombre, indice o `all`;
- `--list-datasets` para listar `VISDRONE`, `NTUT`, `COCO_TEST`, etc.
- `scripts/23_evaluar_coco_distancia.py` ahora permite elegir los escenarios donde se
  evalua distancia:
- menu interactivo si hay TTY;
- `--scenarios` para ejecucion no interactiva por alias `N1`, `N2`, nombre completo o `all`;
- `--list-scenarios` para listar escenarios disponibles.
- Se agrego registro comun en SQLite mediante `utils.run_with_sqlite_registration()` y
  `utils.register_script_run()`.
- Los scripts `17_doe_pipeline_reduced.py`, `18_train_seed_confirm.py`,
  `19_doe_r4_final.py`, `20_plot_r4_seed_moments.py`,
  `21_r4_confirmatory_stats.py`, `22_r4_tukey_shapiro.py`,
  `23_evaluar_coco_distancia.py` y `24_distance_degradation_r4.py`
  ahora escriben su ejecucion en `dataset_master.sqlite`, tabla:
- `script_run_registry`
- El registro guarda `script_name`, `status`, timestamps, duracion, `cwd`, `argv`,
  salidas declaradas y texto de error si aplica.
- Se ajusto el wrapper para tratar `SystemExit(0)` como salida exitosa, de modo que
  comandos tipo `--list-datasets` tambien queden registrados.
- Verificacion real en SQLite:
- `17_doe_pipeline_reduced.py` -> `success`
- `18_train_seed_confirm.py` -> `success` (`--dry_run`)
- `19_doe_r4_final.py` -> `success` (`--list-datasets`)
- `20_plot_r4_seed_moments.py` -> `success`
- `21_r4_confirmatory_stats.py` -> `success`
- `22_r4_tukey_shapiro.py` -> `success`
- `23_evaluar_coco_distancia.py` -> `success` (`--list-scenarios`)
- `24_distance_degradation_r4.py` -> `success` tras correccion

### Compatibilidad NumPy en analisis de degradacion por distancia

- `scripts/24_distance_degradation_r4.py` corregido para usar `np.trapezoid()` cuando
  esta disponible y caer a `np.trapz()` solo como compatibilidad.
- La causa era un fallo en entornos donde `np.trapz` ya no esta expuesto por la version
  instalada de NumPy.

### Fix portable de rutas en evaluacion por distancia

- `scripts/23_evaluar_coco_distancia.py` corregido para que los YAML temporales de
  `temp_distance_subsets/` resuelvan rutas de forma portable.
- La causa del fallo era `create_temp_yaml()` usando `path: '..'` junto con
  `temp_distance_subsets/subset_*.txt`; Ultralytics reinterpretaba esa combinacion fuera
  del root del repo y terminaba buscando archivos como:
- `C:\...\TESIS_DIC_2025\temp_distance_subsets\subset_5m.txt`
- en vez de:
- `C:\...\DCAI_UAV_GITHUB\temp_distance_subsets\subset_5m.txt`
- La solucion ahora ancla el YAML al root real del proyecto calculado en runtime con:
- `path: str(config.DATASET_ROOT.resolve())`
- y mantiene `train`, `val` y `test` apuntando al `.txt` con ruta relativa al proyecto.
- Esto conserva portabilidad entre entornos locales y Docker sin depender del
  `datasets_dir` interno de Ultralytics.
- Verificacion ejecutada con:
- `python scripts/23_evaluar_coco_distancia.py`
- El error original de `images not found` desaparecio y la corrida avanzo evaluando al
  menos `N1_YOLO11n` en `5m` y `10m` correctamente antes de alcanzar el timeout del
  entorno de automatizacion.

### Fix de grafica ANOVA en R4 confirmatorio

- `scripts/21_r4_confirmatory_stats.py` corregido en `plot_anova_effects()`.
- La causa del error era el uso de etiquetas categoricas (`Term`) directamente en
  `ax.text()` sobre un eje de barras, lo que hacia que Matplotlib fallara durante
  `plt.tight_layout()` con:
- `TypeError: only 0-dimensional arrays can be converted to Python scalars`
- La solucion ahora dibuja barras y anotaciones con posiciones numericas explicitas y
  luego asigna las etiquetas del eje X con `set_xticks()` y `set_xticklabels()`.
- Se verifico ejecucion local exitosa con:
- `python scripts/21_r4_confirmatory_stats.py`
- `README.md` y `DOCKER_README.md` actualizados con la causa del fallo y la nota de que
  el comando Docker no cambia.
- Intento de verificacion Docker realizado, pero el daemon no estaba disponible en el
  entorno (`//./pipe/dockerDesktopLinuxEngine` / `//./pipe/docker_engine` no encontrados),
  por lo que la comprobacion end-to-end en contenedor queda condicionada a levantar
  Docker Desktop o el servicio Docker del host.

## 2026-05-16

### Entrenamiento no interactivo en Docker

- `scripts/15_training.py` ahora acepta ejecucion CLI no interactiva con:
- `--datasets`
- `--model`
- `--epochs`
- `--batch`
- `--yes`
- Se evita el `EOFError: EOF when reading a line` al correr Docker sin `-it`.
- El modo interactivo se mantiene disponible ejecutando el script sin argumentos; en
  Docker debe usarse `docker run -it`.
- `DOCKER_README.md` y `README.md` actualizados con comandos no interactivos para el
  paso 15 y nota del modo interactivo.
- `scripts/15_training.py` ahora resuelve `yolo11n.pt` hacia el checkpoint local
  `runs/train/N1_YOLO11n/weights/best.pt` antes de permitir una descarga de Ultralytics.
- `scripts/15_training.py` ya no fuerza `device=0`; usa GPU solo cuando CUDA esta visible
  dentro del contenedor y cae a `device=cpu` si Docker se ejecuto sin `--gpus all`.
- El menu de modelo acepta tambien `1`, `2`, `3` como alias de `n`, `s`, `m`.
- `scripts/07_judge_scoring.py` ahora busca `yolo11x.pt` en la raiz del proyecto montado
  antes de caer en descarga automatica.
- `README.md` y `DOCKER_README.md` documentan que Ultralytics descarga desde GitHub si
  no encuentra los pesos locales, lo cual puede fallar en Docker o redes corporativas.

### Seleccion de datasets en evaluacion externa

- `scripts/16_evaluar_coco_persona.py` ya no queda limitado a evaluar todos los datasets
  externos sin control del usuario.
- Se agrego seleccion interactiva de datasets de evaluacion por indice, nombre o `all`.
- Se agrego `--datasets` para ejecucion no interactiva en Docker, por ejemplo:
- `python scripts/16_evaluar_coco_persona.py --datasets COCO_TEST VISDRONE`
- Se agrego `--list-datasets` para listar los datasets externos disponibles.
- Se agrego seleccion interactiva de pesos detectados en `runs/train/*/weights/best.pt`.
- Se agrego `--weights` para elegir pesos por nombre de run, indice o `all`.
- Se agrego `--list-weights` para listar pesos disponibles antes de ejecutar.
- Los YAML temporales de evaluacion ahora escriben `path` como ruta absoluta calculada
  en runtime para evitar que Ultralytics reinterprete rutas relativas bajo
  `/dataset/datasets`.
- Se corrigio el nombre de escenario `N2_B_Raw_0_yolo11n_e3` removiendo una coma
  accidental al final.
- `README.md` y `DOCKER_README.md` actualizados con ejemplos para evaluar todos o solo
  algunos datasets externos y pesos especificos.

### Seed confirmation training robusto

- `scripts/18_train_seed_confirm.py` recibe las mismas mejoras operativas aplicadas a
  `scripts/15_training.py`.
- `--model` acepta `n/s/m`, `1/2/3` o nombres `.pt` soportados.
- Se agrego `--batch`; si se omite, el batch se calcula automaticamente segun CUDA/VRAM.
- El script usa GPU solo si CUDA esta visible dentro del contenedor; si Docker se ejecuta
  sin `--gpus all`, cae a `device=cpu`.
- Para `--model n`, primero intenta usar el checkpoint local
  `runs/train/N1_YOLO11n/weights/best.pt` para evitar descargas automaticas de
  Ultralytics/GitHub.
- `README.md` y `DOCKER_README.md` actualizados con comandos multi-seed explicitos,
  `--only_missing`, `--batch` y notas de GPU/pesos locales.

## 2026-05-15

### Seleccion de escenarios en dataset generation

- `scripts/13_dataset_generation.py` ahora acepta `--scenarios` para construir solo los
  datasets DoE deseados.
- `--scenarios` acepta tanto nombres completos como aliases cortos definidos en
  `config_analysis`, por ejemplo `N2` -> `N2_B_Raw_0`.
- El comportamiento por defecto se mantiene: sin argumentos sigue construyendo todos los
  escenarios definidos en `config_analysis.SCENARIOS`.
- `README.md` y `DOCKER_README.md` actualizados con ejemplos para correr solo
  `N2_B_Raw_0` o multiples escenarios.

### Higiene de Git/Docker y diagnostico de carpetas grandes

- Se documento el incidente donde el repositorio parecia ocupar ~33 GB por la suma de
  `.git/objects` (~16.5 GB) y `DATASET_KAGGLE/` (~16.3 GB).
- Se agregaron instrucciones de diagnostico Windows/PowerShell con:
- `git count-objects -vH`
- medicion de carpetas grandes con `Get-ChildItem`
- medicion de `.git\objects`
- Se documento la limpieza con `git gc --prune=now` y la verificacion posterior con
  `git count-objects -vH`.
- Se registro que Windows puede bloquear packs viejos dentro de `.git/objects/pack`
  por procesos como OneDrive, VS Code, Docker Desktop, Explorer o antivirus.
- `.dockerignore` ampliado para excluir `.github`, caches, checkpoints, modelos
  adicionales, comprimidos y videos del contexto Docker.
- `.gitignore` agregado para evitar versionar datasets, modelos, bases SQLite, logs,
  resultados, videos, comprimidos, caches y ambientes virtuales.
- `README.md` y `DOCKER_README.md` ampliados para explicar la diferencia entre
  `.gitignore` y `.dockerignore`, y para prevenir builds con contextos de decenas de GB.

### Debug de validacion humana con 10 imagenes

- Se confirmo que `scripts/10_human_validation_audit.py` se mantiene sin cambios.
- `README.md` y `DOCKER_README.md` documentan el flujo debug de 10 imagenes usando:
- `prepare --strategy mixed_stratified --sample-size 10 --annotators 3`
- `review --annotator-id A1`
- `review --annotator-id A2`
- `review --annotator-id A3`
- `score --expected-annotators 3`
- Se aclaro que este modo es solo para pruebas rapidas; para la validacion completa se
  debe usar el flujo normal con `prepare` sin `--strategy mixed_stratified --sample-size 10`.

### Documentacion de visualizacion interactiva Docker/X11

- `README.md` actualizado con un paso previo para scripts que abren ventanas interactivas
  con OpenCV/Qt o Matplotlib (`cv2.imshow()`, `cv2.waitKey()`, `plt.show()`).
- `DOCKER_README.md` actualizado con la solucion aplicada para ejecutar
  `scripts/09_analysis_bad_labels.py` desde Docker con salida grafica en el host.
- Se documento el flujo Windows + PowerShell + XLaunch/VcXsrv:
- instalar y abrir XLaunch/VcXsrv;
- usar `Multiple windows`;
- usar `Start no client`;
- activar `Disable access control`;
- ejecutar Docker con `DISPLAY=host.docker.internal:0.0` y `QT_X11_NO_MITSHM=1`.
- Se documento el flujo Linux/Ubuntu:
- habilitar acceso X11 con `xhost +local:docker`;
- montar `/tmp/.X11-unix`;
- pasar `DISPLAY=$DISPLAY` al contenedor.

## 2026-05-14

### Integracion de dataset Kaggle y estandarizacion Docker/README

- `requirements.txt` actualizado para incluir dependencias del setup de Kaggle:
- `kagglehub==0.3.4`
- `requests==2.32.3`
- `urllib3==2.2.3`
- `dataset_setup.py` fue movido a `scripts/00_dataset_setup.py` y renombrado como paso `00` del flujo operativo.
- `scripts/00_dataset_setup.py` creado como paso `00` para descargar y reorganizar el dataset
  `luisngeld/person-detection-uav-pascal-voc-uaq-msuav` dentro de `DATASET_KAGGLE/`.
- La estructura objetivo documentada y generada por el setup queda alineada a:
- `DATASET_KAGGLE/PASCAL_VOC_UAQ_MSUAV/images`
- `DATASET_KAGGLE/PASCAL_VOC_UAQ_MSUAV/labels`
- `DATASET_KAGGLE/EVALUATION/{COCO_TEST,IRINA,MANIPAL_UAV,NTUT,UAQ_MSUAV_TEST,VISDRONE}/{images,labels}`
- `DATASET_KAGGLE/evaluation_videos/VIDEO_1`
- `DATASET_KAGGLE/evaluation_videos/VIDEO_2`

### Alineacion de rutas del pipeline

- `scripts/config.py` actualizado para que las primeras etapas del repo lean desde:
- `DATASET_KAGGLE/PASCAL_VOC_UAQ_MSUAV/images`
- `DATASET_KAGGLE/PASCAL_VOC_UAQ_MSUAV/labels`
- Se agregaron rutas explicitas:
- `TRAINING_DATASET_DIR`
- `EVALUATION_DATASET_DIR`
- `EVALUATION_VIDEOS_DIR`
- `scripts/16_evaluar_coco_persona.py` migrado para usar `config.EVALUATION_DATASET_DIR`.
- `scripts/19_doe_r4_final.py` migrado para usar `config.EVALUATION_DATASET_DIR`.
- `scripts/23_evaluar_coco_distancia.py` migrado para usar `config.EVALUATION_DATASET_DIR`.

### Docker y documentacion sincronizados

- `docker_dataset.dockerfile` actualizado para instalar `ca-certificates` y seguir resolviendo dependencias desde `requirements.txt`.
- `build_docker.sh` y `build_docker.ps1` actualizados para mostrar el comando de preparacion:
- `python scripts/00_dataset_setup.py`
- `README.md` ampliado con:
- flujo de descarga Kaggle;
- rutas relativas esperadas;
- nota explicita de que el pipeline base usa `PASCAL_VOC_UAQ_MSUAV` y las evaluaciones usan `EVALUATION`.
- `DOCKER_README.md` estandarizado contra `README.md` con la misma estructura Kaggle, el mismo paso de preparacion y ejemplos Docker equivalentes.
- `DOCKER_README.md` corregido para referenciar `fix_permission.sh` (nombre real del script) en lugar de `fix_permissions.sh`.

### Verificacion

- Se valido compilacion con `py_compile` para:
- `scripts/00_dataset_setup.py`
- `scripts/config.py`
- `scripts/16_evaluar_coco_persona.py`
- `scripts/19_doe_r4_final.py`
- `scripts/23_evaluar_coco_distancia.py`
- Se confirmo que la configuracion actual ya resuelve a las nuevas rutas bajo `DATASET_KAGGLE/`.
- Se detecto que el workspace todavia conserva la estructura fisica antigua `DATASET_KAGGLE/images` y `DATASET_KAGGLE/labels`; la estructura nueva se materializa al ejecutar `python scripts/00_dataset_setup.py`.

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

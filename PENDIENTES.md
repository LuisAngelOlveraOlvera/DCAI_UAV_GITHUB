# Pendientes para Publicacion y Replicabilidad

Este proyecto va por buen camino para vivir entre GitHub y Kaggle:

- GitHub debe contener codigo, configuracion, Docker, requirements, documentacion, matrices trazables y scripts reproducibles.
- Kaggle debe contener los datasets grandes (`DATASET_KAGGLE`, `EVALUATION`) y, si aplica, artefactos pesados que no conviene versionar.
- `exports/` debe generarse localmente al ejecutar el pipeline y no debe subirse a GitHub.

## 1. Pendientes Criticos Antes de Subir a GitHub

- Inicializar el repositorio Git en la carpeta correcta con `git init`.
- Crear `.gitignore` para excluir datos, salidas, pesos y caches.
- Confirmar que `.dockerignore` excluye correctamente carpetas pesadas y artefactos locales.
- Verificar que `exports/`, `runs/`, `logs/`, `DATASET_KAGGLE/`, `EVALUATION/`, `*.sqlite` y `*.pt` no entren al commit.
- Agregar una licencia (`LICENSE`) o definir explicitamente que el repositorio queda sin licencia publica por ahora.
- Agregar una seccion clara en `README.md` explicando que los datos se descargan desde Kaggle.
- Documentar la estructura esperada despues de descargar datos:

```text
DATASET_TESIS/
|-- DATASET_KAGGLE/
|   |-- images/
|   `-- labels/
|-- EVALUATION/
|   |-- COCO_TEST/
|   |-- IRINA/
|   |-- MANIPAL_UAV/
|   |-- NTUT/
|   |-- UAQ_MSUAV_TEST/
|   `-- VISDRONE/
|-- scripts/
|-- docker_dataset.dockerfile
|-- requirements.txt
|-- build_docker.ps1
|-- build_docker.sh
`-- README.md
```

## 2. Pendientes de Kaggle

- Subir `DATASET_KAGGLE` como dataset Kaggle independiente.
- Subir `EVALUATION` como dataset Kaggle independiente o como parte del mismo paquete, segun convenga por licencia/tamano.
- Definir nombres finales de los datasets Kaggle.
- Agregar en el README los links finales de Kaggle.
- Crear instrucciones de descarga manual desde Kaggle.
- Opcional: crear un script `scripts/00_download_kaggle_data.py` o `download_data.sh` para descargar datos con Kaggle API.
- Documentar requisitos para usar Kaggle API:

```bash
pip install kaggle
mkdir -p ~/.kaggle
# colocar kaggle.json en ~/.kaggle/kaggle.json
```

- Incluir checksum o conteo esperado de archivos para validar descarga.
- Documentar conteos actuales esperados:

```text
DATASET_KAGGLE/images: 15246 imagenes
DATASET_KAGGLE/labels: 15247 archivos
```

## 3. Pendientes de Docker

- Abrir Docker Desktop y validar build CPU:

```powershell
powershell -ExecutionPolicy Bypass -File .\build_docker.ps1 cpu
```

- Validar build GPU en una maquina con NVIDIA y Docker GPU habilitado:

```powershell
powershell -ExecutionPolicy Bypass -File .\build_docker.ps1 gpu
```

- Probar comando base dentro del contenedor despues de montar el repo:

```powershell
docker run --rm -v "${PWD}:/dataset" validador-imagen python scripts/01_check_db_status.py
```

- Probar que el contenedor use Python 3.12.7:

```powershell
docker run --rm validador-imagen python --version
```

- Probar importaciones clave dentro del contenedor:

```powershell
docker run --rm validador-imagen python -c "import torch, ultralytics, cv2, pandas, scipy, sklearn, seaborn, yaml; print('imports ok')"
```

- Revisar si `opencv-python` es suficiente para scripts interactivos con `cv2.imshow`; si hay problemas en entorno headless, documentar que esos scripts requieren GUI o usar `opencv-python-headless` para flujos no interactivos.

## 4. Pendientes de Requirements y Versiones

- Confirmar que las versiones en `requirements.txt` funcionan con Python 3.12.7.
- Confirmar compatibilidad entre `ultralytics==8.3.40` y `torch==2.5.1`.
- Decidir si conviene fijar versiones exactas para reproducibilidad total o rangos compatibles para mantenimiento.
- Agregar una nota en README indicando:

```text
Python objetivo: 3.12.7
Docker recomendado: usar build_docker.ps1 o build_docker.sh
```

## 5. Pendientes de README

- Separar el README en flujo minimo y flujo completo.
- Agregar seccion "Quickstart con Docker".
- Agregar seccion "Descarga de datos desde Kaggle".
- Agregar seccion "Que no esta incluido en GitHub".
- Aclarar que `exports/` se regenera con `scripts/13_dataset_generation.py` y `scripts/14_validation_report.py`.
- Aclarar que `dataset_master.sqlite` y `analysis_metadata.sqlite` son generados/intermedios o decidir si se publicaran como artefactos externos.
- Corregir textos con encoding roto (`Ã³`, `Ã¡`, `menÃº`, etc.).
- Alinear la estructura documentada con la estructura real:

```text
DATASET_KAGGLE/images
DATASET_KAGGLE/labels
```

en lugar de `images/` y `labels/` directamente en la raiz.

## 6. Pendientes de Archivos Grandes y GitHub

- No subir `exports/` a GitHub.
- No subir `runs/` a GitHub.
- No subir `logs/` a GitHub salvo que haya logs de ejemplo pequenos.
- No subir `DATASET_KAGGLE/` ni `EVALUATION/` a GitHub.
- No subir pesos grandes como `yolo11x.pt`; descargar desde fuente oficial o Kaggle si se necesita fijar version.
- Evaluar si `yolo11n.pt` se deja fuera tambien para mantener el repo limpio.
- Si algun peso debe versionarse, usar Git LFS o Kaggle/Release assets, no Git normal.

## 7. Pendientes de Reproducibilidad del Pipeline

- Probar pipeline desde un clon limpio sin `exports/`.
- Probar pipeline despues de descargar `DATASET_KAGGLE` y `EVALUATION` desde Kaggle.
- Validar orden minimo:

```bash
python scripts/01_check_db_status.py
python scripts/02_analysis_etl.py
python scripts/04_etl_ingestion.py
python scripts/05_validar_integridad.py
python scripts/07_judge_scoring.py
python scripts/08_qa_audit.py
python scripts/13_dataset_generation.py
python scripts/14_validation_report.py
```

- Decidir si `07_judge_scoring.py` debe requerir GPU o permitir CPU explicitamente.
- Confirmar que el pipeline puede reconstruir `analysis_metadata.sqlite` y `dataset_master.sqlite` desde cero.
- Agregar una prueba smoke de baja duracion para validar instalacion sin correr todo el experimento.
- Crear un comando de verificacion tipo:

```bash
python scripts/01_check_db_status.py
python scripts/05_validar_integridad.py
```

## 8. Pendientes de Calidad del Codigo

- Mantener `py_compile` limpio antes de subir:

```powershell
python -B -c "import pathlib, py_compile; [py_compile.compile(str(p), doraise=True) for p in pathlib.Path('scripts').glob('*.py')]; print('py_compile ok')"
```

- Revisar scripts interactivos para dejar claro cuales requieren ventana/GUI.
- Revisar rutas hardcodeadas o supuestos de maquina local.
- Revisar mensajes y comentarios con encoding roto.
- Considerar mover configuracion sensible o variable a `.env.example` si se agrega Kaggle API automatica.

## 9. Pendientes de Documentacion Cientifica

- Mantener `TRACEABILITY_MATRIX_DOE.csv` en GitHub.
- Mantener `LOG_HISTORY.md` si resume decisiones metodologicas y no contiene rutas privadas.
- Decidir si `REFERENCIAS_FINALES/` se excluye del repo por peso/licencias.
- Agregar `CITATION.cff` si se quiere que GitHub sugiera como citar el proyecto.
- Documentar semillas, escenarios DoE y parametros finales usados.

## 10. Criterio de "Listo para GitHub"

El repo estara listo para GitHub cuando:

- Un clon limpio pueda construir Docker CPU.
- Un usuario pueda descargar datos desde Kaggle siguiendo README.
- `exports/` pueda generarse localmente sin estar incluida en GitHub.
- No haya archivos mayores a 100 MB versionados en Git normal.
- `requirements.txt`, Dockerfile y scripts de build esten presentes.
- El README explique claramente GitHub vs Kaggle.
- `py_compile` pase en todos los scripts.
- La estructura documentada coincida con la estructura real.

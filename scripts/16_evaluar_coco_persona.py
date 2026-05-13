"""
16 EXTERNAL EVALUATION (multi-dataset comparison)
-----------------------------------------------------------------------
Compare BASELINE and experimental models by:
1) evaluating multiple external datasets sequentially
2) auto-detecting trained weights from runs/train
3) exporting per-dataset and consolidated CSV results
4) keeping GPU memory usage conservative during evaluation
-----------------------------------------------------------------------
"""

import sys
import gc
import re
import yaml
import torch
import pandas as pd
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

import config
import utils

# ============================================================
# GENERAL CONFIGURATION
# ============================================================

# Scenarios / models to evaluate


SCENARIOS = [
    'N1_YOLO11n',
    'N2_B_Raw_0_yolo11n_e3,'
]

# External datasets (sequential evaluation)
TEST_DATASETS = {
    "VISDRONE": Path(r"EVALUATION\VISDRONE"),
    "COCO_TEST": Path(r"EVALUATION\COCO_TEST"),
    "MANIPAL_UAV": Path(r"EVALUATION\MANIPAL_UAV"),
    "IRINA": Path(r"EVALUATION\IRINA"),
    "NTUT": Path(r"EVALUATION\NTUT"),
    "DOMINIO_DRON": Path(r"EVALUATION\UAQ_MSUAV_TEST"),
}

# Execution options
SAVE_PLOTS = True  # Save evaluation plots (PR, confusion matrix, etc.) for each evaluation.
CLEAN_TEMP_YAMLS = True
SMOKE_TEST = False
SMOKE_MAX_DATASETS = 1
SMOKE_MAX_SCENARIOS = 2

# Logger
logger = utils.setup_logger(
    "Final_Evaluation",
    log_file=str(config.LOGS_DIR / "final_evaluation_multidataset.log")
)


def to_project_relative(path: Path) -> str:
    """Return a project-relative path string when possible."""
    try:
        return path.relative_to(config.DATASET_ROOT).as_posix()
    except ValueError:
        return str(path)

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def clean_gpu():
    """Libera VRAM entre evaluaciones."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def find_best_weights(runs_dir: Path, scenarios):
    """
    Busca automáticamente best.pt para cada escenario
    dentro de runs/train.
    """
    weights = {}

    if not runs_dir.exists():
        logger.error(f"No existe el directorio runs/train: {runs_dir}")
        return weights

    for scen in scenarios:
        candidates = sorted(
            [
                d for d in runs_dir.iterdir()
                if d.is_dir() and (d.name == scen or d.name.startswith(scen + "_"))
            ],
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )

        if candidates:
            best_pt = candidates[0] / "weights" / "best.pt"
            if best_pt.exists():
                weights[scen] = best_pt
                logger.info(f"Pesos detectados: {scen} -> {best_pt}")
            else:
                logger.warning(f"Sin best.pt para {scen}")
        else:
            logger.warning(f"No se encontró carpeta para {scen}")

    return weights

def create_test_yaml(dataset_dir: Path, yaml_path: Path):
    """
    Genera YAML temporal para YOLO (solo clase persona).
    """
    data = {
        'path': to_project_relative(dataset_dir),
        'train': 'images',
        'val': 'images',
        'test': 'images',
        'names': {0: 'person'}
    }

    with open(yaml_path, 'w') as f:
        yaml.dump(data, f)

    return yaml_path

def extract_scenario_id(scenario_name: str) -> str:
    """
    Extrae ID corto tipo N1, N2, ... desde el nombre del escenario.
    Si no existe patrón N#, retorna el nombre original sanitizado.
    """
    m = re.match(r"^(N\d+)", str(scenario_name))
    if m:
        return m.group(1)
    return str(scenario_name).replace(" ", "_")

# ============================================================
# PROCESO PRINCIPAL
# ============================================================

def run_comparison():
    print("\n" + "=" * 90)
    print(" EVALUACIÓN COMPARATIVA FINAL - MULTI DATASET (TESIS)")
    print("=" * 90)
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | SAVE_PLOTS={SAVE_PLOTS} | SMOKE_TEST={SMOKE_TEST}")

    # --------------------------------------------------------
    # 1. Buscar pesos entrenados
    # --------------------------------------------------------
    runs_train_dir = config.DATASET_ROOT / "runs" / "train"
    weights_map = find_best_weights(runs_train_dir, SCENARIOS)

    if not weights_map:
        logger.error(" No se encontraron pesos entrenados.")
        return

    datasets_to_eval = list(TEST_DATASETS.items())
    weights_to_eval = list(weights_map.items())
    if SMOKE_TEST:
        datasets_to_eval = datasets_to_eval[:SMOKE_MAX_DATASETS]
        weights_to_eval = weights_to_eval[:SMOKE_MAX_SCENARIOS]
        print(
            f" Modo smoke test activo: "
            f"{len(datasets_to_eval)} dataset(s), {len(weights_to_eval)} escenario(s)."
        )

    print("\nModelos detectados:")
    for scen, path in weights_to_eval:
        print(f"   {scen:<15} -> {path.parent.parent.name}")

    # --------------------------------------------------------
    # 2. Sequential evaluation by DATASET
    # --------------------------------------------------------
    all_results = []
    trace_rows = []
    trace_root = config.EXPORTS_DIR / "val_trace"
    trace_root.mkdir(parents=True, exist_ok=True)
    created_yaml_paths = []

    for dataset_name, dataset_path in datasets_to_eval:
        print("\n" + "=" * 80)
        print(f" DATASET EXTERNO: {dataset_name}")
        print("=" * 80)

        if not dataset_path.exists():
            logger.warning(f"Dataset no encontrado: {dataset_path}")
            continue

        # YAML temporal por dataset
        yaml_path = config.DATASET_ROOT / f"test_{dataset_name.lower()}.yaml"
        create_test_yaml(dataset_path, yaml_path)
        created_yaml_paths.append(yaml_path)

        dataset_results = []

        for scen, weight_path in weights_to_eval:
            print(f"\n---  {dataset_name} | {scen} ---")
            clean_gpu()
            scenario_id = extract_scenario_id(scen)
            eval_id = f"{scenario_id}_{dataset_name}"
            val_output_dir = trace_root / eval_id
            model = None

            try:
                model = YOLO(str(weight_path))

                metrics = model.val(
                    data=str(yaml_path),
                    split='test',
                    batch=8,
                    imgsz=640,
                    device=device,
                    verbose=False,
                    plots=SAVE_PLOTS,
                    project=str(trace_root),
                    name=eval_id,
                    exist_ok=True,
                    classes=[0]  # LIMITADO A CLASE 0 (person)
                )

                res = {
                    "Dataset": dataset_name,
                    "Escenario": scen,
                    "ScenarioID": scenario_id,
                    "EvalID": eval_id,
                    "WeightPath": to_project_relative(weight_path),
                    "ValOutputDir": to_project_relative(val_output_dir),
                    "mAP50-95": round(metrics.box.map, 4),
                    "mAP50": round(metrics.box.map50, 4),
                    "Precision": round(metrics.box.mp, 4),
                    "Recall": round(metrics.box.mr, 4)
                }

                dataset_results.append(res)
                all_results.append(res)
                trace_rows.append({
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "EvalID": eval_id,
                    "ScenarioID": scenario_id,
                    "Escenario": scen,
                    "Dataset": dataset_name,
                    "WeightPath": to_project_relative(weight_path),
                    "YamlPath": to_project_relative(yaml_path),
                    "ValOutputDir": to_project_relative(val_output_dir),
                    "mAP50-95": res["mAP50-95"],
                    "mAP50": res["mAP50"],
                    "Precision": res["Precision"],
                    "Recall": res["Recall"]
                })

                print(f"    mAP50-95 = {res['mAP50-95']}")

            except Exception as e:
                logger.error(f"Error en {dataset_name} | {scen}: {e}")

            if model is not None:
                del model
            clean_gpu()

        # ----------------------------------------------------
        # Guardar CSV por dataset
        # ----------------------------------------------------
        if dataset_results:
            df_ds = pd.DataFrame(dataset_results)
            out_csv = config.EXPORTS_DIR / f"results_{dataset_name.lower()}.csv"
            df_ds.to_csv(out_csv, index=False)
            print(f"\n CSV dataset guardado en: {out_csv}")

    # --------------------------------------------------------
    # 3. CSV GLOBAL CONSOLIDADO
    # --------------------------------------------------------
    if all_results:
        df_all = pd.DataFrame(all_results)

        out_csv = config.EXPORTS_DIR / "thesis_results_all_datasets.csv"
        df_all.to_csv(out_csv, index=False)

        print("\n" + "=" * 90)
        print(" RESULTADOS GLOBALES – TODOS LOS DATASETS")
        print("=" * 90)

        try:
            from tabulate import tabulate
            print(tabulate(df_all, headers='keys', tablefmt='github', showindex=False))
        except ImportError:
            print(df_all.to_string(index=False))

        print(f"\n CSV global guardado en: {out_csv}")

    # Traceable record per evaluation (N# + DATASET)
    if trace_rows:
        trace_csv = config.EXPORTS_DIR / "val_trace_registry.csv"
        pd.DataFrame(trace_rows).to_csv(trace_csv, index=False)
        print(f" Registro de trazabilidad guardado en: {trace_csv}")

    if CLEAN_TEMP_YAMLS and created_yaml_paths:
        removed = 0
        for ypath in created_yaml_paths:
            try:
                if ypath.exists():
                    ypath.unlink()
                    removed += 1
            except Exception as e:
                logger.warning(f"No se pudo eliminar YAML temporal {ypath}: {e}")
        print(f" YAML temporales eliminados: {removed}")

    print("\n" + "=" * 90)
    print(" EVALUACIÓN FINALIZADA")
    print("=" * 90)

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_comparison()

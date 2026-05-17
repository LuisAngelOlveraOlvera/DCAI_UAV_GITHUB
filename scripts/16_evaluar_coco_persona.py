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
import argparse
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
    'N2_B_Raw_0_yolo11n_e3',
]

# External datasets (sequential evaluation)
TEST_DATASETS = {
    "VISDRONE": config.EVALUATION_DATASET_DIR / "VISDRONE",
    "COCO_TEST": config.EVALUATION_DATASET_DIR / "COCO_TEST",
    "MANIPAL_UAV": config.EVALUATION_DATASET_DIR / "MANIPAL_UAV",
    "IRINA": config.EVALUATION_DATASET_DIR / "IRINA",
    "NTUT": config.EVALUATION_DATASET_DIR / "NTUT",
    "DOMINIO_DRON": config.EVALUATION_DATASET_DIR / "UAQ_MSUAV_TEST",
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


def discover_available_weights(runs_dir: Path):
    """Discover every runs/train/*/weights/best.pt checkpoint."""
    weights = {}

    if not runs_dir.exists():
        logger.error(f"No existe el directorio runs/train: {runs_dir}")
        return weights

    run_dirs = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir()],
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )

    for run_dir in run_dirs:
        best_pt = run_dir / "weights" / "best.pt"
        if best_pt.exists():
            weights[run_dir.name] = best_pt

    return weights


def create_test_yaml(dataset_dir: Path, yaml_path: Path):
    """
    Genera YAML temporal para YOLO (solo clase persona).
    """
    data = {
        # Ultralytics resolves relative YAML paths under its datasets_dir setting.
        # Use the runtime absolute path so Docker mounts remain portable.
        'path': str(dataset_dir.resolve()),
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


def get_available_eval_datasets():
    """Return configured evaluation datasets that exist on disk."""
    return {
        name: path
        for name, path in TEST_DATASETS.items()
        if path.exists()
    }


def resolve_eval_datasets(selection, available_datasets):
    """Resolve dataset selection by name, index, or 'all'."""
    if not available_datasets:
        print(" No se encontraron datasets de evaluacion en DATASET_KAGGLE/EVALUATION.")
        return []

    if not selection or selection == ["all"]:
        return list(available_datasets.items())

    names = list(available_datasets.keys())
    selected = []
    missing = []

    for item in selection:
        item = item.strip()
        if not item:
            continue
        if item.lower() == "all":
            return list(available_datasets.items())
        if item.isdigit():
            idx = int(item) - 1
            if 0 <= idx < len(names):
                selected.append((names[idx], available_datasets[names[idx]]))
            else:
                missing.append(item)
            continue

        match = next((name for name in names if name.lower() == item.lower()), None)
        if match:
            selected.append((match, available_datasets[match]))
        else:
            missing.append(item)

    if missing:
        raise ValueError(f"Datasets de evaluacion no encontrados: {', '.join(missing)}")

    return list(dict(selected).items())


def resolve_weights(selection, available_weights):
    """Resolve weight selection by run name, index, or 'all'."""
    if not available_weights:
        print(" No se encontraron pesos en runs/train/*/weights/best.pt.")
        return []

    if not selection or selection == ["all"]:
        return list(available_weights.items())

    names = list(available_weights.keys())
    selected = []
    missing = []

    for item in selection:
        item = item.strip()
        if not item:
            continue
        if item.lower() == "all":
            return list(available_weights.items())
        if item.isdigit():
            idx = int(item) - 1
            if 0 <= idx < len(names):
                selected.append((names[idx], available_weights[names[idx]]))
            else:
                missing.append(item)
            continue

        match = next((name for name in names if name.lower() == item.lower()), None)
        if match:
            selected.append((match, available_weights[match]))
        else:
            missing.append(item)

    if missing:
        raise ValueError(f"Pesos no encontrados: {', '.join(missing)}")

    return list(dict(selected).items())


def interactive_dataset_menu(available_datasets):
    print("\nDatasets de evaluacion disponibles:")
    for i, name in enumerate(available_datasets.keys(), start=1):
        print(f"  [{i}] {name}")

    print("\nOpciones:")
    print("  - Escribe los numeros separados por coma (ej: 1,3)")
    print("  - Escribe nombres separados por coma (ej: COCO_TEST,VISDRONE)")
    print("  - Escribe 'all' para evaluar TODOS secuencialmente")

    selection = input("\n Seleccion: ").strip()
    if not selection:
        selection = "all"

    tokens = [token.strip() for token in selection.split(",")]
    return resolve_eval_datasets(tokens, available_datasets)


def interactive_weights_menu(available_weights):
    print("\nPesos disponibles en runs/train:")
    for i, (name, path) in enumerate(available_weights.items(), start=1):
        print(f"  [{i}] {name} -> {to_project_relative(path)}")

    print("\nOpciones:")
    print("  - Escribe los numeros separados por coma (ej: 1,3)")
    print("  - Escribe nombres de run separados por coma")
    print("  - Escribe 'all' para evaluar TODOS secuencialmente")

    selection = input("\n Seleccion de pesos: ").strip()
    if not selection:
        selection = "all"

    tokens = [token.strip() for token in selection.split(",")]
    return resolve_weights(tokens, available_weights)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evalua modelos entrenados en datasets externos."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Datasets a evaluar por nombre, indice o 'all'. Default: menu si hay TTY; all si no hay TTY.",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        help="Pesos a evaluar por nombre de run, indice o 'all'. Default: menu si hay TTY; escenarios configurados si no hay TTY.",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="Lista datasets de evaluacion disponibles y sale.",
    )
    parser.add_argument(
        "--list-weights",
        action="store_true",
        help="Lista pesos disponibles en runs/train y sale.",
    )
    return parser.parse_args()


def select_datasets_for_run(args):
    available_datasets = get_available_eval_datasets()

    if args.list_datasets:
        print("\nDatasets de evaluacion disponibles:")
        for i, name in enumerate(available_datasets.keys(), start=1):
            print(f"  [{i}] {name}")
        sys.exit(0)

    if args.datasets is not None:
        return resolve_eval_datasets(args.datasets, available_datasets)

    if sys.stdin.isatty():
        return interactive_dataset_menu(available_datasets)

    return list(available_datasets.items())


def select_weights_for_run(args):
    runs_train_dir = config.DATASET_ROOT / "runs" / "train"
    available_weights = discover_available_weights(runs_train_dir)

    if args.list_weights:
        print("\nPesos disponibles en runs/train:")
        for i, (name, path) in enumerate(available_weights.items(), start=1):
            print(f"  [{i}] {name} -> {to_project_relative(path)}")
        sys.exit(0)

    if args.weights is not None:
        return resolve_weights(args.weights, available_weights)

    if sys.stdin.isatty():
        return interactive_weights_menu(available_weights)

    configured = find_best_weights(runs_train_dir, SCENARIOS)
    return list(configured.items())

# ============================================================
# PROCESO PRINCIPAL
# ============================================================

def run_comparison(datasets_to_eval=None, weights_to_eval=None):
    print("\n" + "=" * 90)
    print(" EVALUACIÓN COMPARATIVA FINAL - MULTI DATASET (TESIS)")
    print("=" * 90)
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | SAVE_PLOTS={SAVE_PLOTS} | SMOKE_TEST={SMOKE_TEST}")

    # --------------------------------------------------------
    # 1. Buscar pesos entrenados
    # --------------------------------------------------------
    if weights_to_eval is None:
        weights_to_eval = list(find_best_weights(config.DATASET_ROOT / "runs" / "train", SCENARIOS).items())
    if not weights_to_eval:
        logger.error(" No se encontraron pesos entrenados.")
        return

    if datasets_to_eval is None:
        datasets_to_eval = list(get_available_eval_datasets().items())
    if not datasets_to_eval:
        logger.error(" No se seleccionaron datasets de evaluacion.")
        return

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
    args = parse_args()
    run_comparison(select_datasets_for_run(args), select_weights_for_run(args))

"""
18 MULTI-SEED CONFIRMATION TRAINING (R4 finalists)
-----------------------------------------------------------------------
Train R4 finalist scenarios by:
1) repeating runs across confirmation seeds
2) executing sequentially to avoid GPU out-of-memory failures
3) saving cumulative CSV outputs for traceability
-----------------------------------------------------------------------
"""

import sys
import gc
import csv
import time
import yaml
import torch
import random
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

import config
import utils


# ============================================================
# LOGGER
# ============================================================
logger = utils.setup_logger(
    "Seed_Confirm_Train",
    log_file=str(config.LOGS_DIR / "training_session_seed_confirm.log")
)

# ============================================================
# CONFIG
# ============================================================
R4_FINALISTS = [
    "B_Raw_0",
    "P_Raw_0",
    "H_JR_0",
    "H_pH_JS_0",
    "H_pH_JR_15",
    "H_pH_JR_40",
    "H_pH_JS_40",
]
DEFAULT_SEEDS = [7, 42, 123, 999]
N1_BASELINE_BEST_PT = config.DATASET_ROOT / "runs" / "train" / "N1_YOLO11n" / "weights" / "best.pt"


# ============================================================
# GESTION DE MEMORIA (CRITICO PARA 6GB VRAM)
# ============================================================
def clean_gpu_memory():
    """Fuerza la liberacion de VRAM entre entrenamientos."""
    if torch.cuda.is_available():
        logger.info("   Limpiando VRAM de GPU...")
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()

        free_mem = torch.cuda.mem_get_info()[0] / 1024**3
        logger.info(f"   VRAM libre: {free_mem:.2f} GB")


def get_safe_batch_size():
    """Calcula un batch size seguro para 6GB VRAM."""
    if not torch.cuda.is_available():
        return 4

    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3

    if total_vram < 5:
        return 4
    elif total_vram < 7:
        return 8
    elif total_vram < 10:
        return 16
    else:
        return 32


def get_training_device():
    """Usa GPU solo cuando CUDA es visible dentro del entorno actual."""
    return 0 if torch.cuda.is_available() else "cpu"


def normalize_model_name(model):
    """Normaliza opciones cortas/numericas a checkpoints YOLO."""
    model = (model or "n").strip().lower()
    numeric_choices = {"1": "n", "2": "s", "3": "m"}
    model = numeric_choices.get(model, model)
    if model in ["n", "s", "m"]:
        return f"yolo11{model}.pt"
    if model in ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"]:
        return model
    raise ValueError("Modelo invalido. Usa n, s, m, 1, 2, 3, yolo11n.pt, yolo11s.pt o yolo11m.pt.")


def resolve_training_model_path(model_name):
    """Prefiere checkpoints locales para evitar descargas de Ultralytics en Docker."""
    candidates = []

    if model_name == "yolo11n.pt":
        candidates.append(N1_BASELINE_BEST_PT)

    candidates.extend(
        [
            config.DATASET_ROOT / model_name,
            Path(model_name),
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())

    logger.warning(
        f"No se encontro checkpoint local para {model_name}. "
        "Ultralytics intentara descargarlo desde GitHub."
    )
    return model_name


# ============================================================
# UTILERIAS DE CONFIGURACION
# ============================================================
def check_dataset_yaml(dataset_name):
    """Valida que exista y sea leible exports/<scenario>/data.yaml."""
    candidates = [
        config.EXPORTS_DIR / dataset_name,
        # Compatibility with folders like N2_B_Raw_0
        *[d for d in config.EXPORTS_DIR.glob(f"*_{dataset_name}") if d.is_dir()],
    ]

    yaml_path = None
    for dataset_dir in candidates:
        p = dataset_dir / "data.yaml"
        if p.exists():
            yaml_path = p
            break

    if yaml_path is None:
        logger.error(f"Falta data.yaml en {dataset_name}")
        return None

    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            yaml.safe_load(f)
    except Exception as e:
        logger.error(f"data.yaml invalido en {dataset_name}: {e}")
        return None

    return str(yaml_path)


def get_available_datasets():
    """Escanea exports/ en busca de datasets validos."""
    if not config.EXPORTS_DIR.exists():
        return []

    datasets = []
    for d in config.EXPORTS_DIR.iterdir():
        if d.is_dir() and (d / "data.yaml").exists():
            datasets.append(d.name)

    return sorted(datasets)


# ============================================================
# REPRODUCIBILIDAD
# ============================================================
def set_seed(seed):
    """Fija seeds para reproducibilidad practica."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# METRICAS / CSV
# ============================================================
def _safe_float(value):
    try:
        if value is None:
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def _first_dict_value(dct, keys):
    for key in keys:
        if key in dct:
            return dct[key]
    return float("nan")


def extract_metrics(metrics):
    """
    Extrae metricas de forma robusta para distintas versiones de Ultralytics.
    Si no existe una metrica, devuelve NaN.
    """
    out = {
        "map50_95": float("nan"),
        "map50": float("nan"),
        "precision": float("nan"),
        "recall": float("nan"),
    }

    box = getattr(metrics, "box", None)
    if box is not None:
        out["map50_95"] = _safe_float(getattr(box, "map", float("nan")))
        out["map50"] = _safe_float(getattr(box, "map50", float("nan")))
        out["precision"] = _safe_float(getattr(box, "mp", float("nan")))
        out["recall"] = _safe_float(getattr(box, "mr", float("nan")))

    results_dict = getattr(metrics, "results_dict", None)
    if isinstance(results_dict, dict):
        if np.isnan(out["map50_95"]):
            out["map50_95"] = _safe_float(
                _first_dict_value(results_dict, ["metrics/mAP50-95(B)", "metrics/mAP50-95"])
            )
        if np.isnan(out["map50"]):
            out["map50"] = _safe_float(
                _first_dict_value(results_dict, ["metrics/mAP50(B)", "metrics/mAP50"])
            )
        if np.isnan(out["precision"]):
            out["precision"] = _safe_float(
                _first_dict_value(results_dict, ["metrics/precision(B)", "metrics/precision"])
            )
        if np.isnan(out["recall"]):
            out["recall"] = _safe_float(
                _first_dict_value(results_dict, ["metrics/recall(B)", "metrics/recall"])
            )

    return out


def append_result_row(csv_path, row):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()

    fieldnames = [
        "timestamp",
        "scenario",
        "seed",
        "model_name",
        "epochs",
        "batch",
        "imgsz",
        "map50_95",
        "map50",
        "precision",
        "recall",
        "run_dir",
    ]

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ============================================================
# CLI
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Entrenamiento secuencial multi-seed para confirmacion estadistica de finalistas R4."
    )
    parser.add_argument(
        "--model",
        default="n",
        help="Modelo YOLO a usar: n|s|m, 1|2|3 o nombre .pt (default: n)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Numero de epochs (default: 100)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Seeds a ejecutar (default: 42 123)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Batch size. Si se omite, se calcula automaticamente.",
    )
    parser.add_argument(
        "--only_missing",
        action="store_true",
        help="Si run_dir ya existe, salta la corrida.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Solo imprime el plan de ejecucion sin entrenar.",
    )
    parser.add_argument(
        "--cooldown_sec",
        type=int,
        default=5,
        help="Segundos de enfriamiento entre corridas (default: 5).",
    )
    return parser.parse_args()


def print_execution_plan(plan, model_name, epochs, batch_size, results_csv):
    print("\n" + "=" * 90)
    print("PLAN DE EJECUCION - SEED CONFIRM")
    print("=" * 90)
    print(f"Modelo: {model_name}")
    print(f"Epochs: {epochs}")
    print(f"Batch: {batch_size}")
    print(f"CSV resultados: {results_csv}")
    print("-" * 90)

    for item in plan:
        status = "RUN"
        if item["yaml_path"] is None:
            status = "SKIP (sin data.yaml)"
        elif item["skip_exists"]:
            status = "SKIP (run_dir existe)"
        print(
            f"[{status}] scenario={item['scenario']} | seed={item['seed']} | run={item['run_name']}"
        )

    print("=" * 90)


# ============================================================
# BUCLE DE ENTRENAMIENTO
# ============================================================
def run_training_seed_confirm():
    args = parse_args()

    model_name = normalize_model_name(args.model)
    epochs = args.epochs
    seeds = [int(s) for s in args.seeds]

    batch_size = args.batch if args.batch is not None else get_safe_batch_size()
    device = get_training_device()
    if "m.pt" in model_name and batch_size > 4:
        batch_size = 4

    project_dir = config.DATASET_ROOT / "runs" / "train"
    project_dir.mkdir(parents=True, exist_ok=True)
    results_csv = project_dir / "seed_confirm_results.csv"

    plan = []
    for scenario in R4_FINALISTS:
        yaml_path = check_dataset_yaml(scenario)
        for seed in seeds:
            run_name = f"{scenario}_{model_name.replace('.pt','')}_e{epochs}_seed{seed}"
            run_dir = project_dir / run_name
            skip_exists = args.only_missing and run_dir.exists()
            plan.append(
                {
                    "scenario": scenario,
                    "seed": seed,
                    "yaml_path": yaml_path,
                    "run_name": run_name,
                    "run_dir": run_dir,
                    "skip_exists": skip_exists,
                }
            )

    if args.dry_run:
        print_execution_plan(plan, model_name, epochs, batch_size, results_csv)
        print(f"Device: {device}")
        return

    total_start = time.time()
    n_ok = 0
    n_skip = 0
    n_err = 0

    for scenario in R4_FINALISTS:
        logger.info(f"\n{'#' * 60}")
        logger.info(f"INICIANDO ESCENARIO: {scenario}")
        logger.info(f"{'#' * 60}")

        yaml_path = check_dataset_yaml(scenario)
        if yaml_path is None:
            logger.warning(f"Saltando escenario por YAML faltante/invalido: {scenario}")
            continue

        for seed in seeds:
            run_name = f"{scenario}_{model_name.replace('.pt','')}_e{epochs}_seed{seed}"
            run_dir = project_dir / run_name

            if args.only_missing and run_dir.exists():
                logger.info(f"SKIP --only_missing: {run_name}")
                n_skip += 1
                continue

            model = None
            try:
                clean_gpu_memory()
                set_seed(seed)

                model_path = resolve_training_model_path(model_name)
                logger.info(f"Cargando {model_path} para seed={seed}...")
                model = YOLO(model_path)

                # Protocolo solicitado
                model.train(
                    data=yaml_path,
                    epochs=epochs,
                    batch=batch_size,
                    imgsz=640,
                    device=device,
                    workers=2,
                    project=str(project_dir),
                    name=run_name,
                    exist_ok=True,
                    patience=30,
                    verbose=True,
                    seed=seed,
                    deterministic=True
                )

                logger.info(f"Entrenamiento finalizado: {run_name}")

                metrics = model.val(split='test')
                m = extract_metrics(metrics)

                if np.isnan(m["map50_95"]):
                    logger.info("Test mAP50-95: NaN")
                else:
                    logger.info(f"Test mAP50-95: {m['map50_95']:.4f}")

                append_result_row(
                    results_csv,
                    {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "scenario": scenario,
                        "seed": seed,
                        "model_name": model_name,
                        "epochs": epochs,
                        "batch": batch_size,
                        "imgsz": 640,
                        "map50_95": m["map50_95"],
                        "map50": m["map50"],
                        "precision": m["precision"],
                        "recall": m["recall"],
                        "run_dir": str(run_dir),
                    },
                )

                n_ok += 1

            except Exception as e:
                logger.error(f"Error entrenando {run_name}: {e}")
                n_err += 1

            finally:
                if model is not None:
                    del model
                clean_gpu_memory()
                logger.info(f"Enfriando GPU ({args.cooldown_sec} segundos)...")
                time.sleep(args.cooldown_sec)

    total_time_h = (time.time() - total_start) / 3600.0
    logger.info("\n" + "=" * 60)
    logger.info("SECUENCIA SEED-CONFIRM COMPLETADA")
    logger.info(f"OK: {n_ok} | SKIP: {n_skip} | ERROR: {n_err}")
    logger.info(f"Tiempo total: {total_time_h:.2f} horas")
    logger.info(f"Runs: {project_dir}")
    logger.info(f"CSV: {results_csv}")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        utils.run_with_sqlite_registration(
            script_name="18_train_seed_confirm.py",
            func=run_training_seed_confirm,
            db_path=config.DB_PATH,
            outputs={"runs_dir": config.DATASET_ROOT / "runs" / "train"},
        )
    except KeyboardInterrupt:
        logger.warning("\nProceso detenido por el usuario.")
        sys.exit(0)

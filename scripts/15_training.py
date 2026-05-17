"""
15 TRAINING (interactive and sequential)
-----------------------------------------------------------------------
Train YOLO models on generated datasets by:
1) selecting scenarios from an interactive menu
2) managing VRAM conservatively for limited GPUs
3) running experiments sequentially to reduce crash risk
-----------------------------------------------------------------------
"""

import sys
import gc
import time
import shutil
import yaml
import torch
import argparse
from pathlib import Path
from ultralytics import YOLO
import config
import utils

# ============================================================
# LOGGER
# ============================================================
logger = utils.setup_logger(
    "Interactive_Train",
    log_file=str(config.LOGS_DIR / "training_session.log")
)

N1_BASELINE_DIR = config.DATASET_ROOT / "runs" / "train" / "N1_YOLO11n" / "weights"
N1_BASELINE_BEST_PT = N1_BASELINE_DIR / "best.pt"

# ============================================================
# MEMORY MANAGEMENT (CRITICAL FOR 6GB VRAM)
# ============================================================
def clean_gpu_memory():
    """Fuerza la liberación de VRAM entre entrenamientos."""
    if torch.cuda.is_available():
        logger.info("    Limpiando VRAM de GPU...")
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
        
        # Reportar memoria actual
        free_mem = torch.cuda.mem_get_info()[0] / 1024**3
        logger.info(f"   VRAM Libre: {free_mem:.2f} GB")

def get_safe_batch_size():
    """Calcula un batch size seguro para 6GB VRAM."""
    if not torch.cuda.is_available():
        return 4
    
    # Obtener VRAM total en GB
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    if total_vram < 5:       # < 4GB
        return 4
    elif total_vram < 7:     # 6GB (Tu caso)
        return 8             # Recommended for YOLO11n on 6GB.
    elif total_vram < 10:    # 8GB
        return 16
    else:                    # > 10GB
        return 32


def get_training_device():
    """Use GPU only when CUDA is visible inside the current environment."""
    return 0 if torch.cuda.is_available() else "cpu"


def resolve_yolo11n_source():
    """Find a local yolo11n.pt checkpoint that can seed the default N1 baseline."""
    candidates = [
        config.DATASET_ROOT / "yolo11n.pt",
        Path("yolo11n.pt"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def ensure_default_n1_baseline():
    """
    Ensure the default N1 baseline checkpoint exists before any training starts.
    Copies yolo11n.pt into runs/train/N1_YOLO11n/weights/best.pt if missing.
    """
    if N1_BASELINE_BEST_PT.exists():
        logger.info(f"Default N1 baseline already present: {N1_BASELINE_BEST_PT}")
        return

    source_ckpt = resolve_yolo11n_source()
    if source_ckpt is None:
        logger.warning(
            "Default N1 baseline was not created because yolo11n.pt was not found "
            "in the project root or current working directory."
        )
        return

    N1_BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_ckpt, N1_BASELINE_BEST_PT)
    logger.info(
        "Default N1 baseline prepared at "
        f"{N1_BASELINE_BEST_PT} from {source_ckpt}"
    )

# ============================================================
# CONFIGURATION UTILITIES
# ============================================================
def check_dataset_yaml(dataset_name):
    """Asegura que el YAML apunte correctamente a las rutas absolutas."""
    dataset_dir = config.EXPORTS_DIR / dataset_name
    yaml_path = dataset_dir / "data.yaml"
    
    if not yaml_path.exists():
        logger.error(f"Falta data.yaml en {dataset_name}")
        return None
        
    return str(yaml_path)

def get_available_datasets():
    """Escanea la carpeta exports/ en busca de datasets válidos."""
    if not config.EXPORTS_DIR.exists():
        return []
    
    # Buscamos carpetas que tengan un data.yaml
    datasets = []
    for d in config.EXPORTS_DIR.iterdir():
        if d.is_dir() and (d / "data.yaml").exists():
            datasets.append(d.name)
    
    return sorted(datasets)


def normalize_model_name(model):
    """Normalize short model choices to YOLO checkpoint names."""
    model = (model or "n").strip().lower()
    numeric_choices = {"1": "n", "2": "s", "3": "m"}
    model = numeric_choices.get(model, model)
    if model in ["n", "s", "m"]:
        return f"yolo11{model}.pt"
    if model in ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"]:
        return model
    raise ValueError("Modelo invalido. Usa n, s, m, yolo11n.pt, yolo11s.pt o yolo11m.pt.")


def resolve_training_model_path(model_name):
    """Prefer local checkpoints so Ultralytics does not download during Docker runs."""
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


def resolve_selected_datasets(selection, available_datasets):
    """Resolve CLI/menu dataset selection against datasets in exports/."""
    if not available_datasets:
        print(" No se encontraron datasets en exports/.")
        print("   Ejecuta primero: python 13_dataset_generation.py")
        sys.exit(0)

    if not selection or selection == ["all"]:
        return available_datasets

    selected = []
    missing = []
    for item in selection:
        item = item.strip()
        if not item:
            continue
        if item.lower() == "all":
            return available_datasets
        if item.isdigit():
            idx = int(item) - 1
            if 0 <= idx < len(available_datasets):
                selected.append(available_datasets[idx])
            else:
                missing.append(item)
        elif item in available_datasets:
            selected.append(item)
        else:
            missing.append(item)

    if missing:
        raise ValueError(f"Datasets no encontrados: {', '.join(missing)}")
    if not selected:
        print(" Ningun dataset seleccionado.")
        sys.exit(0)

    return list(dict.fromkeys(selected))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Entrena modelos YOLO de forma interactiva o no interactiva."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Datasets a entrenar por nombre, indice o 'all'. Default: menu interactivo.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Modelo YOLO: n, s, m, yolo11n.pt, yolo11s.pt o yolo11m.pt.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Numero de epochs. Default interactivo: 50; default no interactivo: 100.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Batch size. Si se omite, se calcula automaticamente.",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Confirma el entrenamiento sin pedir input.",
    )
    return parser.parse_args()


def get_cli_training_config(args):
    available_datasets = get_available_datasets()
    selected_datasets = resolve_selected_datasets(args.datasets, available_datasets)
    model_name = normalize_model_name(args.model or "n")
    epochs = args.epochs if args.epochs is not None else 100
    batch_size = args.batch if args.batch is not None else get_safe_batch_size()
    device = get_training_device()

    if "m.pt" in model_name and batch_size > 4:
        batch_size = 4

    print("\nConfiguracion no interactiva:")
    print(f"  - Datasets: {', '.join(selected_datasets)}")
    print(f"  - Modelo: {model_name}")
    print(f"  - Epochs: {epochs}")
    print(f"  - Batch: {batch_size}")
    print(f"  - Device: {device}")

    if not args.yes:
        print("\nUsa --yes para confirmar ejecucion no interactiva.")
        sys.exit(0)

    return selected_datasets, model_name, epochs, batch_size, device

# ============================================================
# INTERACTIVE MENU
# ============================================================
def interactive_menu():
    print("\n" + "="*60)
    print("   ENTRENADOR DE MODELOS YOLO - SECUENCIAL")
    print("="*60)
    
    # 1. Seleccionar Datasets
    datasets = get_available_datasets()
    if not datasets:
        print(" No se encontraron datasets en exports/.")
        print("   Ejecuta primero: python 13_dataset_generation.py")
        sys.exit(0)
        
    print("\nDatasets disponibles:")
    for i, ds in enumerate(datasets):
        print(f"  [{i+1}] {ds}")
    
    print("\nOpciones:")
    print("  - Escribe los números separados por coma (ej: 1,3)")
    print("  - Escribe 'all' para entrenar TODOS secuencialmente")
    
    selection = input("\n Selección: ").strip().lower()
    
    selected_datasets = []
    if selection == 'all':
        selected_datasets = datasets
    else:
        try:
            indices = [int(x.strip()) - 1 for x in selection.split(',')]
            for idx in indices:
                if 0 <= idx < len(datasets):
                    selected_datasets.append(datasets[idx])
        except ValueError:
            print(" Entrada inválida.")
            sys.exit(1)
            
    if not selected_datasets:
        print(" Ningún dataset seleccionado.")
        sys.exit(0)

    # 2. Seleccionar Modelo
    print("\nModelos YOLO recomendados (para 6GB VRAM):")
    print("  [n] yolo11n.pt (Nano)   - Muy Rápido, menor precisión (Batch 16 ok)")
    print("  [s] yolo11s.pt (Small)  - Balanceado (Batch 8 recomendado)")
    print("  [m] yolo11m.pt (Medium) - Preciso, pero pesado (Posible OOM, usar Batch 4)")
    
    mod_choice = input("\n Elige modelo [n/s/m] (default 'n'): ").strip().lower()
    try:
        model_name = normalize_model_name(mod_choice or "n")
    except ValueError as exc:
        print(f" {exc}")
        sys.exit(1)

    # 3. Epochs
    ep_input = input("\n Número de Epochs (default 50): ").strip()
    epochs = int(ep_input) if ep_input.isdigit() else 50
    
    # 4. Batch Size Seguro
    safe_batch = get_safe_batch_size()
    device = get_training_device()
    # Ajuste por modelo: Si elige Medium en 6GB, bajamos el batch
    if 'm.pt' in model_name and safe_batch > 4:
        safe_batch = 4
        
    print(f"\n Configuración detectada:")
    print(f"   - Device: {device}")
    print(f"   - Batch Size Automático: {safe_batch}")
    print(f"   - Workers: 2")
    print(f"   - AMP (Mixed Precision): Activado")
    
    confirm = input("\n¿Iniciar entrenamiento secuencial? [s/n]: ").lower()
    if confirm != 's':
        sys.exit(0)
        
    return selected_datasets, model_name, epochs, safe_batch, device

# ============================================================
# BUCLE DE ENTRENAMIENTO
# ============================================================
def run_training_sequence(training_config=None):
    ensure_default_n1_baseline()
    if training_config is None:
        datasets, model_name, epochs, batch_size, device = interactive_menu()
    else:
        datasets, model_name, epochs, batch_size, device = training_config
    
    total_start = time.time()
    
    for i, ds_name in enumerate(datasets):
        model = None
        logger.info(f"\n{'#'*60}")
        logger.info(f"INICIANDO ESCENARIO {i+1}/{len(datasets)}: {ds_name}")
        logger.info(f"{'#'*60}")
        
        # 1. Limpieza Preventiva
        clean_gpu_memory()
        
        # 2. Configurar Rutas
        yaml_path = check_dataset_yaml(ds_name)
        project_dir = config.DATASET_ROOT / "runs" / "train"
        run_name = f"{ds_name}_{model_name.replace('.pt','')}_e{epochs}"
        
        try:
            # 3. Cargar Modelo
            model_path = resolve_training_model_path(model_name)
            logger.info(f"Cargando {model_path}...")
            model = YOLO(model_path)
            
            # 4. Entrenar
            # Train with 'exist_ok=True' to avoid creating exp2, exp3 on retries
            results = model.train(
                data=yaml_path,
                epochs=epochs,
                batch=batch_size,
                imgsz=640,
                device=device,
                workers=2,          # More stable on Windows + 16GB RAM
                project=str(project_dir),
                name=run_name,
                exist_ok=True,      # Overwrite folder if it already exists
                patience=30,        # Early stopping if it does not improve in 15 epochs
                verbose=True
            )
            
            logger.info(f" Entrenamiento de {ds_name} finalizado.")
            
            # 5. Test validation (optional, YOLO already validates on Val)
            logger.info("Evaluando en conjunto de TEST...")
            metrics = model.val(split='test')
            logger.info(f"Test mAP50-95: {metrics.box.map:.4f}")

        except Exception as e:
            logger.error(f" Error entrenando {ds_name}: {e}")
            logger.warning("Saltando al siguiente dataset...")
        
        finally:
            # 6. LIMPIEZA AGRESIVA POST-ENTRENAMIENTO
            if model is not None:
                del model
            clean_gpu_memory()
            logger.info("⏳ Enfriando GPU (5 segundos)...")
            time.sleep(5)

    total_time = (time.time() - total_start) / 3600
    logger.info("\n" + "="*60)
    logger.info(f" SECUENCIA COMPLETADA. Tiempo total: {total_time:.2f} horas")
    logger.info(f"Resultados guardados en: {config.DATASET_ROOT}/runs/train")
    logger.info("="*60)

if __name__ == "__main__":
    try:
        args = parse_args()
        has_cli_config = any(
            [
                args.datasets is not None,
                args.model is not None,
                args.epochs is not None,
                args.batch is not None,
                args.yes,
            ]
        )
        config_override = get_cli_training_config(args) if has_cli_config else None
        run_training_sequence(config_override)
    except KeyboardInterrupt:
        logger.warning("\n Proceso detenido por el usuario.")
        sys.exit(0)

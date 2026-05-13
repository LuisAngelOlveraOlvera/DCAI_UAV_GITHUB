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
    model_name = f"yolo11{mod_choice}.pt" if mod_choice in ['n', 's', 'm'] else "yolo11n.pt"

    # 3. Epochs
    ep_input = input("\n Número de Epochs (default 50): ").strip()
    epochs = int(ep_input) if ep_input.isdigit() else 50
    
    # 4. Batch Size Seguro
    safe_batch = get_safe_batch_size()
    # Ajuste por modelo: Si elige Medium en 6GB, bajamos el batch
    if 'm.pt' in model_name and safe_batch > 4:
        safe_batch = 4
        
    print(f"\n Configuración detectada para RTX 3060 (6GB):")
    print(f"   - Batch Size Automático: {safe_batch}")
    print(f"   - Workers: 2")
    print(f"   - AMP (Mixed Precision): Activado")
    
    confirm = input("\n¿Iniciar entrenamiento secuencial? [s/n]: ").lower()
    if confirm != 's':
        sys.exit(0)
        
    return selected_datasets, model_name, epochs, safe_batch

# ============================================================
# BUCLE DE ENTRENAMIENTO
# ============================================================
def run_training_sequence():
    ensure_default_n1_baseline()
    datasets, model_name, epochs, batch_size = interactive_menu()
    
    total_start = time.time()
    
    for i, ds_name in enumerate(datasets):
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
            logger.info(f"Cargando {model_name}...")
            model = YOLO(model_name)
            
            # 4. Entrenar
            # Train with 'exist_ok=True' to avoid creating exp2, exp3 on retries
            results = model.train(
                data=yaml_path,
                epochs=epochs,
                batch=batch_size,
                imgsz=640,
                device=0,           # GPU 0
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
        run_training_sequence()
    except KeyboardInterrupt:
        logger.warning("\n Proceso detenido por el usuario.")
        sys.exit(0)

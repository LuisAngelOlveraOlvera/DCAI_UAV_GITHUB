"""
MÓDULO 06 (EXT): EVALUACIÓN POR DISTANCIA (5m - 25m)
-----------------------------------------------------------------------
TESIS: Validación de Metodología DCAI para Detección de Personas.
OBJEIVO:
Evaluar el desempeño de los modelos en función de la distancia
de captura (5, 10, 15, 20, 25 metros) usando el dataset 'testing'.
-----------------------------------------------------------------------
"""

import sys
import gc
import argparse
import yaml
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO

import config
import utils

# ============================================================
# CONFIGURACIÓN ESPECÍFICA
# ============================================================

# CSV que contiene la relación Nombre -> Distancia
CSV_PATH = config.EVALUATION_DATASET_DIR / "UAQ_MSUAV_TEST" / "lista_archivos.csv"

# Directorio donde están REALMENTE las imágenes en Windows
TEST_IMAGES_DIR = config.EVALUATION_DATASET_DIR / "UAQ_MSUAV_TEST" / "images"

DISTANCES = [5, 10, 15, 20, 25]

# Mismos escenarios que en el script principal
SCENARIOS = [
    'N1_YOLO11n',
    'N2_B_Raw_0_yolo11n_e100',
    'N3_H_Raw_0_yolo11n_e100',
    'N4_H_pH_0_yolo11n_e100',
    'N5_H_JR_0_yolo11n_e100',
    'N6_H_pH_JR_0_yolo11n_e100',
    'N7_H_pH_JS_0_yolo11n_e100',
    'N8_H_pH_JR_15_yolo11n_e100',
    'N9_H_pH_JR_40_yolo11n_e100',
    'N10_H_pH_JS_40_yolo11n_e100',
    'N11_P_Raw_0_yolo11n_e100',
    'N12_P_pH_JR_0_yolo11n_e100',
    'N13_P_pH_JR_15_yolo11n_e100',
    'N14_P_pH_JR_40_yolo11n_e100'
]

SCENARIO_ALIASES = {scen.split("_")[0]: scen for scen in SCENARIOS}

# Logger
logger = utils.setup_logger(
    "Eval_Distance",
    log_file=str(config.LOGS_DIR / "evaluation_by_distance.log")
)

# ============================================================
# FUNCIONES
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evalua modelos por distancia con seleccion de escenarios."
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        help="Escenarios a evaluar por nombre, alias N# o indice. Usa 'all' para todos.",
    )
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="Lista escenarios disponibles y sale.",
    )
    return parser.parse_args()

def clean_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def find_best_weights(runs_dir: Path, scenarios):
    """Reutilizada: busca best.pt para cada escenario."""
    weights = {}
    if not runs_dir.exists():
        logger.error(f"No existe runs/train: {runs_dir}")
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
    return weights


def resolve_scenarios(selection, available_scenarios):
    if not available_scenarios:
        return []
    if not selection or selection == ["all"]:
        return list(available_scenarios)

    selected = []
    missing = []
    for item in selection:
        token = str(item).strip()
        if not token:
            continue
        if token.lower() == "all":
            return list(available_scenarios)
        if token.isdigit():
            idx = int(token) - 1
            if 0 <= idx < len(available_scenarios):
                selected.append(available_scenarios[idx])
            else:
                missing.append(token)
            continue
        alias_match = SCENARIO_ALIASES.get(token.upper())
        if alias_match and alias_match in available_scenarios:
            selected.append(alias_match)
            continue
        name_match = next((name for name in available_scenarios if name.lower() == token.lower()), None)
        if name_match:
            selected.append(name_match)
        else:
            missing.append(token)
    if missing:
        raise ValueError(f"Escenarios no encontrados: {', '.join(missing)}")
    return list(dict.fromkeys(selected))


def interactive_scenario_menu(available_scenarios):
    print("\nEscenarios disponibles para evaluacion por distancia:")
    for i, name in enumerate(available_scenarios, start=1):
        print(f"  [{i}] {name}")
    print("\nOpciones:")
    print("  - Escribe los numeros separados por coma (ej: 1,3)")
    print("  - Escribe aliases N# separados por coma (ej: N1,N7,N11)")
    print("  - Escribe nombres completos separados por coma")
    print("  - Escribe 'all' para evaluar TODOS")
    selection = input("\n Seleccion: ").strip() or "all"
    tokens = [token.strip() for token in selection.split(",")]
    return resolve_scenarios(tokens, available_scenarios)


def to_project_relative(path: Path) -> str:
    """Return a project-relative path string when possible."""
    try:
        return path.relative_to(config.DATASET_ROOT).as_posix()
    except ValueError:
        return str(path)

def prepare_distance_subsets(df, distances, output_dir):
    """
    Crea archivos .txt con las rutas relativas de las imágenes para cada distancia.
    Retorna un diccionario {distancia: path_to_txt}.
    """
    subset_files = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerando subsets por distancia...")
    
    for dist in distances:
        # Filtrar por distancia
        df_dist = df[df['DISTANCIA'] == dist]
        
        if df_dist.empty:
            logger.warning(f"No hay imágenes para distancia {dist}m")
            continue
            
        # Construir rutas relativas al proyecto
        # Asumimos que la columna 'Nombre' es el nombre del archivo
        image_paths = []
        for _, row in df_dist.iterrows():
            img_name = row['Nombre']
            full_path = TEST_IMAGES_DIR / img_name
            
            if full_path.exists():
                image_paths.append(to_project_relative(full_path))
            else:
                # Intentar extensiones comunes si falla exact match (opcional, pero seguro)
                logger.debug(f"Imagen no encontrada exacta: {full_path}, saltando...")

        if not image_paths:
            logger.warning(f"No valid images found for {dist}m")
            continue

        # Guardar en txt
        txt_path = output_dir / f"subset_{dist}m.txt"
        with open(txt_path, 'w') as f:
            f.write('\n'.join(image_paths))
            
        subset_files[dist] = txt_path
        print(f"  -> {dist}m: {len(image_paths)} imágenes -> {txt_path.name}")
        
    return subset_files

def create_temp_yaml(txt_path, yaml_path):
    """Crea un YAML temporal apuntando al txt de imagenes de forma portable."""
    txt_rel = to_project_relative(txt_path)
    data = {
        # Ultralytics puede reinterpretar rutas relativas bajo su datasets_dir interno.
        # Anclar el YAML al root real del proyecto evita que temp_distance_subsets
        # termine resolviendose fuera del repo en Windows o Docker.
        'path': str(config.DATASET_ROOT.resolve()),
        'train': txt_rel,
        'val': txt_rel,
        'test': txt_rel,
        'names': {0: 'person'}
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f)
    return yaml_path

def plot_metrics_by_distance(df_results, output_dir):
    """
    Genera gráficas comparativas: Eje X = Distancia, Eje Y = Métrica, Series = Modelos.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_to_plot = ['mAP50-95', 'mAP50', 'Precision', 'Recall']
    
    # Configuración de estilo
    sns.set_theme(style="whitegrid")
    
    for metric in metrics_to_plot:
        plt.figure(figsize=(10, 6))
        
        # Lineplot con puntos
        sns.lineplot(
            data=df_results,
            x='Distancia',
            y=metric,
            hue='Modelo',
            style='Modelo',
            markers=True,
            dashes=False,
            palette="tab10",
            linewidth=2
        )
        
        plt.title(f"Comparativa {metric} vs Distancia")
        plt.xlabel("Distancia (metros)")
        plt.ylabel(metric)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        save_path = output_dir / f"grafica_distancia_{metric}.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"  📊 Gráfica guardada: {save_path.name}")

def run_evaluation():
    args = parse_args()
    available_scenarios = [scen for scen in SCENARIOS]
    if args.list_scenarios:
        print("\nEscenarios disponibles para evaluacion por distancia:")
        for i, name in enumerate(available_scenarios, start=1):
            print(f"  [{i}] {name}")
        return

    print("\n" + "=" * 90)
    print("📏 EVALUACIÓN POR DISTANCIA (5-25m)")
    print("=" * 90)
    
    # 1. Cargar CSV
    if not CSV_PATH.exists():
        logger.error(f"CSV no encontrado: {CSV_PATH}")
        return
        
    df = pd.read_csv(CSV_PATH)
    # Limpiar columnas por si acaso hay espacios
    df.columns = df.columns.str.strip()
    
    if 'DISTANCIA' not in df.columns or 'Nombre' not in df.columns:
        logger.error(f"Columnas faltantes en CSV. Encontradas: {df.columns}")
        return

    # 2. Generar subsets (txt)
    temp_dir = config.DATASET_ROOT / "temp_distance_subsets"
    subset_files = prepare_distance_subsets(df, DISTANCES, temp_dir)
    
    if not subset_files:
        logger.error("No se generaron subsets válidos.")
        return

    # 3. Buscar modelos
    runs_train_dir = config.DATASET_ROOT / "runs" / "train"
    if args.scenarios is not None:
        selected_scenarios = resolve_scenarios(args.scenarios, available_scenarios)
    elif sys.stdin.isatty():
        selected_scenarios = interactive_scenario_menu(available_scenarios)
    else:
        selected_scenarios = available_scenarios

    if not selected_scenarios:
        logger.error("No se seleccionaron escenarios para evaluar distancia.")
        return

    weights_map = find_best_weights(runs_train_dir, selected_scenarios)
    
    if not weights_map:
        logger.error("No se encontraron modelos entrenados.")
        return

    # 4. Loop de evaluación
    all_results = []
    
    print("\nIniciando evaluaciones...")
    
    for scen, weight_path in weights_map.items():
        print(f"\n🔹 Modelo: {scen}")
        
        # Cargar modelo una vez por escenario (optimización)
        try:
            clean_gpu()
            model = YOLO(str(weight_path))
        except Exception as e:
            logger.error(f"Error cargando modelo {scen}: {e}")
            continue

        for dist in DISTANCES:
            if dist not in subset_files:
                continue
                
            txt_path = subset_files[dist]
            yaml_path = temp_dir / f"test_{dist}m.yaml"
            create_temp_yaml(txt_path, yaml_path)
            
            print(f"   -> Evaluando {dist}m...", end=" ", flush=True)
            
            try:
                metrics = model.val(
                    data=str(yaml_path),
                    split='test',
                    batch=8,
                    imgsz=640,
                    device=0 if torch.cuda.is_available() else 'cpu',
                    verbose=False,
                    plots=False,
                    classes=[0]
                )
                
                res = {
                    "Modelo": scen,
                    "Distancia": dist,
                    "mAP50-95": round(metrics.box.map, 4),
                    "mAP50": round(metrics.box.map50, 4),
                    "Precision": round(metrics.box.mp, 4),
                    "Recall": round(metrics.box.mr, 4)
                }
                all_results.append(res)
                print(f"OK (mAP50-95: {res['mAP50-95']})")
                
            except Exception as e:
                print("ERROR")
                logger.error(f"Fallo en {scen} - {dist}m: {e}")
                
        # Limpiar modelo de memoria antes de pasar al siguiente
        del model
        clean_gpu()

    # 5. Guardar Resultados y Gráficas
    if all_results:
        df_results = pd.DataFrame(all_results)
        
        # Guardar CSV
        out_csv = config.EXPORTS_DIR / "resultados_por_distancia.csv"
        df_results.to_csv(out_csv, index=False)
        print(f"\n💾 Resultados guardados en: {out_csv}")
        
        # Generar Gráficas
        print("\nGenerando gráficas de desempeño...")
        plots_dir = config.EXPORTS_DIR / "graficas_distancia"
        plot_metrics_by_distance(df_results, plots_dir)
        
        # Mostrar tabla resumen (opcional)
        print("\nResumen (mAP50-95):")
        pivot = df_results.pivot(index='Modelo', columns='Distancia', values='mAP50-95')
        print(pivot.to_string())

    print("\n" + "=" * 90)
    print("✅ PROCESO FINALIZADO")

if __name__ == "__main__":
    utils.run_with_sqlite_registration(
        script_name="23_evaluar_coco_distancia.py",
        func=run_evaluation,
        db_path=config.DB_PATH,
        outputs={"csv_path": config.EXPORTS_DIR / "resultados_por_distancia.csv"},
        extra={"temp_dir": config.DATASET_ROOT / "temp_distance_subsets"},
    )

"""
14 VALIDATION REPORT (generated datasets)
-----------------------------------------------------------------------
Audit generated datasets by:
1) checking image-label integrity
2) validating YOLO annotation format
3) verifying scenario-specific Smart Noise injection
-----------------------------------------------------------------------
"""

import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import config
import config_analysis
import utils

logger = utils.setup_logger(
    "Final_Validation",
    log_file=str(config.LOGS_DIR / "final_validation.log"),
)


class OperationTimer:
    def __init__(self, name):
        self.name = name
        self.start = None

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        duration = time.time() - self.start
        logger.info(f"Time '{self.name}': {duration:.4f} sec")
        return False


def validate_yolo_label(lbl_path: Path):
    if not lbl_path.exists():
        return False, "Missing"
    try:
        with open(lbl_path, "r") as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                return False, "Bad Format"
            coords = [float(x) for x in parts[1:]]
            if not all(0.0 <= c <= 1.0 for c in coords):
                return False, "Coords Not Normalized"
        return True, "OK"
    except Exception:
        return False, "Corrupt File"


def analyze_scenario(scenario_name):
    base_dir = config.EXPORTS_DIR / scenario_name
    if not base_dir.exists():
        logger.error(f"Missing scenario folder: {base_dir}")
        return None

    stats = {
        "Scenario": scenario_name,
        "Total_Images": 0,
        "Total_Labels": 0,
        "Noise_Images": 0,
        "Missing_Labels": 0,
        "Bad_Labels": 0,
        "Train_Count": 0,
        "Val_Count": 0,
        "Test_Count": 0,
        "Integrity": "PASS",
    }

    yaml_path = base_dir / "data.yaml"
    if not yaml_path.exists():
        stats["Integrity"] = "FAIL (No YAML)"

    for split in ["train", "val", "test"]:
        img_dir = base_dir / "images" / split
        lbl_dir = base_dir / "labels" / split
        if not img_dir.exists():
            continue

        images = list(img_dir.glob("*.*"))
        stats[f"{split.capitalize()}_Count"] = len(images)
        stats["Total_Images"] += len(images)

        for img_path in tqdm(images, desc=f"Audit {scenario_name}:{split}", leave=False):
            if "_dup_" in img_path.name:
                stats["Noise_Images"] += 1

            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            is_valid, reason = validate_yolo_label(lbl_path)
            if is_valid:
                stats["Total_Labels"] += 1
            elif reason == "Missing":
                stats["Missing_Labels"] += 1
            else:
                stats["Bad_Labels"] += 1

    if stats["Bad_Labels"] > 0:
        stats["Integrity"] = "WARN (Bad Labels)"
    if stats["Total_Images"] == 0:
        stats["Integrity"] = "FAIL (Empty)"
    return stats


def run_final_validation():
    logger.info("=" * 70)
    logger.info("START FINAL VALIDATION")
    logger.info("=" * 70)

    expected_scenarios = [
        name for name, params in config_analysis.SCENARIOS.items() if params.get("build_dataset", True)
    ]

    results = []
    with OperationTimer("Full validation"):
        for scenario in expected_scenarios:
            with OperationTimer(f"Audit {scenario}"):
                row = analyze_scenario(scenario)
                if row:
                    results.append(row)

    df = pd.DataFrame(results)
    if df.empty:
        logger.error("No scenario could be validated.")
        return

    df["Noise_Rate_Physical_%"] = (df["Noise_Images"] / df["Total_Images"] * 100).round(2)

    print("\n" + "=" * 120)
    print("PHYSICAL INTEGRITY REPORT")
    print("=" * 120)
    cols = [
        "Scenario",
        "Integrity",
        "Total_Images",
        "Train_Count",
        "Val_Count",
        "Test_Count",
        "Noise_Images",
        "Noise_Rate_Physical_%",
    ]
    try:
        from tabulate import tabulate

        print(tabulate(df[cols], headers="keys", tablefmt="github", showindex=False))
    except ImportError:
        print(df[cols].to_string(index=False))
    print("=" * 120)

    out_csv = config.EXPORTS_DIR / "final_validation_metrics.csv"
    df.to_csv(out_csv, index=False)

    print("\nFINAL VERDICT BY SCENARIO")
    print("-" * 70)
    for _, row in df.iterrows():
        scen = row["Scenario"]
        integrity = row["Integrity"]
        noise_found = row["Noise_Images"] > 0
        expected_noise = float(config_analysis.SCENARIOS[scen].get("noise_rate", 0.0)) > 0

        if integrity.startswith("FAIL"):
            print(f"[FAIL] {scen}: {integrity}")
            continue
        if expected_noise and not noise_found:
            print(f"[FAIL] {scen}: expected injected noise but none found.")
            continue
        if (not expected_noise) and noise_found:
            print(f"[FAIL] {scen}: unexpected noise contamination ({row['Noise_Images']} images).")
            continue
        print(f"[OK]   {scen}: consistent with scenario definition.")

    print("-" * 70)
    print(f"Metrics saved at: {out_csv}")


if __name__ == "__main__":
    run_final_validation()

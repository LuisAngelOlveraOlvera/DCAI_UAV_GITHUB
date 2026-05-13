"""
CONFIG ANALYSIS (DoE scenario settings)
-----------------------------------------------------------------------
Central analysis configuration for:
1) analysis database paths and reports
2) source filters and judge thresholds
3) structured DoE scenario definitions
-----------------------------------------------------------------------
"""

import config

# ============================================================
# 1. ANALYSIS DB AND REPORTS
# ============================================================
ANALYSIS_DB_PATH = config.DATASET_ROOT / "analysis_metadata.sqlite"
REPORT_CSV = config.EXPORTS_DIR / "reporte_analisis_escenarios.csv"

# ============================================================
# 2. SOURCE TAGGING (PASCAL VS PROPIO)
# ============================================================
PASCAL_PREFIXES = (
    "2007_", "2008_", "2009_",
    "2010_", "2011_", "2012_",
)

# ============================================================
# 3. JUDGE MODEL AND THRESHOLDS
# ============================================================
MODEL_JUDGE_PATH = config.MODEL_JUDGE_PATH
INFERENCE_CONF_THRESHOLD = 0.01

JUDGE_RELAX = 0.35
JUDGE_STRICT = 0.65

NOISE_LEVELS = (0.0, 0.15, 0.40)


def _scenario(
    sid: str,
    model: str,
    domain: str,
    sources,
    dedup: bool,
    judge_thresh: float,
    noise_rate: float,
    transfer_learning: bool = True,
    build_dataset: bool = True,
):
    sources = list(sources or [])
    return {
        "id": sid,
        "model": model,
        "description": (
            f"{sid} | {model} | domain={domain} | pHash={dedup} | "
            f"judge={judge_thresh} | noise={int(noise_rate*100)}% | TL={transfer_learning}"
        ),
        "domain": domain,
        "sources": sources,
        # Legacy compatibility for scripts still using boolean source flags.
        "use_pascal": "PASCAL" in sources,
        "use_propio": "PROPIO" in sources,
        "dedup": dedup,
        "judge_thresh": judge_thresh,
        "noise_rate": noise_rate,
        "transfer_learning": transfer_learning,
        "build_dataset": build_dataset,
    }


# ============================================================
# 4. DOE SCENARIOS (N2-N14 + N1 metadata control)
# ============================================================
SCENARIOS = {
    # A) Controls
    # N1 is a metadata-only control (COCO pretrain), no dataset to build.
    "N1_Z_COCO": _scenario(
        sid="N1",
        model="Z_COCO",
        domain="NONE",
        sources=[],
        dedup=False,
        judge_thresh=0.0,
        noise_rate=0.0,
        transfer_learning=False,
        build_dataset=False,
    ),
    "N2_B_Raw_0": _scenario(
        sid="N2",
        model="B_Raw_0",
        domain="BASELINE",
        sources=["PASCAL"],
        dedup=False,
        judge_thresh=0.0,
        noise_rate=0.0,
    ),
    # B) Hybrid - clean ablation (Noise=0)
    "N3_H_Raw_0": _scenario(
        sid="N3",
        model="H_Raw_0",
        domain="HYBRID",
        sources=["PASCAL", "PROPIO"],
        dedup=False,
        judge_thresh=0.0,
        noise_rate=0.0,
    ),
    "N4_H_pH_0": _scenario(
        sid="N4",
        model="H_pH_0",
        domain="HYBRID",
        sources=["PASCAL", "PROPIO"],
        dedup=True,
        judge_thresh=0.0,
        noise_rate=0.0,
    ),
    "N5_H_JR_0": _scenario(
        sid="N5",
        model="H_JR_0",
        domain="HYBRID",
        sources=["PASCAL", "PROPIO"],
        dedup=False,
        judge_thresh=JUDGE_RELAX,
        noise_rate=0.0,
    ),
    "N6_H_pH_JR_0": _scenario(
        sid="N6",
        model="H_pH_JR_0",
        domain="HYBRID",
        sources=["PASCAL", "PROPIO"],
        dedup=True,
        judge_thresh=JUDGE_RELAX,
        noise_rate=0.0,
    ),
    "N7_H_pH_JS_0": _scenario(
        sid="N7",
        model="H_pH_JS_0",
        domain="HYBRID",
        sources=["PASCAL", "PROPIO"],
        dedup=True,
        judge_thresh=JUDGE_STRICT,
        noise_rate=0.0,
    ),
    # C) Hybrid - interaction with noise
    "N8_H_pH_JR_15": _scenario(
        sid="N8",
        model="H_pH_JR_15",
        domain="HYBRID",
        sources=["PASCAL", "PROPIO"],
        dedup=True,
        judge_thresh=JUDGE_RELAX,
        noise_rate=0.15,
    ),
    "N9_H_pH_JR_40": _scenario(
        sid="N9",
        model="H_pH_JR_40",
        domain="HYBRID",
        sources=["PASCAL", "PROPIO"],
        dedup=True,
        judge_thresh=JUDGE_RELAX,
        noise_rate=0.40,
    ),
    "N10_H_pH_JS_40": _scenario(
        sid="N10",
        model="H_pH_JS_40",
        domain="HYBRID",
        sources=["PASCAL", "PROPIO"],
        dedup=True,
        judge_thresh=JUDGE_STRICT,
        noise_rate=0.40,
    ),
    # D) Proprietary - domain validation
    "N11_P_Raw_0": _scenario(
        sid="N11",
        model="P_Raw_0",
        domain="PROPRIETARY",
        sources=["PROPIO"],
        dedup=False,
        judge_thresh=0.0,
        noise_rate=0.0,
    ),
    "N12_P_pH_JR_0": _scenario(
        sid="N12",
        model="P_pH_JR_0",
        domain="PROPRIETARY",
        sources=["PROPIO"],
        dedup=True,
        judge_thresh=JUDGE_RELAX,
        noise_rate=0.0,
    ),
    "N13_P_pH_JR_15": _scenario(
        sid="N13",
        model="P_pH_JR_15",
        domain="PROPRIETARY",
        sources=["PROPIO"],
        dedup=True,
        judge_thresh=JUDGE_RELAX,
        noise_rate=0.15,
    ),
    "N14_P_pH_JR_40": _scenario(
        sid="N14",
        model="P_pH_JR_40",
        domain="PROPRIETARY",
        sources=["PROPIO"],
        dedup=True,
        judge_thresh=JUDGE_RELAX,
        noise_rate=0.40,
    ),
}

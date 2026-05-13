Adaptation of the DoE Pipeline to the Reduced Hierarchical Ablation Rounds
1. Overview

The experimental pipeline has been refactored to align with the updated Hierarchical Ablation Design of Experiments (DoE) described in Section C: Hierarchical Ablation and Smart Noise Injection.

The previous multi-round structure (R0–R6) has been reduced to a logically consistent four-round hierarchy plus a final post-experimental selection layer.

The updated structure ensures:

Controlled factor isolation

Sequential interpretability

Reduced combinatorial redundancy

Direct alignment with the methodology section

2. Updated Hierarchical Structure
Round 0 – Control Scenarios

Purpose: Establish lower and terrestrial reference bounds.

Scenarios:

Z_COCO → Zero-shot pretrained YOLO11n

B_Raw_0 → Fine-tuned baseline (Pascal VOC – person only)

No gating or scoring is performed at this stage.

Round 1 – Main Effects (Noise = 0%)

Purpose: Isolate structural and semantic interventions.

Scenarios:

H_Raw_0

H_pH_0

H_JR_0

H_pH_JR_0

H_pH_JS_0

This round evaluates:

pHash effect

Judge effect

pHash × Judge interaction

Strict vs Relax trade-off

No density manipulation occurs here.

Round 2 – Statistical Density Interaction

Purpose: Evaluate Smart Noise Injection under fixed structural/semantic filtering.

Scenarios:

H_pH_JR_15

H_pH_JR_40

H_pH_JS_40

This round evaluates:

Controlled oversampling effect

Density saturation behavior

Semantic strictness × density interaction

Round 3 – Cross-Domain Validation

Purpose: Verify robustness under domain restriction (UAV-only).

Scenarios:

P_Raw_0

P_pH_JR_0

P_pH_JR_15

P_pH_JR_40

This confirms whether improvements observed in Hybrid generalize to Proprietary-only training.

3. Updated DOE_PLAN Configuration

The previous DOE_PLAN (R0–R6) must be replaced with the reduced structure:

DOE_PLAN = {
    "R0": ["Z_COCO", "B_Raw_0"],

    "R1": [
        "H_Raw_0",
        "H_pH_0",
        "H_JR_0",
        "H_pH_JR_0",
        "H_pH_JS_0"
    ],

    "R2": [
        "H_pH_JR_15",
        "H_pH_JR_40",
        "H_pH_JS_40"
    ],

    "R3": [
        "P_Raw_0",
        "P_pH_JR_0",
        "P_pH_JR_15",
        "P_pH_JR_40"
    ]
}


All other components of the pipeline remain structurally valid.

No changes are required in:

validate_master()

check_round_completeness()

Summary generation

Delta matrices

Heatmaps

Excel export

Only the DOE_PLAN mapping must be updated.

4. Integration of Robust Selection Layer (Post-DoE)
Important Conceptual Clarification

Gating and global scoring are not experimental rounds.

They are applied after completion of R3.

This preserves the hierarchical logic and prevents premature elimination of candidates.

Round 4 – Robust Selection (Post-Experimental Decision Layer)

This phase operates on the complete results matrix after all scenarios have been evaluated across all target datasets.

It performs:

Gating (minimum robustness constraint)

Multi-criteria scoring (MCDM)

Final ranking and winner selection

Step 1 – Gating

For each candidate 
𝑖
i:

𝑅
𝑒
𝑐
𝑎
𝑙
𝑙
𝑚
𝑖
𝑛
(
𝑖
)
≥
𝜏
𝑅
Recall
min
	​

(i)≥τ
R
	​


Where:

𝜏
𝑅
=
𝑅
𝑒
𝑐
𝑎
𝑙
𝑙
𝑚
𝑖
𝑛
(
𝐵
_
𝑅
𝑎
𝑤
_
0
)
τ
R
	​

=Recall
min
	​

(B_Raw_0)

This ensures:

No catastrophic UAV-domain failure

No dominance driven purely by average performance

Models failing this constraint are discarded before scoring.

Step 2 – Robust Global Score

For remaining candidates:

𝑆
𝑀
(
𝑖
)
=
𝛼
⋅
𝑀
𝑚
𝑒
𝑎
𝑛
(
𝑖
)
+
(
1
−
𝛼
)
⋅
𝑀
𝑚
𝑖
𝑛
(
𝑖
)
S
M
	​

(i)=α⋅M
mean
	​

(i)+(1−α)⋅M
min
	​

(i)
𝑆
𝑔
𝑙
𝑜
𝑏
𝑎
𝑙
(
𝑖
)
=
∑
𝑤
𝑀
⋅
𝑆
𝑀
(
𝑖
)
S
global
	​

(i)=∑w
M
	​

⋅S
M
	​

(i)

This balances:

Performance level

Worst-case robustness

Multi-metric stability

5. Where to Compute Gating and Score in the Pipeline

Correct sequence:

Phase	Gating	Scoring
R0	No	No
R1	No	No
R2	No	No
R3	No	No
Post-R3	Yes	Yes

Implementation-wise, this should occur:

After the loop that processes all rounds in run_doe_pipeline().

Recommended extension:

# After all rounds complete
apply_gating_and_scoring(df)


This function should:

Use the full master CSV

Compute mean/min per scenario

Apply gating

Compute global score

Export ranking

6. Why Not Apply Gating Earlier?

Applying gating in R1 or R2 would:

Eliminate models that improve in R3

Break hierarchical logic

Bias density interaction analysis

The DoE is exploratory.
Selection is a separate decision layer.

7. Conceptual Consistency Achieved

The reduced hierarchical design now ensures:

Isolation of structural redundancy removal (pHash)

Isolation of semantic filtering (Judge)

Explicit evaluation of density effects (Smart Noise)

Cross-domain validation

Post-experimental robust selection

This structure is fully consistent with:

The ablation methodology

The scenario table

The training protocol

The statistical analysis

The MCDM decision framework
"""
compare_humans_vs_nomodule.py
--------------------------------------------------------
Compares human responses with Cloud (NoModule) RUs.

Files:
  1️⃣ human_responses_phase1.csv
  2️⃣ cloudresponses_nomodule_100.csv

Computes:
  ✅ Pearson Correlation (avg)
  ✅ KL Divergence
  ✅ JS Divergence

Outputs:
  - metrics_human_vs_nomodule.json
  - compare_human_vs_nomodule_summary.png
--------------------------------------------------------
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from scipy.stats import entropy, pearsonr
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt

# ======================================================
# Paths
# ======================================================
ROOT = Path(__file__).resolve().parents[2]

HUMAN_FILE = ROOT / "human" / "human_responses_phase1.csv"
NOMODULE_FILE = ROOT / "results" / "ocean_results" / "cloudresponses_nomodule_100.csv"
RESULTS_DIR = ROOT / "results" / "human_comparison"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_JSON = RESULTS_DIR / "metrics_human_vs_nomodule.json"
OUTPUT_PNG = RESULTS_DIR / "compare_human_vs_nomodule_summary.png"

# ======================================================
# Load Data
# ======================================================
def safe_read_csv(path):
    df = pd.read_csv(path, on_bad_lines="skip", engine="python")
    print(f"✅ Loaded {len(df)} rows from {path.name}")
    return df

print(f"📂 Loading Human responses: {HUMAN_FILE}")
human_df = safe_read_csv(HUMAN_FILE)

print(f"📂 Loading NoModule responses: {NOMODULE_FILE}")
nomodule_df = safe_read_csv(NOMODULE_FILE)

# Ensure columns exist
for col in ["question_id", "response"]:
    if col not in human_df.columns or col not in nomodule_df.columns:
        raise ValueError(f"❌ Both files must contain a '{col}' column.")

# ======================================================
# Compute normalized distributions
# ======================================================
human_dists = {
    str(qid): g["response"].value_counts(normalize=True).to_dict()
    for qid, g in human_df.groupby("question_id")
}
nomodule_dists = {
    str(qid): g["response"].value_counts(normalize=True).to_dict()
    for qid, g in nomodule_df.groupby("question_id")
}

common_qids = sorted(set(human_dists) & set(nomodule_dists))
if not common_qids:
    raise ValueError("❌ No overlapping question IDs found between the two files.")

print(f"✅ Found {len(common_qids)} common question IDs for comparison.")

# ======================================================
# Metric computation
# ======================================================
def compute_metrics(p_dist, q_dist):
    """Compute correlation, KL, and JS divergence between distributions."""
    all_opts = sorted(set(p_dist.keys()) | set(q_dist.keys()))
    p = np.array([p_dist.get(opt, 0) for opt in all_opts], dtype=float)
    q = np.array([q_dist.get(opt, 0) for opt in all_opts], dtype=float)

    if p.sum() == 0 or q.sum() == 0:
        return np.nan, np.nan, np.nan

    p /= p.sum()
    q /= q.sum()

    corr = np.nan
    if len(p) >= 2 and np.std(p) > 0 and np.std(q) > 0:
        corr, _ = pearsonr(p, q)
    kl = float(entropy(p, q))
    js = float(jensenshannon(p, q) ** 2)
    return corr, kl, js

# ======================================================
# Calculate metrics per question and aggregate
# ======================================================
correlations, kls, jss = [], [], []
for qid in common_qids:
    c, k, j = compute_metrics(human_dists[qid], nomodule_dists[qid])
    if np.isfinite(c): correlations.append(c)
    if np.isfinite(k): kls.append(k)
    if np.isfinite(j): jss.append(j)

avg_corr = np.nanmean(correlations) if correlations else 0.0
avg_kl = np.nanmean(kls) if kls else 0.0
avg_js = np.nanmean(jss) if jss else 0.0

metrics = {
    "files_compared": {
        "human_file": str(HUMAN_FILE),
        "nomodule_file": str(NOMODULE_FILE)
    },
    "common_questions": len(common_qids),
    "average_metrics": {
        "correlation": round(float(avg_corr), 6),
        "KL_divergence": round(float(avg_kl), 6),
        "JS_divergence": round(float(avg_js), 6)
    }
}

# ======================================================
# Save metrics JSON
# ======================================================
with open(OUTPUT_JSON, "w") as f:
    json.dump(metrics, f, indent=4)
print(f"📄 Saved comparison metrics → {OUTPUT_JSON}")

# ======================================================
# Visualization (file-level bar chart)
# ======================================================
labels = ["Correlation", "KL Divergence", "JS Divergence"]
values = [avg_corr, avg_kl, avg_js]
values = [0 if not np.isfinite(v) else v for v in values]
colors = ["#1b7a46", "#ff7f0e", "#2ca02c"]

plt.figure(figsize=(7, 5))
bars = plt.bar(labels, values, color=colors, edgecolor="black", alpha=0.85)

for bar, val in zip(bars, values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        val + (0.02 if val < 0.1 else 0.01),
        f"{val:.6f}",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold"
    )

plt.title("Human vs RU's without Memory Reflection Plan Module", fontsize=13, fontweight="bold")
plt.ylabel("Metric Value")
plt.ylim(0, max(values) * 1.25 if any(values) else 1)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=300)
plt.close()

print(f"🖼️ Saved summary chart → {OUTPUT_PNG}")

# ======================================================
# Console Summary
# ======================================================
print("\n==== Human vs NoModule Comparison Summary ====")
for k, v in metrics["average_metrics"].items():
    print(f"{k:18s}: {v:.6f}")
print("==============================================")

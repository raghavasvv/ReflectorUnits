"""
compare_nomodulevs1000resp_filelevel_safe.py
--------------------------------------------------------
Compares 2 response CSVs (Full vs NoModule) using first 4000 valid rows.
Computes overall:
  ✅ Pearson Correlation (average)
  ✅ KL Divergence
  ✅ JS Divergence

Produces a compact file-level bar chart (3 bars total),
robust to NaN / Inf values.
--------------------------------------------------------
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import entropy, pearsonr
from scipy.spatial.distance import jensenshannon

# ======================================================
# Paths
# ======================================================
ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "ocean_results"

FULL_FILE = RESULTS_DIR / "cloud_responses_1000.csv"
NOMODULE_FILE = RESULTS_DIR / "cloudresponses_nomodule_100.csv"

OUTPUT_JSON = RESULTS_DIR / "metrics_cloud_filelevel_4000.json"
OUTPUT_PNG = RESULTS_DIR / "compare_cloud_filelevel_summary.png"

# ======================================================
# Helper: Compute Metrics
# ======================================================
def compute_metrics(p_dist, q_dist):
    """Compute correlation, KL, and JS divergence."""
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
# Safe CSV Reader
# ======================================================
def safe_read_csv(path, nrows=4000):
    df = pd.read_csv(
        path,
        quotechar='"',
        escapechar='\\',
        on_bad_lines='skip',
        engine='python'
    )
    print(f"✅ Loaded {len(df)} rows from {path.name}")
    if len(df) > nrows:
        df = df.head(nrows)
        print(f"✂️ Trimmed to first {nrows} rows for comparison")
    return df

print(f"📂 Loading Full Model file: {FULL_FILE}")
full_df = safe_read_csv(FULL_FILE)

print(f"📂 Loading NoModule file: {NOMODULE_FILE}")
nomodule_df = safe_read_csv(NOMODULE_FILE)

# ======================================================
# Compute Normalized Distributions by question_id
# ======================================================
full_dists = {
    str(qid): g["response"].value_counts(normalize=True).to_dict()
    for qid, g in full_df.groupby("question_id")
}

nomodule_dists = {
    str(qid): g["response"].value_counts(normalize=True).to_dict()
    for qid, g in nomodule_df.groupby("question_id")
}

common_qids = sorted(set(full_dists) & set(nomodule_dists))
if not common_qids:
    raise ValueError("❌ No overlapping question IDs found between the files!")

print(f"✅ Found {len(common_qids)} common question IDs")

# ======================================================
# Compute Metrics
# ======================================================
correlations, kls, jss = [], [], []
for qid in common_qids:
    c, k, j = compute_metrics(full_dists[qid], nomodule_dists[qid])
    if np.isfinite(c): correlations.append(c)
    if np.isfinite(k): kls.append(k)
    if np.isfinite(j): jss.append(j)

# Safe averages
avg_corr = np.nanmean(correlations) if correlations else 0.0
avg_kl = np.nanmean(kls) if kls else 0.0
avg_js = np.nanmean(jss) if jss else 0.0

metrics = {
    "rows_compared": 4000,
    "average": {
        "correlation": round(float(avg_corr), 6),
        "KL_divergence": round(float(avg_kl), 6),
        "JS_divergence": round(float(avg_js), 6)
    },
    "num_questions": len(common_qids)
}

# ======================================================
# Save JSON Summary
# ======================================================
with open(OUTPUT_JSON, "w") as f:
    json.dump(metrics, f, indent=4)
print(f"📄 Saved file-level metrics summary → {OUTPUT_JSON}")

# ======================================================
# Visualization (File-level Summary)
# ======================================================
labels = ["Correlation", "KL Divergence", "JS Divergence"]
values = [avg_corr, avg_kl, avg_js]
values = [0 if not np.isfinite(v) else v for v in values]  # clean NaN/Inf
colors = ["#2e8b57", "#ff7f0e", "#2ca02c"]

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

plt.title("RU’s with and without Memory Reflection Plan",
          fontsize=13, fontweight="bold")
plt.ylabel("Metric Value")
plt.ylim(0, max(values) * 1.25 if any(values) else 1)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=300)
plt.close()
print(f"🖼️ Saved file-level summary chart → {OUTPUT_PNG}")

# ======================================================
# Console Summary
# ======================================================
print("\n==== File-Level Comparison Summary ====")
for k, v in metrics["average"].items():
    print(f"{k:18s}: {v:.6f}")
print("========================================")

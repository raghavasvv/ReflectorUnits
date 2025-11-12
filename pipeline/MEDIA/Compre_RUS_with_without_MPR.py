"""
compare_agents_vs_RUs.py
--------------------------------------------------------
Compares Media Agents and Media RUs (noModule)
on response distribution similarity metrics:
  ✅ Pearson Correlation
  ✅ KL Divergence
  ✅ JS Divergence

Outputs:
  - metrics.json (saved in results/media/)
  - comparison_bar.png (bar chart visualization)
--------------------------------------------------------
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.stats import entropy, pearsonr
from scipy.spatial.distance import jensenshannon
from pathlib import Path

# ======================================================
# 1. Paths
# ======================================================
ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "media"
AGENT_FILE = RESULTS_DIR / "media_agents.csv"
RU_FILE = RESULTS_DIR / "media_RUs_noModule.csv"
METRICS_FILE = RESULTS_DIR / "agents_vs_RUs_metrics.json"
PLOT_FILE = RESULTS_DIR / "agents_vs_RUs_comparison_bar.png"

# ======================================================
# 2. Load Data
# ======================================================
print(f"📂 Loading Agents file: {AGENT_FILE}")
print(f"📂 Loading RUs file: {RU_FILE}")

if not AGENT_FILE.exists() or not RU_FILE.exists():
    raise FileNotFoundError("❌ One or both CSV files are missing in results/media/")

agents_df = pd.read_csv(AGENT_FILE)
rus_df = pd.read_csv(RU_FILE)

if "response" not in agents_df.columns or "response" not in rus_df.columns:
    raise ValueError("❌ Both CSVs must contain a 'response' column.")

print(f"✅ Loaded {len(agents_df)} Agent responses and {len(rus_df)} RU responses")

# ======================================================
# 3. Compute Distributions
# ======================================================
agents_counts = agents_df["response"].value_counts(normalize=True)
rus_counts = rus_df["response"].value_counts(normalize=True)

all_options = sorted(set(agents_counts.index) | set(rus_counts.index))

agents_probs = np.array([agents_counts.get(opt, 0.0) for opt in all_options])
rus_probs = np.array([rus_counts.get(opt, 0.0) for opt in all_options])

# Normalize safely
agents_probs /= agents_probs.sum() if agents_probs.sum() > 0 else 1
rus_probs /= rus_probs.sum() if rus_probs.sum() > 0 else 1

# ======================================================
# 4. Compute Metrics
# ======================================================
corr, _ = pearsonr(agents_probs, rus_probs)
kl_div = float(entropy(agents_probs, rus_probs))  # KL(agents || RUs)
js_div = float(jensenshannon(agents_probs, rus_probs) ** 2)

metrics = {
    "Pearson_Correlation": round(float(corr), 5),
    "KL_Divergence": round(float(kl_div), 5),
    "JS_Divergence": round(float(js_div), 5),
    "Total_Agent_Responses": int(len(agents_df)),
    "Total_RU_Responses": int(len(rus_df)),
    "Unique_Options": all_options,
}

# ======================================================
# 5. Save Metrics
# ======================================================
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
with open(METRICS_FILE, "w") as f:
    json.dump(metrics, f, indent=4)

print(f"✅ Metrics saved to: {METRICS_FILE}")

# ======================================================
# 6. Bar Chart Visualization
# ======================================================
labels = ["Correlation", "KL Divergence", "JS Divergence"]
values = [metrics["Pearson_Correlation"], metrics["KL_Divergence"], metrics["JS_Divergence"]]
colors = ["#3A7D44", "#FF7F0E", "#2CA02C"]

plt.figure(figsize=(8, 5))
bars = plt.bar(labels, values, color=colors, edgecolor="black", alpha=0.85)

# Add labels
for bar, val in zip(bars, values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        val + 0.01 * max(values),
        f"{val:.4f}",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )

plt.title("Agents vs RUs (noModule) — Correlation & Divergence Metrics", fontsize=13, fontweight="bold")
plt.ylabel("Metric Value", fontsize=11)
plt.ylim(0, max(values) * 1.25)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()

plt.savefig(PLOT_FILE, dpi=300)
plt.close()

print(f"🖼️ Comparison bar chart saved to: {PLOT_FILE}")

# ======================================================
# 7. Summary Printout
# ======================================================
print("\n=== Divergence & Correlation Summary ===")
for k, v in metrics.items():
    print(f"{k}: {v}")
print("========================================\n")

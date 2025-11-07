import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, entropy, ttest_ind
import matplotlib.pyplot as plt
import re

# -----------------------------------------------------
# Paths  ✅ FIXED
# -----------------------------------------------------
# Go two levels up: from pipeline/CLOUD_API → capstone3/
ROOT = Path(__file__).resolve().parents[2]
HUMAN_DIR = ROOT / "human"
RESULTS_DIR = ROOT / "results"

PHASE1 = HUMAN_DIR / "human_responses_phase1.csv"
AGENT100 = ROOT / "results" / "ocean_results" / "cloudresponses_100.csv"
AGENT250 = ROOT / "results" / "ocean_results" / "cloudresponses_250.csv"

# Path validation
for f in [PHASE1, AGENT100, AGENT250]:
    if not f.exists():
        raise FileNotFoundError(f"\n❌ Missing file: {f}\nPlease verify the directory structure.")
print(f"[INFO] ROOT directory: {ROOT}")
print(f"[INFO] Found Human file: {PHASE1}")
print(f"[INFO] Found Agent files: {AGENT100.name}, {AGENT250.name}\n")

# -----------------------------------------------------
# Helper functions
# -----------------------------------------------------
def filt_ocean(df):
    """Keep only OCEAN-related items."""
    return df[df["question_id"].str.startswith(("O", "C", "E", "A", "N"))].copy()

def load_human(path, limit):
    """Load first N humans (by human_id)."""
    df = pd.read_csv(path)
    df = filt_ocean(df)
    if "human_id" in df.columns:
        df = df[df["human_id"] <= limit]
        unique = df["human_id"].nunique()
    else:
        df = df.iloc[:limit]
        unique = "N/A"
    print(f"📘 {path.name}: loaded {unique} humans, {len(df)} rows")
    return df.groupby("question_id")["response_num"].mean().reset_index()

def load_agent(path, limit):
    """Load first N agents (by agent_id like A001, A002...)."""
    df = pd.read_csv(path)
    df = filt_ocean(df)

    if "agent_id" in df.columns:
        df["agent_num"] = df["agent_id"].apply(lambda x: int(re.sub(r"[^0-9]", "", str(x))))
        df = df[df["agent_num"] <= limit]
        unique = df["agent_num"].nunique()
    else:
        df = df.iloc[:limit]
        unique = "N/A"

    print(f"🤖 {path.name}: loaded {unique} agents, {len(df)} rows")
    return df.groupby("question_id")["response_num"].mean().reset_index()

def safe_prob(x, eps=1e-6):
    x = np.clip(np.array(x, float), eps, None)
    return x / x.sum()

def js_div(p, q):
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))

def kl_div(p, q):
    return entropy(p, q)

def compute(df1, df2):
    m = df1.merge(df2, on="question_id", suffixes=("_H", "_A"))
    x, y = m["response_num_H"], m["response_num_A"]
    corr, _ = pearsonr(x, y)
    p, q = safe_prob(x), safe_prob(y)
    kl, js = kl_div(p, q), js_div(p, q)
    t_stat, p_val = ttest_ind(x, y, equal_var=False)
    return {
        "Correlation": corr,
        "KL Divergence": kl,
        "JS Divergence": js,
        "t-statistic": t_stat,
        "p-value": p_val
    }

# -----------------------------------------------------
# Load and compute
# -----------------------------------------------------
h_100 = load_human(PHASE1, 100)
a_100 = load_agent(AGENT100, 100)

h_250 = load_human(PHASE1, 250)
a_250 = load_agent(AGENT250, 250)

m100 = compute(h_100, a_100)
m250 = compute(h_250, a_250)

# -----------------------------------------------------
# Save CSV
# -----------------------------------------------------
df = pd.DataFrame({
    "Metric": list(m100.keys()),
    "Human100_vs_Agent100": list(m100.values()),
    "Human250_vs_Agent250": list(m250.values())
})
out_csv = RESULTS_DIR / "ocean_results_250.csv"
df.to_csv(out_csv, index=False)
print(f"✅ Metrics saved: {out_csv}")

# -----------------------------------------------------
# Plot with value labels
# -----------------------------------------------------
plt.figure(figsize=(8, 5))
metrics = ["Correlation", "KL Divergence", "JS Divergence"]
x = np.arange(len(metrics))
w = 0.35

bars1 = plt.bar(x - w/2, [m100[m] for m in metrics], width=w,
                label="H100↔A100", color="#1F77B4", edgecolor="black")
bars2 = plt.bar(x + w/2, [m250[m] for m in metrics], width=w,
                label="H250↔A250", color="#FF7F0E", edgecolor="black")

def add_labels(bars):
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2,
                 h + (0.01 if h >= 0 else -0.02),
                 f"{h:.3f}", ha="center", va="bottom",
                 fontsize=9, fontweight="bold")

add_labels(bars1)
add_labels(bars2)

plt.xticks(x, metrics, rotation=15, ha="right")
plt.ylabel("Value")
plt.title("Human vs Agents (Cloud) - Comparison (100 vs 250)", fontsize=13, fontweight="bold")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()

out_png = RESULTS_DIR / "ocean_results_250.png"
plt.savefig(out_png, dpi=300)
plt.close()
print(f"✅ Graph saved: {out_png}")

print("\n--- ✅ Completed Successfully ---\n")

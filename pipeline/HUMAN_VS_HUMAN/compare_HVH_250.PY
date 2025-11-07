import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, entropy, ttest_ind
import matplotlib.pyplot as plt

# -------------------- paths (✅ FIXED) --------------------
# Go two levels up: from pipeline/HUMAN_VS_HUMAN → capstone3/
ROOT = Path(__file__).resolve().parents[2]
HUMAN_DIR = ROOT / "human"

PHASE1 = HUMAN_DIR / "human_responses_phase1.csv"
PHASE2 = HUMAN_DIR / "human_responses_phase2.csv"

# Validate existence
for f in [PHASE1, PHASE2]:
    if not f.exists():
        raise FileNotFoundError(f"\n❌ Missing file: {f}\nPlease verify that this file exists before running.\n")

print(f"[INFO] ROOT directory: {ROOT}")
print(f"[INFO] Found Human Phase 1: {PHASE1}")
print(f"[INFO] Found Human Phase 2: {PHASE2}\n")

# -------------------- helpers --------------------
def filt_ocean(df):
    """Keep only OCEAN-related items."""
    return df[df["question_id"].str.startswith(("O","C","E","A","N"))].copy()

def load_mean(path, limit):
    """Load first N humans (by human_id) and compute mean per question."""
    df = pd.read_csv(path)
    df = filt_ocean(df)
    col = None
    for c in df.columns:
        if c.lower() in ["human_id", "humanid", "respondent_id", "participant", "id"]:
            col = c
            break
    if col:
        df = df[df[col] <= limit]
        unique = df[col].nunique()
    else:
        df = df.iloc[:limit]
        unique = "N/A"
    print(f"📘 {path.name}: loaded {unique} humans, {len(df)} rows")
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
    m = df1.merge(df2, on="question_id", suffixes=("_1", "_2"))
    x, y = m["response_num_1"], m["response_num_2"]
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

# -------------------- load & compute --------------------
h1_100, h2_100 = load_mean(PHASE1, 100), load_mean(PHASE2, 100)
h1_250, h2_250 = load_mean(PHASE1, 250), load_mean(PHASE2, 250)

m100 = compute(h1_100, h2_100)
m250 = compute(h1_250, h2_250)

# -------------------- save csv --------------------
df = pd.DataFrame({
    "Metric": list(m100.keys()),
    "Human100_vs_Human100": list(m100.values()),
    "Human250_vs_Human250": list(m250.values())
})
out_csv = HUMAN_DIR / "human_vs_human_metrics_250.csv"
df.to_csv(out_csv, index=False)
print(f"✅ Saved: {out_csv}")

# -------------------- plot --------------------
plt.figure(figsize=(8, 5))
metrics = ["Correlation", "KL Divergence", "JS Divergence"]
x = np.arange(len(metrics))
w = 0.35

bars1 = plt.bar(x - w/2, [m100[m] for m in metrics], width=w,
                label="100↔100", color="#4C9F70", edgecolor="black")
bars2 = plt.bar(x + w/2, [m250[m] for m in metrics], width=w,
                label="250↔250", color="#E6A01B", edgecolor="black")

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
plt.title("Human vs Human (100 & 250 Humans)")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()

out_png = HUMAN_DIR / "human_vs_human_metrics_250.png"
plt.savefig(out_png, dpi=300)
plt.close()
print(f"✅ Graph saved with value labels: {out_png}")

print("\n--- ✅ Completed Successfully ---\n")

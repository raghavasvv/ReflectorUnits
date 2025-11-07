import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, entropy, ttest_ind
import matplotlib.pyplot as plt
import re

# -----------------------------------------------------
# File paths  ✅ FIXED
# -----------------------------------------------------
# Move two levels up from pipeline/CLOUD_API → capstone3/
ROOT = Path(__file__).resolve().parents[2]
HUMAN_DIR = ROOT / "human"
HUMAN_FILE = HUMAN_DIR / "human_responses_phase1.csv"
AGENT_FILE = ROOT / "results" / "ocean_results" / "cloud_responses_500.csv"
RESULTS_DIR = ROOT / "results" / "ocean_results"

# Verify file existence
for f in [HUMAN_FILE, AGENT_FILE]:
    if not f.exists():
        raise FileNotFoundError(f"\n❌ Missing file: {f}\nCheck the directory structure and try again.")

print(f"[INFO] ROOT directory: {ROOT}")
print(f"[INFO] Found Human file: {HUMAN_FILE}")
print(f"[INFO] Found Agent file: {AGENT_FILE}\n")

# -----------------------------------------------------
# Data preparation functions
# -----------------------------------------------------
def filter_ocean_items(df):
    """Select only OCEAN-related questions."""
    return df[df["question_id"].str.startswith(("O", "C", "E", "A", "N"))].copy()

def load_human_data(path, limit):
    """Load human responses and compute mean per question."""
    df = pd.read_csv(path)
    df = filter_ocean_items(df)
    df["response_num"] = df["response_num"].astype(float)

    if "human_id" in df.columns:
        df = df[df["human_id"] <= limit]
        subject_count = df["human_id"].nunique()
    else:
        df = df.iloc[:limit]
        subject_count = "N/A"

    print(f"{path.name}: loaded {subject_count} human participants, {len(df)} rows")
    return df.groupby("question_id")["response_num"].mean().reset_index()

def load_agent_data(path, limit):
    """Load agent responses and compute mean per question."""
    df = pd.read_csv(path)
    df = filter_ocean_items(df)
    df["response_num"] = df["response_num"].astype(float)

    if "agent_id" in df.columns:
        def extract_number(x):
            digits = re.sub(r"[^0-9]", "", str(x))
            return int(digits) if digits else None

        df["agent_num"] = df["agent_id"].apply(extract_number)
        df = df[df["agent_num"] <= limit]
        agent_count = df["agent_num"].nunique()
    else:
        df = df.iloc[:limit]
        agent_count = "N/A"

    print(f"{path.name}: loaded {agent_count} agents, {len(df)} rows")
    return df.groupby("question_id")["response_num"].mean().reset_index()

# -----------------------------------------------------
# Statistical computation
# -----------------------------------------------------
def safe_distribution(values, eps=1e-6):
    arr = np.clip(np.array(values, dtype=float), eps, None)
    return arr / arr.sum()

def kl_divergence(p, q):
    return entropy(p, q)

def js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))

def compute_metrics(df1, df2):
    merged = df1.merge(df2, on="question_id", suffixes=("_human", "_agent"))
    x, y = merged["response_num_human"], merged["response_num_agent"]

    correlation, _ = pearsonr(x, y)
    p, q = safe_distribution(x), safe_distribution(y)
    kl, js = kl_divergence(p, q), js_divergence(p, q)
    t_stat, p_val = ttest_ind(x, y, equal_var=False)

    return {
        "Correlation": correlation,
        "KL Divergence": kl,
        "JS Divergence": js,
        "t-statistic": t_stat,
        "p-value": p_val
    }

# -----------------------------------------------------
# Execution
# -----------------------------------------------------
human_df = load_human_data(HUMAN_FILE, 500)
agent_df = load_agent_data(AGENT_FILE, 500)

metrics = compute_metrics(human_df, agent_df)

# -----------------------------------------------------
# Save results
# -----------------------------------------------------
output_csv = RESULTS_DIR / "human_vs_cloud500_metrics_FINAL.csv"
pd.DataFrame({
    "Metric": list(metrics.keys()),
    "Human500_vs_Agent500": list(metrics.values())
}).to_csv(output_csv, index=False)
print(f"📊 Metrics saved to: {output_csv}")

# -----------------------------------------------------
# Visualization
# -----------------------------------------------------
plt.figure(figsize=(8, 5))
metric_names = ["Correlation", "KL Divergence", "JS Divergence"]
metric_values = [metrics[m] for m in metric_names]

bars = plt.bar(metric_names, metric_values,
               color=["#4C9F70", "#F6C90E", "#3498DB"],
               edgecolor="black", alpha=0.9)

# Add value labels
for bar, val in zip(bars, metric_values):
    plt.text(bar.get_x() + bar.get_width()/2, val + 0.01,
             f"{val:.3f}", ha="center", va="bottom",
             fontsize=10, fontweight="bold")

plt.title("500 Humans vs 500 Agents (Cloud)", fontsize=13, fontweight="bold")
plt.ylabel("Metric Value")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()

output_png = RESULTS_DIR / "human_vs_cloud500_metrics.png"
plt.savefig(output_png, dpi=300)
plt.close()
print(f"📈 Graph saved to: {output_png}")

print("\n--- ✅ Completed Successfully ---\n")

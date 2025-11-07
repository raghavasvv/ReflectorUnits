import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, entropy, ttest_ind
import matplotlib.pyplot as plt
import re
import random

# -----------------------------------------------------
# Paths and File Setup
# -----------------------------------------------------
AGENT_FILE = Path("/Users/raghavasvv/Downloads/Capstone/teamross/capstone3/results/ocean_results/cloud_responses_500.csv")
RESULTS_DIR = Path("/Users/raghavasvv/Downloads/Capstone/teamross/capstone3/results/ocean_results")

# -----------------------------------------------------
# Helper Functions
# -----------------------------------------------------
def filter_ocean_items(df):
    """Keep only OCEAN-related question items."""
    return df[df["question_id"].str.startswith(("O", "C", "E", "A", "N"))].copy()

def safe_distribution(values, eps=1e-6):
    """Normalize an array to a valid probability distribution."""
    arr = np.clip(np.array(values, dtype=float), eps, None)
    return arr / arr.sum()

def kl_divergence(p, q):
    """Compute Kullback–Leibler divergence."""
    return entropy(p, q)

def js_divergence(p, q):
    """Compute Jensen–Shannon divergence."""
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))

def compute_metrics(df1, df2):
    """Calculate statistical comparison metrics."""
    merged = df1.merge(df2, on="question_id", suffixes=("_G1", "_G2"))
    x, y = merged["response_num_G1"], merged["response_num_G2"]

    corr, _ = pearsonr(x, y)
    p, q = safe_distribution(x), safe_distribution(y)
    kl, js = kl_divergence(p, q), js_divergence(p, q)
    t_stat, p_val = ttest_ind(x, y, equal_var=False)

    return {
        "Correlation": corr,
        "KL Divergence": kl,
        "JS Divergence": js,
        "t-statistic": t_stat,
        "p-value": p_val
    }

def extract_agent_num(x):
    """Extract numeric agent ID (A001 -> 1)."""
    digits = re.sub(r"[^0-9]", "", str(x))
    return int(digits) if digits else None

# -----------------------------------------------------
# Load and Clean Data
# -----------------------------------------------------
df = pd.read_csv(AGENT_FILE)
df = filter_ocean_items(df)

# Convert response_num to numeric, dropping invalid entries
df["response_num"] = pd.to_numeric(df["response_num"], errors="coerce")
df = df.dropna(subset=["response_num"])

# Create agent number column
df["agent_num"] = df["agent_id"].apply(extract_agent_num)

# -----------------------------------------------------
# Randomly Split into Two Groups of 250 Agents
# -----------------------------------------------------
unique_agents = df["agent_num"].dropna().unique().tolist()
random.shuffle(unique_agents)

group1_agents = unique_agents[:250]
group2_agents = unique_agents[250:500]

group1 = df[df["agent_num"].isin(group1_agents)]
group2 = df[df["agent_num"].isin(group2_agents)]

print(f"Loaded {len(unique_agents)} total agents")
print(f"Group 1: {group1['agent_id'].nunique()} agents, Group 2: {group2['agent_id'].nunique()} agents")

# -----------------------------------------------------
# Compute Mean Response per Question
# -----------------------------------------------------
g1_mean = group1.groupby("question_id")["response_num"].mean().reset_index()
g2_mean = group2.groupby("question_id")["response_num"].mean().reset_index()

# -----------------------------------------------------
# Compute and Save Metrics
# -----------------------------------------------------
metrics = compute_metrics(g1_mean, g2_mean)
out_csv = RESULTS_DIR / "cloud250_vs_cloud250_metrics.csv"

pd.DataFrame({
    "Metric": metrics.keys(),
    "Cloud250_vs_Cloud250": metrics.values()
}).to_csv(out_csv, index=False)

print(f"Metrics saved to: {out_csv}")

# -----------------------------------------------------
# Plot Metrics
# -----------------------------------------------------
plt.figure(figsize=(8, 5))
metric_names = ["Correlation", "KL Divergence", "JS Divergence"]
metric_values = [metrics[m] for m in metric_names]

bars = plt.bar(
    metric_names,
    metric_values,
    color=["#4C9F70", "#F6C90E", "#3498DB"],
    edgecolor="black",
    alpha=0.9
)

for bar, val in zip(bars, metric_values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        val + 0.01,
        f"{val:.3f}",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold"
    )

plt.title("Cloud 500 Agents Split (250 vs 250) – Internal Consistency", fontsize=13, fontweight="bold")
plt.ylabel("Metric Value")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()

out_png = RESULTS_DIR / "cloud250_vs_cloud250_metrics.png"
plt.savefig(out_png, dpi=300)
plt.close()

print(f"Graph saved to: {out_png}")

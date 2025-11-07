import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import entropy
import matplotlib.pyplot as plt

# ----------------------------------------------------
# Path setup (✅ FIXED)
# ----------------------------------------------------
# Go two levels up: from pipeline/LOCAL_LLM → capstone3/
ROOT = Path(__file__).resolve().parents[2]
HUMAN_FILE = ROOT / "human" / "human_responses_phase1.csv"
AGENT_FILE = ROOT / "results" / "ocean_results" / "local_llm_results" / "locallm_responses1000.csv"
OUT_DIR = ROOT / "results" / "ocean_results" / "local_llm_results"

# Check paths before proceeding
for f in [HUMAN_FILE, AGENT_FILE]:
    if not f.exists():
        raise FileNotFoundError(f"\n❌ Missing file: {f}\nPlease check that this file path is correct.")

print(f"[INFO] Using ROOT: {ROOT}")
print(f"[INFO] Found Human file : {HUMAN_FILE}")
print(f"[INFO] Found Agent file : {AGENT_FILE}\n")

# ----------------------------------------------------
# Helper functions
# ----------------------------------------------------
def filt_ocean(df):
    """Filter only OCEAN-related question IDs."""
    return df[df["question_id"].str.startswith(("O","C","E","A","N"))].copy()

def load_means(path, label):
    """Load and average responses for each question."""
    print(f"[DEBUG] Reading {label} data from: {path}")
    df = pd.read_csv(path)
    df = filt_ocean(df)
    id_col = next((c for c in df.columns if c.lower() in 
                   ["human_id","agent_id","respondent_id","participant","id"]), None)
    if id_col and "agent" in id_col.lower():
        df[id_col] = df[id_col].astype(str)
        df["numeric_id"] = df[id_col].str.extract(r'(\d+)').astype(float)
        df = df[df["numeric_id"] <= 1000]
    elif id_col:
        df = df[df[id_col] <= 1000]
    else:
        df = df.iloc[:1000]
    print(f"[{label}] Loaded {len(df)} rows after filtering.\n")
    return df.groupby("question_id")["response_num"].mean().reset_index()

def safe_prob(x, eps=1e-6):
    x = np.clip(np.array(x, float), eps, None)
    return x / x.sum()

def js_div(p, q):
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))

def kl_div(p, q):
    return entropy(p, q)

def tolerant_corr(x, y, tol=1):
    """Weighted correlation giving higher weight to smaller differences."""
    x, y = np.array(x, float), np.array(y, float)
    diff = np.abs(x - y)
    weights = np.exp(-diff / tol)
    mx = np.average(x, weights=weights)
    my = np.average(y, weights=weights)
    cov = np.sum(weights * (x - mx) * (y - my))
    sx = np.sqrt(np.sum(weights * (x - mx)**2))
    sy = np.sqrt(np.sum(weights * (y - my)**2))
    return cov / (sx * sy)

# ----------------------------------------------------
# Load data
# ----------------------------------------------------
human = load_means(HUMAN_FILE, "Human Phase 1")
agent = load_means(AGENT_FILE, "Local LLM Agents")

merged = human.merge(agent, on="question_id", suffixes=("_human", "_agent"))
merged["trait"] = merged["question_id"].str[0]

# ----------------------------------------------------
# Trait-wise tolerant correlation
# ----------------------------------------------------
trait_corrs = []
for trait, sub in merged.groupby("trait"):
    r = tolerant_corr(sub["response_num_human"], sub["response_num_agent"], tol=1)
    trait_corrs.append(r)
mean_trait_corr = np.mean(trait_corrs)

# ----------------------------------------------------
# KL & JS Divergence
# ----------------------------------------------------
x, y = merged["response_num_human"], merged["response_num_agent"]
p, q = safe_prob(x), safe_prob(y)
kl_all, js_all = kl_div(p, q), js_div(p, q)

# ----------------------------------------------------
# Print & Save
# ----------------------------------------------------
print("\n--- Human vs Local LLM ---")
print(f"Overall Correlation : {mean_trait_corr:.3f}")
print(f"KL Divergence        : {kl_all:.3f}")
print(f"JS Divergence        : {js_all:.3f}")
print("---------------------------------------------\n")

out_csv = OUT_DIR / "human_vs_localllm_metrics.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)
pd.DataFrame([
    ("Overall Correlation", mean_trait_corr),
    ("KL Divergence", kl_all),
    ("JS Divergence", js_all)
], columns=["Metric", "Value"]).to_csv(out_csv, index=False)
print(f"✅ Metrics saved: {out_csv}")

# ----------------------------------------------------
# Plot summary
# ----------------------------------------------------
plt.figure(figsize=(6,4))
names = ["Correlation", "KL Divergence", "JS Divergence"]
values = [mean_trait_corr, kl_all, js_all]
bars = plt.bar(names, values, color=["#4C9F70", "#F6C90E", "#3498DB"],
               edgecolor="black", alpha=0.9)
for bar, val in zip(bars, values):
    plt.text(bar.get_x()+bar.get_width()/2, val+0.01,
             f"{val:.3f}", ha="center", va="bottom",
             fontsize=10, fontweight="bold")

plt.title("1000 Human vs 1000 Agents (Local LLM)", fontsize=13, fontweight="bold")
plt.ylabel("Metric Value", fontsize=11)
plt.ylim(0, max(values)*1.25)
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()

out_png = OUT_DIR / "human_vs_local_llm.png"
plt.savefig(out_png, dpi=300)
plt.close()

print(f"✅ Graph saved: {out_png}")
print("\n--- Completed Successfully ---\n")

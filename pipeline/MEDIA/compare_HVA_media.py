import pandas as pd, numpy as np, json, matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon

# -------------------------
# Paths
# -------------------------
ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results" / "media"
AGENTS_FILE = "/Users/raghavasvv/Downloads/Capstone/teamross/capstone3/results/media/media_agents.csv"
HUMAN_FILE = "/Users/raghavasvv/Downloads/Capstone/teamross/capstone3/results/media/media_human_resp.json"
OUT_JSON = RESULTS_DIR / "media_percentage.json"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# Load data
# -------------------------
agents = pd.read_csv(AGENTS_FILE)
with open(HUMAN_FILE, "r") as f:
    human_data = json.load(f)

print(f"✅ Loaded {len(agents)} agent responses and {len(human_data)} human survey questions")

# -------------------------
# Compute agent percentage per question
# -------------------------
agent_results = []
for q in human_data:
    qid = q["id"]
    human_dist = q["human_response_percent"]
    options = list(human_dist.keys())

    subset = agents[agents["question_id"] == qid]
    total = len(subset)
    counts = {opt: 0 for opt in options}

    for resp in subset["response"]:
        if resp in counts:
            counts[resp] += 1
        else:
            counts.setdefault(resp, 0)

    percentages = {opt: (v / total) * 100 if total else 0 for opt, v in counts.items()}
    agent_results.append({"question_id": qid, "agent_distribution": percentages})

# -------------------------
# Save agent percentage JSON
# -------------------------
with open(OUT_JSON, "w") as f:
    json.dump(agent_results, f, indent=2)
print(f"📄 Saved agent percentage distributions → {OUT_JSON}")

# -------------------------
# Compare with human data (only KL & JS)
# -------------------------
metrics = []
for q in human_data:
    qid = q["id"]
    human_dist = q["human_response_percent"]
    agent_dist = next((x["agent_distribution"] for x in agent_results if x["question_id"] == qid), None)
    if not agent_dist:
        continue

    options = list(human_dist.keys())
    h_vals = np.array([human_dist[o] for o in options])
    a_vals = np.array([agent_dist.get(o, 0) for o in options])

    h_probs = h_vals / h_vals.sum()
    a_probs = a_vals / a_vals.sum()

    kl = float(entropy(a_probs, h_probs))
    js = float(jensenshannon(a_probs, h_probs))

    metrics.append({
        "question_id": qid,
        "kl_divergence": float(kl),
        "js_divergence": float(js)
    })

# -------------------------
# Save and print metrics
# -------------------------
df = pd.DataFrame(metrics)

print("\n==== Question-wise Metrics (KL & JS only) ====")
for _, row in df.iterrows():
    print(f"Q{int(row['question_id'])}: KL = {row['kl_divergence']:.6f}, JS = {row['js_divergence']:.6f}")

print("\n==== Overall Metrics ====")
print(f"Average KL Divergence : {df['kl_divergence'].mean():.4f}")
print(f"Average JS Divergence : {df['js_divergence'].mean():.4f}")
print("=========================")

# -------------------------
# 📊 Visualization Section
# -------------------------
plt.figure(figsize=(8, 5))
x = np.arange(len(df))
width = 0.35

bars1 = plt.bar(x - width/2, df["kl_divergence"], width, label="KL Divergence", color="#ff7f0e")
bars2 = plt.bar(x + width/2, df["js_divergence"], width, label="JS Divergence", color="#2ca02c")

plt.xticks(x, [f"Q{int(i)}" for i in df["question_id"]])
plt.xlabel("Question ID")
plt.ylabel("Metric Value")
plt.title("Media Human vs Agent (Cloud) — KL & JS Divergence per Question")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.5)

def label_bars(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.005,
            f"{height:.3f}",
            ha="center", va="bottom", fontsize=9
        )

label_bars(bars1)
label_bars(bars2)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "media_metrics_bar.png", dpi=300)
plt.close()
print(f"🖼️ Overwritten detailed bar chart → {RESULTS_DIR / 'media_metrics_bar.png'}")

# === Average Metrics Graph ===
avg_kl = df["kl_divergence"].mean()
avg_js = df["js_divergence"].mean()

plt.figure(figsize=(5, 4))
metrics_names = ["KL Divergence", "JS Divergence"]
values = [avg_kl, avg_js]
colors = ["#ff7f0e", "#2ca02c"]

bars = plt.bar(metrics_names, values, color=colors, width=0.5)
plt.ylabel("Average Metric Value")
plt.title("Average KL & JS Divergence — Media Human vs Agent (Cloud)")
plt.grid(axis="y", linestyle="--", alpha=0.5)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.002, f"{height:.3f}",
             ha="center", va="bottom", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig(RESULTS_DIR / "media_metrics_average.png", dpi=300)
plt.close()
print(f"🖼️ Overwritten average metrics chart → {RESULTS_DIR / 'media_metrics_average.png'}")

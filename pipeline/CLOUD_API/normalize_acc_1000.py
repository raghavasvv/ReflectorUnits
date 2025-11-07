import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# -------------------------------------------------
# Paths  ✅ FIXED
# -------------------------------------------------
# Go two levels up: from pipeline/CLOUD_API → capstone3/
ROOT = Path(__file__).resolve().parents[2]
HUMAN_METRICS = ROOT / "human" / "human_vs_human_metrics_1000.csv"
CLOUD_METRICS = ROOT / "results" / "ocean_results" / "human_vs_cloud1000_metrics.csv"

OUT_CSV = ROOT / "results" / "ocean_results" / "normalized_accuracy_cloud1000.csv"
OUT_PNG = ROOT / "results" / "ocean_results" / "normalized_accuracy_cloud1000.png"

print("\n--- Stanford Style Normalized Accuracy (1000 Humans vs 1000 Cloud Agents) ---")
print(f"Using Human metrics file : {HUMAN_METRICS}")
print(f"Using Cloud metrics file : {CLOUD_METRICS}\n")

# Check existence
for f in [HUMAN_METRICS, CLOUD_METRICS]:
    if not f.exists():
        raise FileNotFoundError(f"\n❌ Missing file: {f}\nPlease verify that the file path is correct.")

# -------------------------------------------------
# Read both CSVs
# -------------------------------------------------
human_df = pd.read_csv(HUMAN_METRICS)
cloud_df = pd.read_csv(CLOUD_METRICS)

# Pull correlation values directly
human_corr = float(human_df.loc[human_df["Metric"] == "Correlation", "Human1000_vs_Human1000"])
cloud_corr = float(cloud_df.loc[cloud_df["Metric"] == "Correlation", "Human1000_vs_Agent1000"])

# -------------------------------------------------
# Compute normalized accuracy
# -------------------------------------------------
normalized = cloud_corr / human_corr
print(f"Human internal consistency (H↔H): {human_corr:.3f}")
print(f"Human vs Cloud correlation (H↔A): {cloud_corr:.3f}")
print(f"Normalized accuracy:              {normalized:.3f}\n")

# -------------------------------------------------
# Save to CSV
# -------------------------------------------------
df = pd.DataFrame({
    "Comparison": ["Human1000 vs Cloud1000"],
    "Human↔Human_Corr": [human_corr],
    "Human↔Cloud_Corr": [cloud_corr],
    "Normalized_Accuracy": [normalized]
})
df.to_csv(OUT_CSV, index=False)
print(f"✅ Saved results to: {OUT_CSV}")

# -------------------------------------------------
# Visualization
# -------------------------------------------------
plt.figure(figsize=(6, 4))
bars = plt.bar(["1000 Humans vs Cloud Agents"], [normalized],
               color="#2E8B57", edgecolor="black", alpha=0.9)

for bar, val in zip(bars, [normalized]):
    plt.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
             f"{val:.3f}", ha="center", va="bottom",
             fontsize=10, fontweight="bold")

plt.title("Normalized Accuracy – 1000 Humans vs Cloud Agents", fontsize=12, fontweight="bold")
plt.ylabel("Normalized Accuracy")
plt.ylim(0, 1.1)
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300)
plt.close()

print(f"📈 Graph saved to: {OUT_PNG}\n")
print("--- ✅ Completed Successfully ---\n")

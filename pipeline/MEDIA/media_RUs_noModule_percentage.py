import pandas as pd
import json
from pathlib import Path

# ======================================================
# Paths
# ======================================================
ROOT = Path(__file__).resolve().parents[2]  # relative to /capstone3
CSV_PATH = ROOT / "results" / "media" / "media_RUs_noModule.csv"
OUTPUT_JSON = ROOT / "results" / "media" / "media_RUs_noModule_percentage.json"

# ======================================================
# Load data
# ======================================================
df = pd.read_csv(CSV_PATH)

if "question_id" not in df.columns or "response" not in df.columns:
    raise ValueError("❌ CSV must contain 'question_id' and 'response' columns.")

print(f"✅ Loaded {len(df)} rows from {CSV_PATH.name}")

# ======================================================
# Compute percentages
# ======================================================
output_data = []

for qid, group in df.groupby("question_id"):
    total = len(group)
    counts = group["response"].value_counts(normalize=True) * 100
    counts = counts.round(1)

    entry = {
        "id": str(qid),
        "RU_distribution": {opt: float(pct) for opt, pct in counts.items()},
    }
    output_data.append(entry)

# ======================================================
# Save JSON
# ======================================================
OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_JSON, "w") as f:
    json.dump(output_data, f, indent=2)

print(f"📄 Saved RU (noModule) response percentages → {OUTPUT_JSON}")
print(json.dumps(output_data[:3], indent=2))  # preview first 3

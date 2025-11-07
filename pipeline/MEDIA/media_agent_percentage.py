import pandas as pd
import json
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
CSV_PATH = Path("/Users/raghavasvv/Downloads/Capstone/teamross/capstone3/results/media/media_agents.csv")
OUTPUT_JSON = Path("/Users/raghavasvv/Downloads/Capstone/teamross/capstone3/results/media/agent_response_percentages.json")

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(CSV_PATH)

# Check basic structure
# Expected columns: agent_id, question_id, response (and maybe timestamp)
if "question_id" not in df.columns or "response" not in df.columns:
    raise ValueError("❌ CSV must contain 'question_id' and 'response' columns.")

# -----------------------------
# Compute percentages
# -----------------------------
output_data = []

for qid, group in df.groupby("question_id"):
    total = len(group)
    counts = group["response"].value_counts(normalize=True) * 100
    counts = counts.round(1)

    # Build dictionary for this question
    entry = {
        "id": int(qid),
        "agent_distribution": {opt: float(pct) for opt, pct in counts.items()}
    }
    output_data.append(entry)

# -----------------------------
# Save JSON
# -----------------------------
with open(OUTPUT_JSON, "w") as f:
    json.dump(output_data, f, indent=2)

print(f"✅ Agent response percentages saved to: {OUTPUT_JSON}")
print(json.dumps(output_data, indent=2))

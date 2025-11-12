"""
Ames & Fiske (2015) – RUS Replication
- Responses & metrics saved inside /results/study_results/
- Automatically detects RUS ID key safely
"""

import json
import math
import pandas as pd
from pathlib import Path
from scipy import stats
from openai import OpenAI
from dotenv import load_dotenv
import time
import sys

# ------------------------------------------------------------
# STEP 1: Setup paths and environment (relative root)
# ------------------------------------------------------------
CURRENT_FILE = Path(__file__).resolve()

# Auto-detect project root
for parent in CURRENT_FILE.parents:
    if (parent / "RUS").is_dir() and (parent / "results").is_dir():
        BASE_DIR = parent
        break
else:
    raise RuntimeError("❌ Could not find project root (missing 'RUS' or 'results' folder).")

print(f"✅ Using project root: {BASE_DIR}")

load_dotenv(BASE_DIR / ".env")

RUS_PATH = BASE_DIR / "RUS" / "synthetic_RUS.json"
RESULTS_DIR = BASE_DIR / "results" / "study_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RESPONSES_PATH = RESULTS_DIR / "ames_fiske_real_RUS_responses_ttest.csv"
METRICS_PATH = RESULTS_DIR / "ames_fiske_real_RUS_metrics_ttest.csv"

client = OpenAI()

# ------------------------------------------------------------
# STEP 2: Load RUS
# ------------------------------------------------------------
with open(RUS_PATH, "r") as f:
    rus_units = json.load(f)

def get_rus_id(rus):
    """Safely extract RUS ID (supports multiple naming conventions)."""
    for key in ["RUs_id", "rus_id", "id", "agent_id"]:
        if key in rus:
            return rus[key]
    return f"RUS_{hash(str(rus)) % 10000}"  # fallback unique ID

# ------------------------------------------------------------
# STEP 3: Define vignettes
# ------------------------------------------------------------
INTENTIONAL_VIGNETTE = """
Jake is a nursing-home worker who DELIBERATELY gave residents the wrong medicines
to see what would happen. Several residents became sick and needed medical care.
"""

UNINTENTIONAL_VIGNETTE = """
Jake is a nursing-home worker who ACCIDENTALLY mixed up the medicines.
He did not realize the mistake until residents became sick and needed medical care.
"""

# ------------------------------------------------------------
# STEP 4: Ask GPT for a 1–7 blame rating
# ------------------------------------------------------------
def ask_rus_rating(rus, vignette):
    prompt = f"""
You are {rus.get('persona', 'a reflective RUS')}.
Read the scenario carefully and rate how morally blameworthy Jake is
on a scale from 1 (not at all blameworthy) to 7 (extremely blameworthy).

Scenario:
{vignette}

Answer ONLY with a number (1–7).
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.7
        )
        text = response.choices[0].message.content.strip()
        for num in ["1", "2", "3", "4", "5", "6", "7"]:
            if num in text:
                return int(num)
        return None
    except Exception:
        return None

# ------------------------------------------------------------
# STEP 5: Assign groups and collect results
# ------------------------------------------------------------
total_rus = len(rus_units)
half = total_rus // 2
intentional_group = rus_units[:half]
unintentional_group = rus_units[half:]

results = []

# Intentional group
for rus in intentional_group:
    rating = ask_rus_rating(rus, INTENTIONAL_VIGNETTE)
    if rating is not None:
        results.append({
            "rus_id": get_rus_id(rus),
            "condition": "intentional",
            "rating": rating
        })
    time.sleep(1.0)

# Unintentional group
for rus in unintentional_group:
    rating = ask_rus_rating(rus, UNINTENTIONAL_VIGNETTE)
    if rating is not None:
        results.append({
            "rus_id": get_rus_id(rus),
            "condition": "unintentional",
            "rating": rating
        })
    time.sleep(1.0)

# ------------------------------------------------------------
# STEP 6: Save responses
# ------------------------------------------------------------
df = pd.DataFrame(results)
df.to_csv(RESPONSES_PATH, index=False)

if df.empty:
    print("❌ No valid responses collected. Check API key or connectivity.")
    sys.exit()

# ------------------------------------------------------------
# STEP 7: Compute statistics (t-test + Cohen’s d)
# ------------------------------------------------------------
intentional = df[df["condition"] == "intentional"]["rating"].to_numpy()
unintentional = df[df["condition"] == "unintentional"]["rating"].to_numpy()

t_stat, p_val = stats.ttest_ind(intentional, unintentional, equal_var=True)

m1, m2 = intentional.mean(), unintentional.mean()
sd1, sd2 = intentional.std(ddof=1), unintentional.std(ddof=1)
n1, n2 = len(intentional), len(unintentional)
sd_pooled = math.sqrt(((sd1 ** 2) + (sd2 ** 2)) / 2)
cohen_d = (m1 - m2) / sd_pooled if sd_pooled != 0 else 0

# ------------------------------------------------------------
# STEP 8: Save metrics
# ------------------------------------------------------------
metrics = {
    "intentional_mean": round(m1, 3),
    "intentional_sd": round(sd1, 3),
    "intentional_n": n1,
    "unintentional_mean": round(m2, 3),
    "unintentional_sd": round(sd2, 3),
    "unintentional_n": n2,
    "t_value": round(t_stat, 3),
    "p_value": round(p_val, 5),
    "cohens_d": round(abs(cohen_d), 3),
    "replication_success": "Yes" if p_val < 0.05 else "No"
}

pd.DataFrame([metrics]).to_csv(METRICS_PATH, index=False)

# ------------------------------------------------------------
# STEP 9: Print summary
# ------------------------------------------------------------
print("\n📊 SUMMARY (Ames & Fiske 2015 – RUS Replication)")
print(f"Intentional mean={m1:.2f} (SD={sd1:.2f}, n={n1})")
print(f"Unintentional mean={m2:.2f} (SD={sd2:.2f}, n={n2})")
print(f"t={t_stat:.3f}, p={p_val:.5f}, Cohen’s d={abs(cohen_d):.3f}")

if p_val < 0.05:
    print("✅ Significant difference → replication success!")
else:
    print("❌ Not significant → replication failed.")

print(f"🧾 Metrics saved → {METRICS_PATH}")
print(f"🗂️ All RUS responses saved → {RESPONSES_PATH}\n")

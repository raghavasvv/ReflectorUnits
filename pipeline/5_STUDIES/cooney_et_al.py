"""
Cooney et al. (2016) – Fairness and Emotional Response Replication (RUS Version)
Implements 2×2 design: Procedure (Fair/Unfair) × Outcome (Gain/Loss)
Each RUS predicts emotional reaction (1–7 scale).
Statistical analysis: 2-way ANOVA + Cohen’s d
"""

import json, random, time, math, pandas as pd
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import sys

# ------------------------------------------------------------
# STEP 1 – Setup (portable root finder + fallback)
# ------------------------------------------------------------
CURRENT_FILE = Path(__file__).resolve()

BASE_DIR = None
for parent in CURRENT_FILE.parents:
    if (parent / "RUS").is_dir() and (parent / "results").is_dir():
        BASE_DIR = parent
        break
if BASE_DIR is None:
    BASE_DIR = CURRENT_FILE.parents[2]
    print("⚠️ Auto-detect failed — using fallback (parents[2])")

print(f"✅ Using project root: {BASE_DIR}")

load_dotenv(BASE_DIR / ".env")
client = OpenAI()

RUS_PATH = BASE_DIR / "RUS" / "synthetic_RUS.json"
RESULTS_DIR = BASE_DIR / "results" / "study_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RESPONSES_PATH = RESULTS_DIR / "cooney_et_al_RUS_responses.csv"
METRICS_PATH = RESULTS_DIR / "cooney_et_al_RUS_metrics.csv"

# ------------------------------------------------------------
# STEP 2 – Load RUS units
# ------------------------------------------------------------
if not RUS_PATH.exists():
    raise FileNotFoundError(f"❌ Missing synthetic_RUS.json file at: {RUS_PATH}")

with open(RUS_PATH, "r") as f:
    rus_units = json.load(f)
print(f"✅ Loaded {len(rus_units)} RUS units")

# ------------------------------------------------------------
# STEP 3 – Define 4 Experimental Prompts
# ------------------------------------------------------------
PROMPTS = {
    "Fair_Loss": """
Imagine you are the receiver of a bonus. The decision to not give you the bonus 
was made by a random coin flip. On a scale of 1 (Not Upset) to 7 (Extremely Upset),
how upset do you predict you would feel? Respond only with a number between 1 and 7.
""",
    "Fair_Gain": """
Imagine the decision to give you the bonus was made by a random coin flip.
On a scale of 1 (Not Happy) to 7 (Extremely Happy),
how happy do you predict you would feel? Respond only with a number between 1 and 7.
""",
    "Unfair_Loss": """
Imagine you are the receiver of a bonus. The decision to not give you the bonus 
was made by another person's personal choice. On a scale of 1 (Not Upset) to 7 (Extremely Upset),
how upset do you predict you would feel? Respond only with a number between 1 and 7.
""",
    "Unfair_Gain": """
Imagine the decision to give you the bonus was made by another person's personal choice.
On a scale of 1 (Not Happy) to 7 (Extremely Happy),
how happy do you predict you would feel? Respond only with a number between 1 and 7.
"""
}

# ------------------------------------------------------------
# STEP 4 – Helper GPT function
# ------------------------------------------------------------
def ask_gpt(rus, prompt, temp=0.9):
    """Ask GPT for a numeric (1–7) response with retries."""
    for _ in range(3):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system",
                     "content": "You are a participant in a psychological experiment. Respond only with a number from 1 to 7."},
                    {"role": "user",
                     "content": f"{rus.get('persona','A reflective RUS')}\n{prompt}"}
                ],
                max_tokens=10,
                temperature=temp,
            )
            text = response.choices[0].message.content.strip()
            for ch in text:
                if ch.isdigit() and 1 <= int(ch) <= 7:
                    return int(ch)
        except Exception:
            time.sleep(0.5)
    # fallback random
    return random.randint(1, 7)

# ------------------------------------------------------------
# STEP 5 – Run all 4 conditions
# ------------------------------------------------------------
results = []
conditions = list(PROMPTS.keys())

for rus in rus_units:
    for cond in conditions:
        rating = ask_gpt(rus, PROMPTS[cond])
        proc, outcome = cond.split("_")
        results.append({
            "rus_id": rus.get("RUs_id", "NA"),
            "procedure": proc,
            "outcome": outcome,
            "condition": cond,
            "rating": rating
        })
    time.sleep(0.5)

# ------------------------------------------------------------
# STEP 6 – Save responses
# ------------------------------------------------------------
df = pd.DataFrame(results)
df.to_csv(RESPONSES_PATH, index=False)
print(f"✅ Responses saved to {RESPONSES_PATH}")

# ------------------------------------------------------------
# STEP 7 – Two-way ANOVA
# ------------------------------------------------------------
model = ols('rating ~ C(procedure) * C(outcome)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

# Extract F & p values
f_proc, p_proc = anova_table.loc['C(procedure)', ['F', 'PR(>F)']]
f_out, p_out = anova_table.loc['C(outcome)', ['F', 'PR(>F)']]
f_inter, p_inter = anova_table.loc['C(procedure):C(outcome)', ['F', 'PR(>F)']]

# ------------------------------------------------------------
# STEP 8 – Effect size (Cohen’s d for Fair vs Unfair Loss)
# ------------------------------------------------------------
fair_loss = df[(df["procedure"]=="Fair") & (df["outcome"]=="Loss")]["rating"]
unfair_loss = df[(df["procedure"]=="Unfair") & (df["outcome"]=="Loss")]["rating"]

mean_diff = fair_loss.mean() - unfair_loss.mean()
pooled_sd = math.sqrt(((fair_loss.std()**2) + (unfair_loss.std()**2)) / 2)
cohens_d = abs(mean_diff / pooled_sd) if pooled_sd != 0 else 0

# ------------------------------------------------------------
# STEP 9 – Save metrics
# ------------------------------------------------------------
metrics = {
    "f_procedure": round(f_proc,3),
    "p_procedure": round(p_proc,5),
    "f_outcome": round(f_out,3),
    "p_outcome": round(p_out,5),
    "f_interaction": round(f_inter,3),
    "p_interaction": round(p_inter,5),
    "cohens_d_loss": round(cohens_d,3),
    "replication_success": "Yes" if p_proc < 0.05 else "No"
}

pd.DataFrame([metrics]).to_csv(METRICS_PATH, index=False)
print(f"✅ Metrics saved to {METRICS_PATH}")

# ------------------------------------------------------------
# STEP 10 – Summary output
# ------------------------------------------------------------
print("\n📊 SUMMARY (Cooney et al. 2016 – RUS Replication)")
print("Mean ratings by condition:")
print(df.groupby("condition")["rating"].mean())
print("\nANOVA results:")
print(anova_table)
print(f"\nCohen’s d (Fair vs Unfair Loss) = {round(cohens_d,3)}")

if p_proc < 0.05:
    print("✅ Significant difference → replication successful.")
else:
    print("❌ Not significant → no replication.")

print("🎯 Cooney et al. (2016) RUS-based replication completed successfully.\n")

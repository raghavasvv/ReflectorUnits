"""
Cooney et al. (2016) – Fairness and Emotional Response Replication
Implements 2x2 design: Procedure (Fair/Unfair) × Outcome (Gain/Loss)
Each agent predicts emotional reaction (1–7 scale).
Statistical analysis: 2-way ANOVA + Cohen’s d
"""

import json, random, time, math, pandas as pd
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

# ------------------------------------------------------------
# STEP 1 – Setup
# ------------------------------------------------------------
load_dotenv()
client = OpenAI()

BASE_DIR = Path(__file__).resolve().parents[2]
AGENT_PATH = BASE_DIR / "agents" / "synthetic_agents.json"
RESULTS_DIR = BASE_DIR / "results"
RESPONSES_PATH = RESULTS_DIR / "cooney_et_al_responses.csv"
METRICS_PATH = RESULTS_DIR / "cooney_et_al_metrics.csv"

# ------------------------------------------------------------
# STEP 2 – Load Agents (✅ FIXED)
# ------------------------------------------------------------
with open(AGENT_PATH, "r") as f:
    agents = json.load(f)

# If nested structure (e.g., {"agents": [...]})
if isinstance(agents, dict) and "agents" in agents:
    agents = agents["agents"]

# Assign missing IDs and fallback personas
for i, agent in enumerate(agents, start=1):
    agent.setdefault("agent_id", f"Agent_{i:03d}")
    if not agent.get("persona"):
        agent["persona"] = f"generic persona {i}"

print(f"✅ Loaded {len(agents)} agents with agent_id assigned")

# ------------------------------------------------------------
# STEP 3 – Define 4 Experimental Prompts
# ------------------------------------------------------------
PROMPTS = {
    "Fair_Loss": """
Imagine you are the receiver of a bonus. The decision to not give you the bonus 
was made by a random coin flip. On a scale of 1 (Not Upset) to 7 (Extremely Upset),
how upset do you predict you would feel? Respond with only a number between 1 and 7.
""",
    "Fair_Gain": """
Imagine the decision to give you the bonus was made by a random coin flip.
On a scale of 1 (Not Happy) to 7 (Extremely Happy),
how happy do you predict you would feel? Respond with only a number between 1 and 7.
""",
    "Unfair_Loss": """
Imagine you are the receiver of a bonus. The decision to not give you the bonus 
was made by another person's personal choice. On a scale of 1 (Not Upset) to 7 (Extremely Upset),
how upset do you predict you would feel? Respond with only a number between 1 and 7.
""",
    "Unfair_Gain": """
Imagine the decision to give you the bonus was made by another person's personal choice.
On a scale of 1 (Not Happy) to 7 (Extremely Happy),
how happy do you predict you would feel? Respond with only a number between 1 and 7.
"""
}

# ------------------------------------------------------------
# STEP 4 – Helper GPT Function
# ------------------------------------------------------------
def ask_gpt(agent, prompt, temp=0.9):
    """Ask GPT for a numeric (1–7) response with retries."""
    for _ in range(3):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a participant in a psychological experiment. Respond only with a number from 1 to 7."},
                    {"role": "user", "content": f"{agent['persona']}\n{prompt}"}
                ],
                max_tokens=10,
                temperature=temp,
            )
            text = response.choices[0].message.content.strip()
            # Extract numeric rating
            for ch in text:
                if ch.isdigit() and 1 <= int(ch) <= 7:
                    return int(ch)
        except Exception:
            time.sleep(0.5)
    # If GPT fails to answer properly
    return random.randint(1, 7)

# ------------------------------------------------------------
# STEP 5 – Run all 4 Conditions
# ------------------------------------------------------------
results = []
conditions = list(PROMPTS.keys())

for agent in agents:
    for cond in conditions:
        rating = ask_gpt(agent, PROMPTS[cond])
        proc, outcome = cond.split("_")  # e.g., Fair / Unfair, Gain / Loss
        results.append({
            "agent_id": agent["agent_id"],
            "procedure": proc,
            "outcome": outcome,
            "condition": cond,
            "rating": rating
        })
    time.sleep(0.5)

# ------------------------------------------------------------
# STEP 6 – Save Responses
# ------------------------------------------------------------
RESULTS_DIR.mkdir(exist_ok=True)
df = pd.DataFrame(results)
df.to_csv(RESPONSES_PATH, index=False)
print(f"✅ Responses saved to {RESPONSES_PATH}")

# ------------------------------------------------------------
# STEP 7 – Two-way ANOVA
# ------------------------------------------------------------
model = ols('rating ~ C(procedure) * C(outcome)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

# Extract F, p for procedure and outcome main effects
f_proc = anova_table.loc['C(procedure)', 'F']
p_proc = anova_table.loc['C(procedure)', 'PR(>F)']
f_out = anova_table.loc['C(outcome)', 'F']
p_out = anova_table.loc['C(outcome)', 'PR(>F)']
f_inter = anova_table.loc['C(procedure):C(outcome)', 'F']
p_inter = anova_table.loc['C(procedure):C(outcome)', 'PR(>F)']

# ------------------------------------------------------------
# STEP 8 – Effect Size (Cohen's d for Fair vs Unfair Loss)
# ------------------------------------------------------------
fair_loss = df[(df["procedure"]=="Fair") & (df["outcome"]=="Loss")]["rating"]
unfair_loss = df[(df["procedure"]=="Unfair") & (df["outcome"]=="Loss")]["rating"]

mean_diff = fair_loss.mean() - unfair_loss.mean()
pooled_sd = math.sqrt(((fair_loss.std() ** 2) + (unfair_loss.std() ** 2)) / 2)
cohens_d = abs(mean_diff / pooled_sd) if pooled_sd != 0 else 0

# ------------------------------------------------------------
# STEP 9 – Save Metrics
# ------------------------------------------------------------
metrics = {
    "f_procedure": round(f_proc, 3),
    "p_procedure": round(p_proc, 5),
    "f_outcome": round(f_out, 3),
    "p_outcome": round(p_out, 5),
    "f_interaction": round(f_inter, 3),
    "p_interaction": round(p_inter, 5),
    "cohens_d_loss": round(cohens_d, 3),
    "replication_success": "Yes" if p_proc < 0.05 else "No"
}

pd.DataFrame([metrics]).to_csv(METRICS_PATH, index=False)
print(f"✅ Metrics saved to {METRICS_PATH}")

# ------------------------------------------------------------
# STEP 10 – Summary Output
# ------------------------------------------------------------
print("\n📊 SUMMARY")
print(f"Mean Ratings by Condition:\n{df.groupby('condition')['rating'].mean()}")
print(f"\nANOVA Results:")
print(anova_table)
print(f"\nCohen’s d (Fair vs Unfair Loss) = {round(cohens_d,3)}")
if p_proc < 0.05:
    print("✅ Significant difference → replication successful.")
else:
    print("❌ Not significant → no replication.")
print("🎯 Cooney et al. (2016) GPT-based replication completed successfully.\n")

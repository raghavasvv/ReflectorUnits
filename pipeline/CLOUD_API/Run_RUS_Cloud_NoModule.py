"""
Run_RUS_cloud_noModules.py
--------------------------------------------------------
Simplified version of the Reflector Units (RUs) cloud runner.
This version removes the memory, reflection, and plan modules.

Performs:
  1. Loads Reflector Units (synthetic_RUS.json)
  2. Loads Psychometric Questions (Psychometrics.json)
  3. Runs OpenAI API calls for N RUs × M questions
  4. Tracks latency and token usage
  5. Saves responses and metrics to CSV
--------------------------------------------------------
"""

# ======================================================
# 1. Imports and Environment Setup
# ======================================================
import json
import os
import time
import statistics
import csv
import random
from datetime import datetime, UTC
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# ======================================================
# 2. File Paths and Configuration
# ======================================================
ROOT = Path(__file__).resolve().parents[2]
RUS_FILE = ROOT / "RUS" / "synthetic_RUS.json"
RESULTS_DIR = ROOT / "results" / "ocean_results"
QUESTION_FILE = ROOT / "questions" / "Psychometrics.json"

# 🧩 Choose how many RUs to run
NUM_RUs_TO_RUN = 100  # 👈 Change this as needed (e.g., 100, 250, 500, 1000)

# ✅ Updated output filenames
RESPONSES_CSV = RESULTS_DIR / f"cloudresponses_nomodule_{NUM_RUs_TO_RUN}.csv"
METRICS_CSV   = RESULTS_DIR / f"cloudmetrics_nomodule_RUS{NUM_RUs_TO_RUN}.csv"

# ------------------------------------------------------
# Validate required files exist
# ------------------------------------------------------
if not RUS_FILE.exists():
    raise FileNotFoundError(f"❌ RUS file not found: {RUS_FILE}")
if not QUESTION_FILE.exists():
    raise FileNotFoundError(f"❌ Question file not found: {QUESTION_FILE}")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"✅ Found RUS file: {RUS_FILE.name}")
print(f"✅ Found Question file: {QUESTION_FILE.name}\n")

# ======================================================
# 3. Load API Key and Initialize Client
# ======================================================
load_dotenv(ROOT / ".env")
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("❌ No OPENAI_API_KEY found in .env file")

client = OpenAI(api_key=API_KEY)
print("✅ Loaded API key successfully.\n")

# ======================================================
# 4. Load Reflector Units (RUs)
# ======================================================
with open(RUS_FILE, "r") as f:
    rus_data = json.load(f)

RUs = rus_data["RUs"] if isinstance(rus_data, dict) and "RUs" in rus_data else rus_data
for i, r in enumerate(RUs, 1):
    r.setdefault("RU_id", f"RU_{i:03d}")

print(f"✅ Loaded {len(RUs)} total RUs from {RUS_FILE.name}")

original_count = len(RUs)
RUs = random.sample(RUs, min(NUM_RUs_TO_RUN, original_count))
print(f"🎯 Randomly selected {len(RUs)} RUs out of {original_count} total.\n")

# ======================================================
# 5. Load Question File (Psychometrics)
# ======================================================
with open(QUESTION_FILE, "r") as f:
    qd = json.load(f)

questions = qd["questions"] if isinstance(qd, dict) and "questions" in qd else qd
print(f"✅ Loaded {len(questions)} questions from {QUESTION_FILE.name}\n")

# ======================================================
# 6. Response Normalization (Text → Numeric)
# ======================================================
SCALE = {
    "very inaccurate": 1, "moderately inaccurate": 2, "neither accurate nor inaccurate": 3,
    "moderately accurate": 4, "very accurate": 5,
    "strongly disagree": 1, "disagree": 2, "neutral": 3,
    "agree": 4, "strongly agree": 5
}

def normalize(text):
    """Convert textual response to numeric (1–5)."""
    if not text:
        return None
    lower = text.strip().lower()
    for k, v in SCALE.items():
        if k in lower:
            return v
    return None

# ======================================================
# 7. Prompt Builder (Simplified)
# ======================================================
def build_prompt(RU, q):
    """Build a simple persona + demographics prompt."""
    demo = ", ".join(f"{k}: {v}" for k, v in RU.get("demographics", {}).items()) if RU.get("demographics") else ""
    persona = f" Persona: {RU.get('persona')}" if RU.get("persona") else ""
    return f"""
You are a Reflector Unit (RU) simulating a human personality.
Demographics: {demo}.{persona}

Question [{q['id']}]: {q['question']}
Options: {', '.join(q['options'])}

Answer with one option only.
""".strip()

# ======================================================
# 8. Query Model
# ======================================================
def query_model(RU, q, model="gpt-4o-mini"):
    """Send one question prompt to the GPT model for a single RU."""
    try:
        start_time = time.perf_counter()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are simulating a human survey respondent. Choose exactly one option."},
                {"role": "user", "content": build_prompt(RU, q)}
            ],
            max_tokens=30,
            temperature=0.7
        )
        end_time = time.perf_counter()
        text = response.choices[0].message.content.strip()

        usage = getattr(response, "usage", None)
        return text, {
            "prompt": getattr(usage, "prompt_tokens", 0) if usage else 0,
            "completion": getattr(usage, "completion_tokens", 0) if usage else 0,
            "total": getattr(usage, "total_tokens", 0) if usage else 0,
        }, end_time - start_time

    except Exception as e:
        return f"ERROR_{type(e).__name__}", {"prompt": 0, "completion": 0, "total": 0}, 0

# ======================================================
# 9. Main Execution
# ======================================================
def run_RUs_cloud():
    """Run simplified cloud-based RU simulation (no memory modules)."""
    latencies = []
    total_calls = 0
    token_prompt = token_comp = token_total = 0
    start_time = time.perf_counter()

    with open(RESPONSES_CSV, "w", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=["timestamp", "RU_id", "question_id", "question", "response", "response_num"])
        writer.writeheader()

        for idx, RU in enumerate(RUs, 1):
            for q in questions:
                total_calls += 1
                text, usage, elapsed = query_model(RU, q)
                latencies.append(elapsed)
                token_prompt += usage["prompt"]
                token_comp += usage["completion"]
                token_total += usage["total"]

                record = {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "RU_id": RU["RU_id"],
                    "question_id": q["id"],
                    "question": q["question"],
                    "response": text,
                    "response_num": normalize(text)
                }
                writer.writerow(record)

            if idx % 50 == 0 or idx == len(RUs):
                elapsed = time.perf_counter() - start_time
                avg_time = elapsed / idx
                print(f"✅ Completed {idx}/{len(RUs)} RUs | Elapsed: {elapsed/60:.2f} min | Avg per RU: {avg_time:.2f} sec")

    # ======================================================
    # 10. Save Metrics
    # ======================================================
    total_time = time.perf_counter() - start_time
    throughput = total_calls / total_time if total_time > 0 else 0
    avg_latency = statistics.mean(latencies) if latencies else 0
    median_latency = statistics.median(latencies) if latencies else 0
    p95_latency = sorted(latencies)[int(0.95 * len(latencies)) - 1] if latencies else 0

    with open(METRICS_CSV, "w", newline="") as f_metrics:
        writer = csv.writer(f_metrics)
        writer.writerow([
            "total_RUs", "total_questions", "total_responses", "total_seconds",
            "avg_latency_sec", "median_latency_sec", "p95_latency_sec",
            "responses_per_second", "avg_time_per_RU_sec",
            "total_prompt_tokens", "total_completion_tokens", "total_tokens",
            "tokens_per_RU", "timestamp"
        ])
        writer.writerow([
            len(RUs), len(questions), total_calls,
            round(total_time, 2), round(avg_latency, 4), round(median_latency, 4),
            round(p95_latency, 4), round(throughput, 3), round(total_time / len(RUs), 2),
            token_prompt, token_comp, token_total, round(token_total / len(RUs), 2),
            datetime.now(UTC).isoformat()
        ])

    print("\n✅ Cloud run completed successfully.")
    print(f"📄 Responses saved to: {RESPONSES_CSV}")
    print(f"📊 Metrics saved to:   {METRICS_CSV}")
    print(f"⏱️  Total runtime: {total_time/60:.2f} min for {len(RUs)} RUs")

# ======================================================
# 11. Entrypoint
# ======================================================
if __name__ == "__main__":
    print(f"🚀 Starting run_RUs_cloud_noModules for {NUM_RUs_TO_RUN} RUs using OpenAI API...\n")
    run_RUs_cloud()
import json
import os
import time
import statistics
import csv
import requests
from datetime import datetime, UTC
from pathlib import Path
import sys
from dotenv import load_dotenv

# === Add project root to sys.path so imports work ===
sys.path.append(str(Path(__file__).resolve().parent.parent))

# === Managers ===
from pipeline.memory_manager import MemoryManager
from pipeline.reflection_manager import ReflectionManager
from pipeline.plan_manager import PlanManager

# -------------------------
# Resolve paths
# -------------------------
ROOT = Path(__file__).resolve().parent.parent
AGENTS_FILE = ROOT / "agents" / "synthetic_agents.json"
OCEAN_FILE = ROOT / "questions" / "OCEAN.json"
RESULTS_DIR = ROOT / "results"

RESULTS_FILE_JSONL = RESULTS_DIR / "responses.jsonl"
RESULTS_FILE_CSV = RESULTS_DIR / "responses.csv"
METRICS_FILE = RESULTS_DIR / "metrics.json"

# -------------------------
# Managers init
# -------------------------
memory_manager = MemoryManager(ROOT / "memory")
reflection_manager = ReflectionManager(ROOT / "reflections")
plan_manager = PlanManager(ROOT / "plans")

# -------------------------
# Load agents (1000)
# -------------------------
with open(AGENTS_FILE, "r") as f:
    agents = json.load(f)

if isinstance(agents, dict) and "agents" in agents:
    agents = agents["agents"]

# Auto-assign agent IDs if missing
for i, a in enumerate(agents, start=1):
    if "agent_id" not in a:
        a["agent_id"] = f"Agent_{i:03d}"

agents = agents[:1000]
print(f"✅ Loaded {len(agents)} agents")

# -------------------------
# Load OCEAN questions (first 50)
# -------------------------
print("📥 Loading OCEAN questions...")
with open(OCEAN_FILE, "r") as f:
    ocean_data = json.load(f)

if isinstance(ocean_data, dict) and "questions" in ocean_data:
    ocean_questions = ocean_data["questions"]
else:
    ocean_questions = ocean_data

ocean_questions = ocean_questions[:50]
print(f"✅ Loaded {len(ocean_questions)} OCEAN questions")

# -------------------------
# Scaling Maps
# -------------------------
SCALE_MAP = {
    "very inaccurate": 1,
    "moderately inaccurate": 2,
    "neither accurate nor inaccurate": 3,
    "moderately accurate": 4,
    "very accurate": 5,
    "strongly disagree": 1,
    "disagree": 2,
    "neutral": 3,
    "agree": 4,
    "strongly agree": 5
}

def normalize_response(text: str) -> int | None:
    if not text:
        return None
    lower = text.strip().lower()
    for key, val in SCALE_MAP.items():
        if key in lower:
            return val
    return None

# -------------------------
# Build prompt
# -------------------------
def build_prompt(agent: dict, question: dict) -> str:
    demo = ", ".join([f"{k}: {v}" for k, v in agent.get("demographics", {}).items()]) if agent.get("demographics") else ""
    persona = f" Persona: {agent.get('persona')}" if agent.get("persona") else ""

    memory = memory_manager.load(agent["agent_id"])
    reflection = reflection_manager.load(agent["agent_id"])
    plan = plan_manager.load(agent["agent_id"])

    extras = []
    if memory: extras.append(f"Memory: {memory}")
    if reflection: extras.append(f"Reflection: {reflection}")
    if plan: extras.append(f"Plan: {plan}")
    extras_text = "\n".join(extras)

    return f"""
You are an AI agent simulating a human survey respondent.

Background:
Demographics: {demo}.{persona}
{extras_text}

Question [{question['id']}]: {question['question']}
Options: {', '.join(question['options'])}

Think silently using Memory, Reflection, and Plan but DO NOT output them.
Answer with exactly one of the options only.
""".strip()

# -------------------------
# Local Ollama Response
# -------------------------
def local_llm_response(agent: dict, question: dict, model: str = "phi3"):
    try:
        prompt = build_prompt(agent, question)
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt},
            stream=True,
            timeout=300
        )
        text = ""
        for line in response.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line.decode("utf-8"))
                text += data.get("response", "")
            except json.JSONDecodeError:
                continue

        # keep only valid option
        for opt in [
            "Very Inaccurate", "Moderately Inaccurate",
            "Neither Accurate Nor Inaccurate",
            "Moderately Accurate", "Very Accurate"
        ]:
            if opt.lower() in text.lower():
                return opt
        return ""
    except Exception as e:
        return f"ERROR_{type(e).__name__}"

# -------------------------
# Run Agents
# -------------------------
def run_agents():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # Empty memory, reflection, plan folders before starting
    for folder in [ROOT / "memory", ROOT / "reflections", ROOT / "plans"]:
        for f in folder.glob("*.json"):
            f.unlink()

    total_calls, latencies = 0, []
    total_tokens = 0  # placeholder, since local models don't return token usage
    run_start = time.perf_counter()

    with open(RESULTS_FILE_JSONL, "w") as f_jsonl, open(RESULTS_FILE_CSV, "w", newline="") as f_csv:
        csv_writer = csv.DictWriter(
            f_csv,
            fieldnames=["timestamp", "agent_id", "question_id", "question", "response", "response_num"]
        )
        csv_writer.writeheader()

        for ai, agent in enumerate(agents, 1):
            memory_manager.reset(agent["agent_id"])
            reflection_manager.reset(agent["agent_id"])
            plan_manager.reset(agent["agent_id"])

            for qi, q in enumerate(ocean_questions, 1):
                total_calls += 1
                t0 = time.perf_counter()
                response = local_llm_response(agent, q)
                t1 = time.perf_counter()

                latency = t1 - t0
                latencies.append(latency)
                score = normalize_response(response)

                record = {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "agent_id": agent["agent_id"],
                    "question_id": q["id"],
                    "question": q["question"],
                    "response": response,
                    "response_num": score
                }

                f_jsonl.write(json.dumps(record) + "\n")
                f_jsonl.flush()
                csv_writer.writerow(record)
                f_csv.flush()

                memory_manager.append(agent["agent_id"], {"q": q["id"], "a": response})
                reflection_manager.append(agent["agent_id"], {"insight": f"Answered {q['id']} as {response}"})
                plan_manager.append(agent["agent_id"], {"next_action": f"Reflect consistency for {q['id']}"})

            if ai % 10 == 0:
                print(f"Progress: {ai}/{len(agents)} agents completed...", flush=True)

    run_end = time.perf_counter()
    total_seconds = max(1e-9, run_end - run_start)
    responses_per_sec = total_calls / total_seconds
    tokens_per_agent = (total_tokens / len(agents)) if len(agents) else 0

    metrics = {
        "run": {
            "total_agents": len(agents),
            "total_questions": len(ocean_questions),
            "expected_total": len(agents) * len(ocean_questions),
            "actual_total": total_calls,
        },
        "timing": {
            "total_seconds": total_seconds,
            "avg_latency": statistics.mean(latencies) if latencies else None,
            "median_latency": statistics.median(latencies) if latencies else None,
            "p95_latency": sorted(latencies)[int(0.95 * len(latencies)) - 1] if latencies else None,
            "responses_per_second": responses_per_sec
        },
        "tokens": {
            "total_tokens": total_tokens,
            "tokens_per_agent": tokens_per_agent
        },
        "timestamp": datetime.now(UTC).isoformat()
    }

    with open(METRICS_FILE, "w") as mf:
        json.dump(metrics, mf, indent=2)

    # ---- Print Summary ----
    print("\n===== Run Summary =====")
    print(f"📂 Responses JSONL : {RESULTS_FILE_JSONL}")
    print(f"📂 Responses CSV   : {RESULTS_FILE_CSV}")
    print(f"📂 Metrics JSON    : {METRICS_FILE}")
    print(f"Total Agents       : {len(agents)}")
    print(f"Total Questions    : {len(ocean_questions)}")
    print(f"Expected total     : {len(agents) * len(ocean_questions)}")
    print(f"Actual total rows  : {total_calls}")
    print(f"Throughput (resp/s): {responses_per_sec:.2f}")
    print(f"Avg latency (s)    : {statistics.mean(latencies):.4f}")
    print(f"Median latency (s) : {statistics.median(latencies):.4f}")
    print(f"p95 latency (s)    : {metrics['timing']['p95_latency']:.4f}")
    print(f"Tokens per agent   : {tokens_per_agent:.2f}")
    print("=======================")
    print("✅ Finished run_agents successfully (Local LLM, 1000×50)")

# -------------------------
# Entrypoint
# -------------------------
if __name__ == "__main__":
    print("🚀 Starting run_agents using Local Ollama (phi3)...", flush=True)
    run_agents()

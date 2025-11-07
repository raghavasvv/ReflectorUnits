import json
import os
import time
import statistics
import csv
import sys
from datetime import datetime, UTC
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# --- Path setup ---
sys.path.append(str(Path(__file__).resolve().parent.parent))
from pipeline.memory_manager import MemoryManager
from pipeline.reflection_manager import ReflectionManager
from pipeline.plan_manager import PlanManager

ROOT = Path(__file__).resolve().parent.parent
AGENTS_FILE = ROOT / "agents" / "synthetic_agents.json"
OCEAN_FILE = ROOT / "questions" / "OCEAN.json"
RESULTS_DIR = ROOT / "results" / "ocean_results"

RESPONSES_CSV = RESULTS_DIR / "cloudresponses.csv"
METRICS_CSV = RESULTS_DIR / "cloudmetrics.csv"

# --- Load API key ---
load_dotenv(ROOT / ".env")
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("No OPENAI_API_KEY found in .env")
print("Loaded API key successfully.")
client = OpenAI(api_key=API_KEY)

# --- Managers ---
memory_manager = MemoryManager(ROOT / "memory")
reflection_manager = ReflectionManager(ROOT / "reflections")
plan_manager = PlanManager(ROOT / "plans")

# --- Load agents ---
with open(AGENTS_FILE, "r") as f:
    data = json.load(f)
agents = data["agents"] if isinstance(data, dict) and "agents" in data else data
for i, a in enumerate(agents, 1):
    a.setdefault("agent_id", f"A{i:03d}")
print(f"Loaded {len(agents)} agents from synthetic_agents.json")

# --- Load OCEAN questions ---
with open(OCEAN_FILE, "r") as f:
    qd = json.load(f)
questions = qd["questions"] if isinstance(qd, dict) and "questions" in qd else qd
print(f"Loaded {len(questions)} OCEAN questions")

# --- Scaling Map ---
SCALE = {
    "very inaccurate": 1, "moderately inaccurate": 2, "neither accurate nor inaccurate": 3,
    "moderately accurate": 4, "very accurate": 5,
    "strongly disagree": 1, "disagree": 2, "neutral": 3,
    "agree": 4, "strongly agree": 5
}

def normalize(text):
    if not text:
        return None
    lower = text.strip().lower()
    for k, v in SCALE.items():
        if k in lower:
            return v
    return None

# --- Build prompt for each question ---
def build_prompt(agent, q):
    demo = ", ".join(f"{k}: {v}" for k, v in agent.get("demographics", {}).items()) if agent.get("demographics") else ""
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
You are an AI agent simulating a human personality.
Demographics: {demo}.{persona}

{extras_text}

Question [{q['id']}]: {q['question']}
Options: {', '.join(q['options'])}

Answer with one option only.
""".strip()

# --- Query model ---
def query_model(agent, q, model="gpt-4o-mini"):
    try:
        start_time = time.perf_counter()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are simulating a human survey respondent. Choose exactly one option."},
                {"role": "user", "content": build_prompt(agent, q)}
            ],
            max_tokens=30,
            temperature=0.5
        )
        end_time = time.perf_counter()

        text = response.choices[0].message.content.strip()
        u = getattr(response, "usage", None)
        usage = {
            "prompt": getattr(u, "prompt_tokens", 0) if u else 0,
            "completion": getattr(u, "completion_tokens", 0) if u else 0,
            "total": getattr(u, "total_tokens", 0) if u else 0,
        }
        return text, usage, end_time - start_time
    except Exception as e:
        return f"ERROR_{type(e).__name__}", {"prompt": 0, "completion": 0, "total": 0}, 0

# --- Main run ---
def run_agents_cloud():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for m in (memory_manager, reflection_manager, plan_manager):
        for a in agents:
            m.reset(a["agent_id"])

    latencies = []
    total_calls = 0
    token_prompt = token_comp = token_total = 0
    start_time = time.perf_counter()

    with open(RESPONSES_CSV, "w", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=["timestamp", "agent_id", "question_id", "question", "response", "response_num"])
        writer.writeheader()

        for idx, agent in enumerate(agents, 1):
            for q in questions:
                total_calls += 1
                text, usage, elapsed = query_model(agent, q)
                latencies.append(elapsed)
                token_prompt += usage["prompt"]
                token_comp += usage["completion"]
                token_total += usage["total"]

                record = {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "agent_id": agent["agent_id"],
                    "question_id": q["id"],
                    "question": q["question"],
                    "response": text,
                    "response_num": normalize(text)
                }
                writer.writerow(record)
                f_csv.flush()

                memory_manager.append(agent["agent_id"], {"q": q["id"], "a": text})
                reflection_manager.append(agent["agent_id"], {"insight": f"Answered {q['id']} as {text}"})
                plan_manager.append(agent["agent_id"], {"next_action": f"Reflect on {q['id']} consistency."})

            # Print progress every 100 agents
            if idx % 100 == 0 or idx == len(agents):
                elapsed = time.perf_counter() - start_time
                avg_time = elapsed / idx
                print(f"{idx} agents completed | Elapsed: {elapsed/60:.2f} min | Avg/agent: {avg_time:.2f} sec")

    total_time = time.perf_counter() - start_time
    throughput = total_calls / total_time
    avg_latency = statistics.mean(latencies) if latencies else 0
    median_latency = statistics.median(latencies) if latencies else 0
    p95_latency = sorted(latencies)[int(0.95 * len(latencies)) - 1] if latencies else 0

    # --- Save metrics to CSV ---
    with open(METRICS_CSV, "w", newline="") as f_metrics:
        writer = csv.writer(f_metrics)
        writer.writerow([
            "total_agents", "total_questions", "total_responses",
            "total_seconds", "avg_latency_sec", "median_latency_sec",
            "p95_latency_sec", "responses_per_second", "avg_time_per_agent_sec",
            "total_prompt_tokens", "total_completion_tokens", "total_tokens",
            "tokens_per_agent", "timestamp"
        ])
        writer.writerow([
            len(agents), len(questions), total_calls,
            round(total_time, 2), round(avg_latency, 4), round(median_latency, 4),
            round(p95_latency, 4), round(throughput, 3), round(total_time / len(agents), 2),
            token_prompt, token_comp, token_total, round(token_total / len(agents), 2),
            datetime.now(UTC).isoformat()
        ])

    print("\nRun completed successfully.")
    print(f"Responses saved to: {RESPONSES_CSV}")
    print(f"Metrics saved to:   {METRICS_CSV}")
    print(f"Total runtime: {total_time/60:.2f} min for {len(agents)} agents")

# --- Entrypoint ---
if __name__ == "__main__":
    print("Starting run_agents_cloud using OpenAI API...\n")
    run_agents_cloud()

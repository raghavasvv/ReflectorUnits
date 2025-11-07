import json, os, csv, random, time
from datetime import datetime, UTC
from pathlib import Path
import sys
from dotenv import load_dotenv
from openai import OpenAI

# === Path setup ===
sys.path.append(str(Path(__file__).resolve().parent.parent))
from pipeline.memory_manager import MemoryManager
from pipeline.reflection_manager import ReflectionManager
from pipeline.plan_manager import PlanManager

# -------------------------
# Paths
# -------------------------
ROOT = Path(__file__).resolve().parent.parent
AGENTS_FILE = ROOT / "agents" / "synthetic_agents.json"
MEDIA_Q_FILE = "/Users/rachurivijay/Desktop/Capstone/teamross/capstone3/questions/media_questions.json"
RESULTS_DIR = ROOT / "results" / "media"
RESULTS_CSV = RESULTS_DIR / "media_agents.csv"

# -------------------------
# Load API key
# -------------------------
load_dotenv(ROOT / ".env")
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

# -------------------------
# Managers
# -------------------------
memory_manager = MemoryManager(ROOT / "memory")
reflection_manager = ReflectionManager(ROOT / "reflections")
plan_manager = PlanManager(ROOT / "plans")


# -------------------------
# Helpers
# -------------------------
def load_agents():
    with open(AGENTS_FILE, "r") as f:
        agents = json.load(f)
    if isinstance(agents, dict) and "agents" in agents:
        agents = agents["agents"]
    for i, a in enumerate(agents, start=1):
        a.setdefault("agent_id", f"Agent_{i:03d}")
    return agents[:1000]

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

# -------------------------
# Prompt builder
# -------------------------
def build_prompt(agent, question):
    demo = ", ".join([f"{k}: {v}" for k, v in agent.get("demographics", {}).items()]) if agent.get("demographics") else ""
    persona = f" Persona: {agent.get('persona')}" if agent.get("persona") else ""

    options = question["options"][:]
    random.shuffle(options)  # prevent order bias

    return f"""
You are an {persona}. Demographics: {demo}
Answer the following survey question honestly, based on your personal opinion and experience.
There is no right or wrong answer.

Question [{question['id']}]: {question['question']}
Options (copy one EXACTLY as written): {', '.join(options)}

Give only one line in this format:
Final Answer: <paste one option exactly>
""".strip(), options

# -------------------------
# GPT Response
# -------------------------
def gpt_reason_and_answer(prompt, model="gpt-4o-mini"):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=1.1,
            top_p=0.9,
            presence_penalty=0.2,
            frequency_penalty=0.2,
            max_tokens=60,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"ERROR_{type(e).__name__}"

# -------------------------
# Main Function
# -------------------------
def run_media_agents():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    agents = load_agents()
    media_qs = load_json(MEDIA_Q_FILE)
    print(f"✅ Loaded {len(agents)} agents and {len(media_qs)} media questions")

    # Clear old memory/reflection/plan
    for folder in [ROOT / "memory", ROOT / "reflections", ROOT / "plans"]:
        folder.mkdir(exist_ok=True)
        for f in folder.glob("*.json"):
            f.unlink()

    # Create CSV
    with open(RESULTS_CSV, "w", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=["timestamp", "agent_id", "question_id", "response"])
        writer.writeheader()

        for i, agent in enumerate(agents, start=1):
            memory_manager.reset(agent["agent_id"])
            reflection_manager.reset(agent["agent_id"])
            plan_manager.reset(agent["agent_id"])

            for q in media_qs:
                qid = str(q["id"])
                prompt, opt_list = build_prompt(agent, q)
                response_text = gpt_reason_and_answer(prompt)

                # --- Parse choice exactly ---
                text = response_text.strip().lower()
                chosen = None
                prefix = "final answer:"
                if prefix in text:
                    tail = text.split(prefix, 1)[1].strip()
                    tail = " ".join(tail.split())
                    for opt in opt_list:
                        if tail == opt.lower():
                            chosen = opt
                            break
                if not chosen:
                    for opt in opt_list:
                        if f" {opt.lower()} " in f" {text} ":
                            chosen = opt
                            break
                if not chosen:
                    chosen = random.choice(opt_list)  # uniform fallback

                # --- Update cognitive state ---
                memory_manager.append(agent["agent_id"], {"q": qid, "a": chosen})
                reflection_manager.append(agent["agent_id"], {"insight": f"Answered {qid} as {chosen}"})
                plan_manager.append(agent["agent_id"], {"next_action": f"Reflect consistency for {qid}"})

                # --- Write CSV row ---
                writer.writerow({
                    "timestamp": datetime.now(UTC).isoformat(),
                    "agent_id": agent["agent_id"],
                    "question_id": qid,
                    "response": chosen
                })

            if i % 50 == 0:
                print(f"   Completed {i}/{len(agents)} agents")

    print("\n✅ All agents finished answering all questions.")
    print(f"📄 Responses saved: {RESULTS_CSV}")
    print("🧠 Memory, Reflection, and Plan updated for each agent.")
    print("===========================================")

# -------------------------
# Entrypoint
# -------------------------
if __name__ == "__main__":
    print("🚀 Running media survey ...")
    run_media_agents()
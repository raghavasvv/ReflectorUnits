import json, os, csv, random, time
from datetime import datetime, UTC
from pathlib import Path
import sys
from dotenv import load_dotenv
from openai import OpenAI

# =====================================================
# ✅ PATH SETUP (auto-detect capstone3 root)
# =====================================================
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent  # capstone3 root directory
sys.path.append(str(PROJECT_ROOT))

from pipeline.memory_manager import MemoryManager
from pipeline.reflection_manager import ReflectionManager
from pipeline.plan_manager import PlanManager

# =====================================================
# CONFIGURATION
# =====================================================
NUM_RUS = 100   # 🔹 change this number as you like (10, 100, 500, 1000)

RUS_FILE = PROJECT_ROOT / "RUS" / "synthetic_RUS.json"
MEDIA_Q_FILE = PROJECT_ROOT / "questions" / "media_questions.json"
RESULTS_DIR = PROJECT_ROOT / "results" / "media"
RESULTS_CSV = RESULTS_DIR / "media_RUs_PMR.csv"

# =====================================================
# ✅ LOAD API KEY
# =====================================================
load_dotenv(PROJECT_ROOT / ".env")
API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise ValueError("❌ API key not found. Make sure .env file exists in capstone3 directory.")

print(f"✅ API key loaded successfully (starts with: {API_KEY[:7]}...)")
client = OpenAI(api_key=API_KEY)

# =====================================================
# MANAGERS
# =====================================================
memory_manager = MemoryManager(PROJECT_ROOT / "memory")
reflection_manager = ReflectionManager(PROJECT_ROOT / "reflections")
plan_manager = PlanManager(PROJECT_ROOT / "plans")

# =====================================================
# HELPERS
# =====================================================
def load_rus():
    """Load Reflector Units (RUs) data from synthetic_RUS.json"""
    with open(RUS_FILE, "r") as f:
        rus_units = json.load(f)
    if isinstance(rus_units, dict) and "RUs" in rus_units:
        rus_units = rus_units["RUs"]
    for i, r in enumerate(rus_units, start=1):
        r.setdefault("RUs_id", f"RU_{i:03d}")
    return rus_units[:NUM_RUS]

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

# =====================================================
# PROMPT BUILDER
# =====================================================
def build_prompt(ru, question):
    demo = ", ".join([f"{k}: {v}" for k, v in ru.get("demographics", {}).items()]) if ru.get("demographics") else ""
    options = question["options"][:]
    random.shuffle(options)  # prevent order bias

    return f"""
You are a Reflector Unit (RU) representing a human survey respondent.
Demographics: {demo}

Answer the following survey question honestly, based on your reasoning.

Question [{question['id']}]: {question['question']}
Options (copy one EXACTLY as written): {', '.join(options)}

Give only one line in this format:
Final Answer: <paste one option exactly>
""".strip(), options

# =====================================================
# GPT RESPONSE HANDLER
# =====================================================
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

# =====================================================
# MAIN FUNCTION
# =====================================================
def run_media_RUs():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rus_units = load_rus()
    media_qs = load_json(MEDIA_Q_FILE)
    print(f"✅ Loaded {len(rus_units)} RUs and {len(media_qs)} media questions")

    # Clear old memory/reflection/plan
    for folder in [PROJECT_ROOT / "memory", PROJECT_ROOT / "reflections", PROJECT_ROOT / "plans"]:
        folder.mkdir(exist_ok=True)
        for f in folder.glob("*.json"):
            f.unlink()

    # Create CSV output file
    with open(RESULTS_CSV, "w", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=["timestamp", "RUs_id", "question_id", "response"])
        writer.writeheader()

        for i, ru in enumerate(rus_units, start=1):
            memory_manager.reset(ru["RUs_id"])
            reflection_manager.reset(ru["RUs_id"])
            plan_manager.reset(ru["RUs_id"])

            for q in media_qs:
                qid = str(q["id"])
                prompt, opt_list = build_prompt(ru, q)
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
                    chosen = random.choice(opt_list)  # fallback random

                # --- Update cognitive state ---
                memory_manager.append(ru["RUs_id"], {"q": qid, "a": chosen})
                reflection_manager.append(ru["RUs_id"], {"insight": f"Answered {qid} as {chosen}"})
                plan_manager.append(ru["RUs_id"], {"next_action": f"Reflect consistency for {qid}"})

                # --- Write CSV row ---
                writer.writerow({
                    "timestamp": datetime.now(UTC).isoformat(),
                    "RUs_id": ru["RUs_id"],
                    "question_id": qid,
                    "response": chosen
                })

            if i % 50 == 0:
                print(f"   ✅ Completed {i}/{len(rus_units)} RUs")

    print("\n✅ All RUs finished answering all questions.")
    print(f"📄 Responses saved: {RESULTS_CSV}")
    print("🧠 Memory, Reflection, and Plan updated for each RU.")
    print("===========================================")

# =====================================================
# ENTRYPOINT
# =====================================================
if __name__ == "__main__":
    print("🚀 Running media survey for Reflector Units (RUs)...")
    run_media_RUs()
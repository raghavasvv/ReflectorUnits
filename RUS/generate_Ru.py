import json, random, time, os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# =======================================================
# 1️⃣ Load your OpenAI API key
# =======================================================
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =======================================================
# 2️⃣ Configuration
# =======================================================
NUM_AGENTS = 50   # 👈 change this number anytime (e.g., 100, 1000)
MODEL = "gpt-4o-mini"
GENDERS = ["male", "female"]
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = BASE_DIR / "diverse_RUS.json"

# =======================================================
# 3️⃣ Persona Generation Prompt
# =======================================================
# This ensures we cover a broad range of roles in society
BASE_PROMPT = """
You are creating a short persona (max 25 words) for a fictional human.
Each persona should sound realistic and natural.

Make sure to vary occupations broadly across society — include:
- Students (undergraduate, graduate, medical, etc.)
- Doctors, nurses, and healthcare workers
- Bus drivers, taxi drivers, delivery people
- Shop owners, restaurant chefs, baristas
- Engineers, scientists, professors
- Artists, writers, musicians
- Office workers, managers, accountants
- Farmers, construction workers, electricians
- Homemakers, social workers, police officers, etc.

For each, describe:
1. Occupation or role
2. 2–3 personality traits
3. A brief daily-life or social behavior hint.

Examples:
- Graduate student, curious and hardworking, enjoys late-night study sessions and campus life.
- Bus driver, cheerful and talkative, knows everyone in the neighborhood.
- Doctor, compassionate and focused, cares deeply about her patients.
- Artist, imaginative and introverted, finds inspiration in city life.
- Shop owner, honest and patient, loves chatting with regular customers.

Return only ONE line per persona — no numbering, no quotes.
"""

# =======================================================
# 4️⃣ Function to create one agent
# =======================================================
def create_agent():
    age = random.randint(18, 70)
    gender = random.choice(GENDERS)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": BASE_PROMPT}],
        temperature=1.2,
        max_tokens=70,
    )

    persona = response.choices[0].message.content.strip()

    return {
        "age": age,
        "gender": gender,
        "persona": persona,
        "memory": [],
        "reflection": [],
        "plan": []
    }

# =======================================================
# 5️⃣ Generate agents
# =======================================================
agents = []
for i in range(NUM_AGENTS):
    print(f"🧠 Creating agent {i+1}/{NUM_AGENTS} ...")
    agent = create_agent()
    agents.append(agent)
    time.sleep(0.5)  # small delay for API safety

# =======================================================
# 6️⃣ Save to JSON file
# =======================================================
with open(OUTPUT_FILE, "w") as f:
    json.dump(agents, f, indent=4)

print(f"\n✅ Generated {NUM_AGENTS} diverse agents → {OUTPUT_FILE}")

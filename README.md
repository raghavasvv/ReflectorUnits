# 🪜 **Breaking the Project into Smaller Parts (Short & Clear)**

### **Phase 1: OCEAN Personality Simulation**

1. **Build Synthetic Agents** → Create 1,000 AI agents (`synthetic_agents.json`) with demographic and personality traits.  
2. **Collect Human Responses** → Two real human datasets (`phase1`, `phase2`) used for validation.  
3. **Run OCEAN Questions** → Agents answer Big Five (OCEAN) personality questions using:
   - **Cloud-based LLMs** via OpenAI API.
   - **Local LLM** (e.g., Ollama / Llama-3).  
4. **Compute Metrics** → Compare Human vs Agent responses using:
   - **Correlation**
   - **KL Divergence**
   - **JS Divergence**
   - **Normalized Accuracy**
5. **Generate Visualizations** → Produce metrics and comparison charts under `results/ocean_results/`.

---

### **Phase 2: Media Survey Simulation**

6. **Collect Media Poll Data** → Real-world public poll questions (2020–2025).  
7. **Run Agent Responses** → Agents answer identical media poll questions using both cloud and local models.  
8. **Compare with Humans** → Evaluate trends and response alignment.  
9. **Visualize Media Results** → Generate comparative charts for percentage distributions and time trends.

---

### **Phase 3: Deliverables**

10. **Datasets** → Synthetic agent responses + Human datasets (`phase1`, `phase2`).  
11. **Analysis Outputs** → Charts for correlation, KL/JS divergence, and normalized accuracy.  
12. **Final Report** → Methodology, results, and summary of human-agent alignment.  
13. **Presentation** → Slide deck summarizing pipeline, results, and charts.

---

## 🧩 **Key Aspects**

**Agent Design**  
- 1,000 OCEAN-based synthetic agents.  
- Each agent answers using either Cloud API or Local LLM pipeline.  

**Datasets**
- `OCEAN Dataset` → Big Five Personality Questions  
- `Media Dataset` → Public opinion poll questions (2020 vs 2025)

**Pipeline**
1. Load Agents + Questions.  
2. Build prompts (Cloud or Local).  
3. Execute simulations.  
4. Store all responses in CSV/JSON under `results/`.  
5. Compute metrics & plot graphs.

**Analysis**
- Compare Human vs Agent distributions.
- Compute accuracy, divergence, and consistency scores.
- Visualize metrics across runs.

---

## 📂 **Project Structure**

TEAMROSS/capstone3/
├── agents/
│ └── synthetic_agents.json
│
├── human/
│ ├── human_responses_phase1.csv
│ ├── human_responses_phase2.csv
│ ├── human_vs_human_metrics_250.csv / .png
│ ├── human_vs_human_metrics_1000.csv / .png
│
├── pipeline/
│ ├── CLOUD_API/
│ │ ├── Runagent_cloud.py
│ │ ├── compare_HVA_1000.py
│ │ ├── compare_ocean_HVA500.py
│ │ ├── normalize_acc_1000.py
│ │ └── cloud_vs_cloud250.py
│ ├── LOCAL_LLM/
│ │ ├── compare_localvshuman.py
│ │ └── run_agents.py
│ ├── HUMAN_VS_HUMAN/
│ │ ├── compare_HVH_250.py
│ │ └── compre_ocean_HVH_1000.py
│ ├── 5_STUDIES/ # Future Extension
│ │ ├── ames_and_frisky.py
│ │ ├── cooney_et_al.py
│ │ ├── halevy_halali.py
│ │ ├── rai_et_al_final.py
│ │ └── schilke_reimann_cook.py
│
├── questions/
│ ├── OCEAN.json
│ └── media.json
│
├── results/
│ ├── ocean_results/ # OCEAN metrics
│ ├── media/ # Media metrics
│ └── study_results/ # (future studies)
│
├── requirements.txt
└── README.md

---

## ⚙️ **Setup Instructions**

1. **Clone this repo** (or copy project folder).  
2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate       # macOS/Linux
   venv\Scripts\activate          # Windows


## Install dependencies:

bash
Copy code
pip install -r requirements.txt
Add API Key for cloud runs:
Create a .env file in the project root and add:

ini
Copy code
OPENAI_API_KEY=your_openai_api_key_here

## Running the Pipeline

1️⃣ Generate Agent Responses (Cloud)
python pipeline/CLOUD_API/Runagent_cloud.py

2️⃣ Generate Agent Responses (Local LLM)
python pipeline/LOCAL_LLM/run_agents.py

3️⃣ Compare Human vs Agent (Cloud)
python pipeline/CLOUD_API/compare_HVA_1000.py

4️⃣ Compare Human vs Agent (Local)
python pipeline/LOCAL_LLM/compare_localvshuman.py

5️⃣ Validate Human Internal Consistency
python pipeline/HUMAN_VS_HUMAN/compare_HVH_250.py



# Example Outputs
## OCEAN Results (Human vs Agent)
File	Description
normalized_accuracy_cloud1000.png	Normalized accuracy of agent–human responses
Correlation_UHCL_Hawks_Final.png	Correlation plot for OCEAN metrics
KL_Divergence_UHCL_Hawks_Final.png	KL Divergence (Human vs Agent distributions)
JS_Divergence_UHCL_Hawks_Final.png	JS Divergence indicating behavioral overlap

# Example Charts


Figure 1 – Normalized accuracy (Human vs Agent).


Figure 2 – JS divergence showing distribution similarity.

Media Results (Poll Comparisons)
File	Description
media_metrics_bar.png	Overall media poll results (agents vs humans)
media_metrics_average.png	Average comparison for multiple questions
media_q1_trend_chart.png	2020 vs 2025 agent response trends for question 1

# Example Charts


Figure 3 – Human vs Agent response percentages.


Figure 4 – Temporal agent response trends.

# Metric Interpretation
Metric	Ideal Range	Interpretation
	
KL < 0.05	Minimal information loss	
JS < 0.02	>98% behavioral similarity	


# 🧠 Generative Reflector Units (RUs)  
### *Simulating 1,000 Human-like Respondents through Local and Cloud LLM Execution*

---

## ⚙️ Part 1 – Project Setup

### 🧩 1. System Requirements
| Tool | Purpose |
|------|----------|
| **Python 3.10 or later** | Runs all RU scripts |
| **Anaconda / Miniconda (or venv)** | Creates a clean virtual environment |
| **Git** | Clone this repository |
| **OpenAI API Key (Cloud mode)** | Get from https://platform.openai.com |
| **Ollama (optional)** | Needed only for Local LLM mode (e.g., Llama 3) |

---

### 🧱 2. Clone the Repository
```bash
git clone https://github.com/<yourusername>/capstone3.git
cd capstone3
```

---

### 🧮 3. Create and Activate a Virtual Environment
**Conda (recommended):**
```bash
conda create -n capstone3 python=3.12 -y
conda activate capstone3
```
**or venv:**
```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

---

### 📦 4. Install Dependencies
```bash
pip install -r requirements.txt
```

If missing, create:
```bash
# requirements.txt
openai
python-dotenv
pandas
numpy
matplotlib
scipy
tqdm
```

---

### 🔑 5. Add OpenAI API Key (for Cloud Mode)
Create `.env` in project root:
```
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
```

---

### 🧠 6. Install Ollama (for Local Mode)
```bash
# macOS
brew install ollama
# Windows
winget install Ollama.Ollama
```
Then download the model:
```bash
ollama pull llama3
```

> Ollama runs system-wide; the Python environment simply connects to it.

---

### 🧾 7. Verify Setup
```bash
python -c "import openai, pandas, numpy, matplotlib, scipy; print('✅ All dependencies installed successfully!')"
```

---

## 🚀 Part 2A – Running Reflector Units in Local Mode (Ollama + Llama 3)

### 🧩 1. Start Ollama Service
```bash
ollama serve
```
Keep this terminal open while running RUs.

---

### 🧰 2. Check Model
```bash
ollama list
# If missing:
ollama pull llama3
```

---

### 🧮 3. Activate Environment
```bash
conda activate capstone3
# or
source venv/bin/activate
```

---

### 🧾 4. Run Reflector Units
```bash
python RUS/run_RUS_LLM.py
```

The script will load RU profiles (`RUS/synthetic_RUs.json`), read questions from `questions/`, send prompts to Llama 3 through Ollama, and save results to `results/`.

---

### 📁 5. Outputs
```
results/
 └── media/
      ├── media_RUs.csv
      ├── media_log.json
      ├── response_snapshots/
      └── batch_metrics.csv
```

---

### ⚠️ 6. Troubleshooting
| Issue | Cause | Fix |
|-------|--------|-----|
| `ConnectionRefusedError` | Ollama not running | `ollama serve` |
| `Model llama3 not found` | Model not downloaded | `ollama pull llama3` |
| `ModuleNotFoundError` | Missing packages | `pip install -r requirements.txt` |
| Slow responses | Heavy CPU/RAM load | Reduce batch size in script |

---

### 🧩 7. Notes
- Works completely offline once Llama 3 is downloaded.  
- Adjust batch size or temperature inside `run_RUS_LLM.py`.  
- Logs stored in `results/local_logs/`.

---

## ☁️ Part 2B – Running Reflector Units in Cloud Mode (OpenAI API)

### 🔑 1. Confirm API Key
`.env` must contain:
```
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
```

---

### 🧩 2. Activate Environment
```bash
conda activate capstone3
# or
source venv/bin/activate
```

---

### 🚀 3. Run Cloud Mode
```bash
python RUS/run_RUS_cloud.py
```

The script loads Reflector Unit profiles and questions, calls **GPT-4o-mini**, and stores responses and metrics under `results/`.

---

### 📁 4. Outputs
```
results/
 └── media/
      ├── media_RUs_cloud.csv
      ├── cloud_run_log.json
      ├── batch_metrics_cloud.csv
      └── comparison_graphs/
```

---

### ⚠️ 5. Common Errors
| Issue | Cause | Fix |
|-------|--------|-----|
| `AuthenticationError` | Invalid API key | Re-check `.env` |
| `RateLimitError` | Too many requests | Lower batch size / add delays |
| `FileNotFoundError` | Wrong path | Verify file paths |
| `Timeout` | Slow internet / large batch | Re-run smaller batches |

---

### 📊 6. Usage Tips
- Monitor token usage on OpenAI dashboard.  
- Adjust `temperature`, `max_tokens`, `batch_size` inside `run_RUS_cloud.py`.  
- Results auto-timestamp in `results/`.

---

## 📊 Part 3 – Comparing Human vs Reflector Unit Results

### 🧩 1. Required Files
| File | Description |
|------|--------------|
| `results/media/media_RUs.csv` or `media_RUs_cloud.csv` | RU responses |
| `results/media/media_human_resp.json` | Human survey data |

---

### 🚀 2. Run Comparison
```bash
python pipeline/compare_human_vs_RU.py
```

Computes KL-Divergence, JS-Divergence, and t-tests, then plots graphs.

---

### 📁 3. Outputs
```
results/media/
 ├── KL_JS_metrics.csv
 ├── human_vs_RU_summary.csv
 └── comparison_graphs/
      ├── kl_divergence.png
      ├── js_divergence.png
      └── distribution_overlap.png
```

---

### 📈 4. Metric Interpretation
| Metric | Meaning |
|---------|----------|
| **KL ↓** | Smaller = closer to human |
| **JS ↓** | Symmetric distance (0 ≈ perfect) |
| **t-Test p ↑** | > 0.05 → no significant difference |
| **Consistency α ↑** | Higher = more stable RUs |

Example: `KL 0.028  JS 0.014  α 0.91`

---

## 🧩 Part 4 – Visualization and Result Interpretation

### 🖼️ 1. Graph Location
```
results/media/comparison_graphs/
```

Files: `kl_divergence.png`, `js_divergence.png`, `distribution_overlap.png`, `batch_consistency.png`

---

### 🧮 2. Regenerate Plots
```bash
python pipeline/metrics_visualizer.py
```

---

### 📊 3. How to Read Charts
| Plot | Shows | Read As |
|------|--------|---------|
| KL Bar Chart | Info loss RU→Human | Lower = better |
| JS Heatmap | Similarity across topics | Cooler colors = closer |
| Distribution Overlap | Probabilities per question | More overlap = similar |
| Consistency Histogram | Stability per batch | Peaks near 1.0 = good |

---

### 🧠 4. Alignment Quality
| Range | Quality | Meaning |
|--------|----------|----------|
| 0–0.02 | Excellent | Almost human-like |
| 0.02–0.05 | Good | Minor variation |
| 0.05–0.10 | Moderate | Some topic shift |
| > 0.10 | Low | Needs tuning |

---

### 🧩 5. Tips for Reports
- Include Local vs Cloud comparisons.  
- Mention internal consistency α values.  
- Label plots clearly as “RUs vs Humans”.

---

## 🗂️ Part 5 – Project Folder Structure and Execution Flow

### 📁 1. Folder Layout
```
capstone3/
├── RUS/
│   ├── run_RUS_LLM.py
│   ├── run_RUS_cloud.py
│   └── synthetic_RUs.json
│
├── pipeline/
│   ├── memory_manager.py
│   ├── reflection_manager.py
│   ├── plan_manager.py
│   ├── compare_human_vs_RU.py
│   ├── internal_consistency.py
│   └── metrics_visualizer.py
│
├── questions/
│   ├── media_questions.json
│   ├── psychometrics.json
│   ├── classic_studies.json
│   └── uhcl_survey.json
│
├── results/
│   ├── media/
│   └── local_logs/
│
├── .env
├── requirements.txt
└── README.md
```

---

### 🔄 2. Execution Flow
```
Reflector Units (RUs)
     │
     ▼
MemoryManager → ReflectionManager → PlanManager
     │
     ▼
Response Generation (Llama 3 or GPT-4o-mini)
     │
     ▼
Results Storage (CSV/JSON)
     │
     ▼
Human vs RU Comparison (KL, JS)
     │
     ▼
Visualization & Metrics Plots
```

---

### 🧠 3. Step Summary
| Step | Module | Input | Output |
|------|---------|--------|---------|
| 1 | `run_RUS_LLM.py` / `run_RUS_cloud.py` | RU profiles + questions | RU responses (CSV/JSON) |
| 2 | `compare_human_vs_RU.py` | Human + RU data | KL/JS metrics |
| 3 | `internal_consistency.py` | RU responses | α (reliability) |
| 4 | `metrics_visualizer.py` | Metric CSVs | PNG graphs |

---

✅ **Setup complete and ready for execution.**  
Run either Local or Cloud mode, compare results, and review the plots in `results/media/comparison_graphs/`.


### 🧾 License
MIT License – Free to use and modify with attribution.


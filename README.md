<<<<<<< HEAD
# Patient Medical Retrieval-Augmented Generation (RAG-LLM) System for Rural Healthcare

> **Author:** Sam Daniel  
> **License:** MIT © 2026 Sam Daniel

A production-ready, AI-powered clinical decision support system designed for rural healthcare settings. It uses **Groq Cloud API (Llama-3.3-70b-versatile)** for near-instant LLM inference, **PubMedBERT** for domain-specific medical embeddings, **Qdrant** as a local vector database, and **FastAPI + LangChain** as the backend orchestration framework.

---

## 🧠 Key Features

- **Groq Cloud LLM Integration** — Powered by `llama-3.3-70b-versatile` via Groq's API. Reduces answer latency from ~5 minutes (local CPU) to ~2 seconds while leveraging a 70B parameter reasoning model.
- **Dynamic Patient Filtering** — Automatically extracts patient IDs (e.g. `10040025`) from natural language queries using Regex, applying exact-match Qdrant metadata filters for isolated, hallucination-free retrieval.
- **Structured Clinical Reports** — Every response follows a strict 3-section Markdown format:
  - 👤 **Patient Overview** — Brief background on the patient
  - 📋 **Detailed Findings** — Bulleted, comprehensive answer to the query
  - 📝 **Summary** — Concluding 1–2 sentence answer
- **Qdrant Vector Database** — Self-hosted, local document store enabling semantic similarity search over structured patient records (diagnoses, vitals, triage, medications).
- **PubMedBERT Embeddings** — `NeuML/pubmedbert-base-embeddings`, a model purpose-built to understand complex medical terminology.
- **Objective Evaluation** — Built-in `/evaluate` endpoint that benchmarks retrieval performance on a ground-truth test set, reporting **Accuracy, Precision, Recall, and F1 Score** with auto-generated charts.
- **Premium UI** — Glassmorphism CSS design, shimmer loading states, and dynamic Markdown rendering for a modern, polished user experience.

---

## 🗂️ Dataset

This system is built on the **[MIMIC-IV MEDS Demo Dataset](https://physionet.org/content/mimic-iv-demo-meds/0.0.1/)** (Medical Event Data Standard format), a publicly available de-identified clinical dataset from PhysioNet.

- **64 unique patients** with ED stay records
- Source files ingested: `edstays`, `diagnosis`, `triage`, `vitalsign`, `medrecon`, `pyxis` (via MEDS `.parquet` format)
- Chunked with overlap (`chunk_size=1000`, `chunk_overlap=200`) and stored in Qdrant

---

## 🏗️ Project Structure

```
Patient-Medical-RAG-LLM-System-for-Rural-Healthcare/
│
├── main.py              # FastAPI application (routes, LLM chain, evaluation)
├── ingest.py            # Data ingestion pipeline (parquet → Qdrant)
│
├── data/                # Raw CSV exports and MIMIC MEDS zip/parquet files
│   └── mimic/           # Extracted MEDS parquet files
│
├── local_qdrant/        # Persisted Qdrant vector database (auto-created by ingest.py)
├── models/              # (optional) cached embedding model weights
│
├── templates/
│   └── index.html       # Frontend Jinja2 template
├── static/              # Served static files (CSS, JS, eval charts)
│
├── .env                 # Environment variables (API keys, config)
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## ⚙️ Technology Stack

| Layer | Technology |
|---|---|
| LLM | Groq Cloud API — `llama-3.3-70b-versatile` |
| Embeddings | `NeuML/pubmedbert-base-embeddings` (SentenceTransformer) |
| Vector DB | Qdrant (local, on-disk) |
| Orchestration | LangChain (`langchain-core`, `langchain-groq`, `langchain-community`) |
| Backend | FastAPI + Uvicorn |
| Frontend | Jinja2 Templates, Vanilla CSS (Glassmorphism) |
| Evaluation | scikit-learn, matplotlib, seaborn |
| Data | pandas, pyarrow (MIMIC-IV MEDS `.parquet`) |

---

## 🚀 How to Run

### 1. Environment Setup

Create a `.env` file in the root directory:

```ini
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
EMBED_MODEL=NeuML/pubmedbert-base-embeddings
QDRANT_PATH=local_qdrant
COLLECTION=patient_records
TOP_K=5
```

Get your free Groq API key at [console.groq.com](https://console.groq.com).

---

### 2. Install Dependencies

```bash
venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

### 3. Ingest the Patient Data

Run once to build the Qdrant vector database from the MIMIC-IV MEDS dataset:

```bash
venv\Scripts\python.exe -u ingest.py
```

> ⚠️ Do not close the terminal until you see `"Vector DB successfully created"`. This process takes approximately **20–30 minutes on CPU** as it generates embeddings for all patient records.

To force a rebuild of an existing database:

```bash
venv\Scripts\python.exe -u ingest.py --force
```

---

### 4. Start the Web Server

```bash
venv\Scripts\python.exe -m uvicorn main:app
```

> ⚠️ Do **not** use `--reload`. It causes multiple processes to fight over the Qdrant file lock.

Open your browser at: **http://127.0.0.1:8000**

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serves the main web UI |
| `GET` | `/health` | Health check — returns model and collection info |
| `POST` | `/get_response` | Submit a natural language query; returns clinical report |
| `GET` | `/evaluate` | Runs retrieval benchmark and returns metrics + chart images |

### Example: Health Check

```
GET http://127.0.0.1:8000/health
→ { "status": "ready", "model": "llama-3.3-70b-versatile", "collection": "patient_records" }
```

### Example: Query

```
POST http://127.0.0.1:8000/get_response
Form body: query=What is the medical history of patient 10040025?
```

---

## 💬 Sample Queries

Use any of the 64 patient IDs in the dataset:

- `What is the medical history of patient 10040025?`
- `What medications were given to patient 10018328?`
- `Summarize the diagnosis and ED stay for patient 10014729.`
- `What are the vitals and triage notes for patient 10000032?`

---

## 📊 Evaluation

Navigate to **http://127.0.0.1:8000/evaluate** to benchmark the system.

The `/evaluate` endpoint runs a ground-truth test suite and returns:

| Metric | Description |
|--------|-------------|
| **Accuracy** | % of queries that retrieved the correct patient's records |
| **Precision** | Of retrieved results, how many were correct |
| **Recall** | Of all correct records, how many were retrieved |
| **F1 Score** | Harmonic mean of Precision and Recall |

Evaluation charts (`metrics_bar.png`, `metrics_pie.png`, `metrics_cm.png`) are saved to `static/` and rendered in the UI.

---

## 📝 License

MIT License © 2026 Sam Daniel. See [LICENSE](LICENSE) for full terms.
=======
# Patient-Medical-RAG-LLM-System-for-Rural-Healthcare
Retrieval-Augmented Generation (RAG) system for querying patient medical history from EHR datasets. Combines semantic search, patient-level metadata filtering, and LLM-based generation to produce structured clinical reports. Designed for accurate, context-aware retrieval and efficient analysis of longitudinal patient data.
Patient Medical RAG-LLM System designed for rural healthcare settings. Enables natural language querying of patient medical history from EHR data using semantic retrieval and strict patient-level filtering, generating structured clinical reports to support efficient and accurate decision-making in low-resource environments.
>>>>>>> 4a48ba6bf92f548dcd8048d46fdb6aba2428c754

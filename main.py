import os
import asyncio
import logging
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# --------------------------------------------------------------------------- #
#  Logging
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Config
# --------------------------------------------------------------------------- #
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL     = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
EMBED_MODEL    = os.getenv("EMBED_MODEL", "NeuML/pubmedbert-base-embeddings")
QDRANT_PATH    = os.getenv("QDRANT_PATH", "local_qdrant")
COLLECTION     = os.getenv("COLLECTION",  "patient_records")
TOP_K          = int(os.getenv("TOP_K", "5"))

if not GROQ_API_KEY:
    log.error("GROQ_API_KEY is not set! Please add it to your .env file.")
    raise SystemExit("Missing GROQ_API_KEY")

# --------------------------------------------------------------------------- #
#  FastAPI app
# --------------------------------------------------------------------------- #
app = FastAPI(title="Patient Medical Retrieval-Augmented Generation (RAG-LLM) System for Rural Healthcare")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# --------------------------------------------------------------------------- #
#  LLM — Groq Cloud
# --------------------------------------------------------------------------- #
log.info("Initializing Groq LLM (%s) …", GROQ_MODEL)
llm = ChatGroq(
    model=GROQ_MODEL,
    api_key=GROQ_API_KEY,
    temperature=0.3,
    max_tokens=2048,
)
log.info("Groq LLM ready ✓")

# --------------------------------------------------------------------------- #
#  Embeddings + Vector Store
# --------------------------------------------------------------------------- #
log.info("Loading PubMedBERT embeddings …")
embeddings = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)

try:
    client = QdrantClient(path=QDRANT_PATH)
    db = Qdrant(client=client, embeddings=embeddings, collection_name=COLLECTION)
    log.info("Qdrant collection '%s' loaded ✓", COLLECTION)
except Exception as e:
    log.error(
        "Failed to open Qdrant at '%s': %s\n"
        "Make sure ingest.py was run first, and no other process holds the lock.",
        QDRANT_PATH, e,
    )
    raise

# Pre-warm embedding model
_ = embeddings.embed_query("warmup")
log.info("Embeddings pre-warmed ✓")

# --------------------------------------------------------------------------- #
#  Prompt + LLM Setup
# --------------------------------------------------------------------------- #
system_prompt = """\
You are an expert AI clinical assistant reviewing patient records. Your task is to generate a professional, highly structured medical report based *only* on the provided records.

If the records do not contain the answer to the user's query or the patient is missing from the database, clearly state: "Insufficient information in the retrieved patient records." Do not fabricate or hallucinate any details.

You MUST structure your response EXACTLY with these 3 sections, in order. Do NOT skip the Summary section:

### 👤 Patient Overview
[Provide a brief 1-2 sentence overview of the patient based on records]

### 📋 Detailed Findings
[Answer the user's query comprehensively here. Use bullet points for readability. Group related items if applicable]

### 📝 Summary
[Provide a final 1-2 sentence concluding summary directly answering the user's query. This section is MANDATORY.]\
"""

human_prompt = """\
**Context (Patient Records):**
{context}

**User Query:** {question}

**Generated Clinical Report:**
### 👤 Patient Overview\
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", human_prompt)
])

# Create a simple runnable chain combining prompt and LLM
from langchain_core.runnables import RunnableSequence
qa_chain = prompt | llm | (lambda x: "### 👤 Patient Overview\n" + x.content if not x.content.startswith("###") else x.content)

log.info("LLM components ready ✓")

# --------------------------------------------------------------------------- #
#  Ground-truth test set for real evaluation
# --------------------------------------------------------------------------- #
EVAL_TEST_CASES = [
    {"query": "Show medical history of patient 10000032", "expected_patient_id": 10000032},
    {"query": "What lab results exist for patient 10000032?", "expected_patient_id": 10000032},
    {"query": "What medications were given to patient 10001725?", "expected_patient_id": 10001725},
    {"query": "What diagnoses does patient 10001725 have?", "expected_patient_id": 10001725},
    {"query": "Show visits for patient 10002428", "expected_patient_id": 10002428},
    {"query": "What is the treatment history of patient 10002428?", "expected_patient_id": 10002428},
    {"query": "Lab tests for patient 10004235", "expected_patient_id": 10004235},
    {"query": "Medical events for patient 10004235", "expected_patient_id": 10004235},
    {"query": "History of patient 10006008", "expected_patient_id": 10006008},
    {"query": "What happened to patient 10006008?", "expected_patient_id": 10006008},
]

# --------------------------------------------------------------------------- #
#  Routes
# --------------------------------------------------------------------------- #
@app.get("/health")
async def health():
    return {"status": "ready", "model": GROQ_MODEL, "collection": COLLECTION}

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "title": "Patient Medical Retrieval-Augmented Generation (RAG-LLM) System for Rural Healthcare"
    })

@app.post("/get_response")
async def get_response(query: str = Form(...)):
    query = query.strip()
    if not query:
        return JSONResponse(status_code=400, content={"error": "Query cannot be empty."})

    try:
        match = re.search(r'\b(100\d{5})\b', query)
        target_patient_id = int(match.group(1)) if match else None

        search_kwargs = {"k": TOP_K}
        if target_patient_id:
            search_kwargs["filter"] = {"patient_id": target_patient_id}

        def _search():
            return db.similarity_search(query, **search_kwargs)
            
        source_docs = await asyncio.to_thread(_search)

        if not source_docs:
            msg = "No relevant records found in the database."
            if target_patient_id:
                msg += f"\nNote: Verified that patient {target_patient_id} has no records matching this query, or does not exist."
            return JSONResponse(content={"answer": msg, "source_document": "None", "doc": "None"})

        context_str = "\n\n".join([d.page_content for d in source_docs])
        
        final_answer = await asyncio.to_thread(
            qa_chain.invoke, 
            {"context": context_str, "question": query}
        )

        return JSONResponse(content={
            "answer": f"**Retrieved Context:**\n{context_str}\n\n**Generated Report:**\n{final_answer}",
            "source_document": source_docs[0].page_content,
            "doc": source_docs[0].metadata.get("source", "Unknown"),
        })

    except Exception as exc:
        log.exception("Error during query: %s", exc)
        return JSONResponse(status_code=500, content={"error": f"An error occurred: {str(exc)}"})

@app.get("/evaluate")
async def evaluate_model():
    y_true, y_pred = [], []

    for case in EVAL_TEST_CASES:
        expected = case["expected_patient_id"]
        try:
            match = re.search(r'\b(100\d{5})\b', case["query"])
            target_patient_id = int(match.group(1)) if match else None

            search_kwargs = {"k": 1}
            if target_patient_id:
                search_kwargs["filter"] = {"patient_id": target_patient_id}

            def _search():
                return db.similarity_search(case["query"], **search_kwargs)
                
            docs = await asyncio.to_thread(_search)
            predicted = docs[0].metadata.get("patient_id") if docs else None
            y_true.append(1)
            y_pred.append(1 if predicted == expected else 0)
        except Exception:
            y_true.append(1)
            y_pred.append(0)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    os.makedirs("static", exist_ok=True)

    plt.figure(figsize=(6, 4))
    sns.barplot(
        x=["Precision", "Recall", "F1 Score"],
        y=[prec, rec, f1],
        hue=["Precision", "Recall", "F1 Score"],
        legend=False,
        palette="viridis",
    )
    plt.ylim(0, 1)
    plt.title("Retrieval Performance Metrics")
    plt.savefig("static/metrics_bar.png", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(4, 4))
    plt.pie([acc, 1 - acc], labels=["Correct", "Incorrect"], autopct="%1.1f%%", colors=["#41ab5d", "#f03b20"])
    plt.title(f"Accuracy: {acc*100:.1f}%")
    plt.savefig("static/metrics_pie.png", bbox_inches="tight")
    plt.close()

    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Retrieved", "Not Retrieved"],
                yticklabels=["Retrieved", "Not Retrieved"])
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    plt.savefig("static/metrics_cm.png", bbox_inches="tight")
    plt.close()

    return JSONResponse({
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    })

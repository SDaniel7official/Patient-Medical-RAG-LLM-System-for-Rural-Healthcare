import os
import sys
import zipfile
import glob
import logging
import pandas as pd
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Qdrant

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
ZIP_PATH       = os.getenv("ZIP_PATH",      "data/mimic-iv-demo-data-in-the-medical-event-data-standard-meds-0.0.1.zip")
EXTRACT_PATH   = os.getenv("EXTRACT_PATH",  "data/mimic")
QDRANT_PATH    = os.getenv("QDRANT_PATH",   "local_qdrant")
COLLECTION     = os.getenv("COLLECTION",    "patient_records")
EMBED_MODEL    = os.getenv("EMBED_MODEL",   "NeuML/pubmedbert-base-embeddings")
CHUNK_SIZE     = int(os.getenv("CHUNK_SIZE",    "1000"))
CHUNK_OVERLAP  = int(os.getenv("CHUNK_OVERLAP", "200"))
MAX_PATIENTS   = int(os.getenv("MAX_PATIENTS",  "100"))   # 0 = all
MAX_EVENTS     = int(os.getenv("MAX_EVENTS",    "500"))   # per patient

# --------------------------------------------------------------------------- #
#  Extraction
# --------------------------------------------------------------------------- #
if not os.path.exists(EXTRACT_PATH):
    log.info("Extracting dataset …")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(EXTRACT_PATH)
    log.info("Extraction complete.")

# Validate extraction completeness
parquet_files = glob.glob(f"{EXTRACT_PATH}/**/*.parquet", recursive=True)
data_parquets = [f for f in parquet_files if "metadata" not in f.replace("\\", "/").split("/")]
if len(data_parquets) < 3:
    log.error("Expected ≥3 data parquet files; found %d. Check the zip.", len(data_parquets))
    sys.exit(1)

# --------------------------------------------------------------------------- #
#  Guard against accidental overwrite
# --------------------------------------------------------------------------- #
FORCE = "--force" in sys.argv
if os.path.exists(QDRANT_PATH) and os.listdir(QDRANT_PATH):
    if not FORCE:
        answer = input(
            f"Qdrant collection at '{QDRANT_PATH}' already exists. "
            "Overwrite? [y/N]: "
        ).strip().lower()
        if answer != "y":
            log.info("Aborted by user.")
            sys.exit(0)
    log.info("Overwriting existing collection …")
    import shutil
    shutil.rmtree(QDRANT_PATH)

# --------------------------------------------------------------------------- #
#  Load parquet data
# --------------------------------------------------------------------------- #
log.info("Loading %d parquet file(s) …", len(data_parquets))
df = pd.concat(
    [pd.read_parquet(f) for f in data_parquets],
    ignore_index=True,
)
df = df.dropna(subset=["code"])

# Load code descriptions (human-readable labels)
codes_file = glob.glob(f"{EXTRACT_PATH}/**/codes.parquet", recursive=True)
code_desc: dict = {}
if codes_file:
    codes_df = pd.read_parquet(codes_file[0])
    if "code" in codes_df.columns and "description" in codes_df.columns:
        code_desc = dict(zip(codes_df["code"], codes_df["description"]))
    log.info("Loaded %d code descriptions.", len(code_desc))
else:
    log.warning("codes.parquet not found — raw codes will be used.")

# --------------------------------------------------------------------------- #
#  Build patient documents
# --------------------------------------------------------------------------- #
log.info("Grouping events by patient …")
grouped = df.sort_values(by=["subject_id", "time"]).groupby("subject_id")
total_groups = len(grouped)

documents: list[Document] = []
has_text = "text_value" in df.columns
has_num  = "numeric_value" in df.columns

for i, (subject_id, group) in enumerate(grouped):
    if MAX_PATIENTS and i >= MAX_PATIENTS:
        break

    # Truncate to MAX_EVENTS per patient
    group = group.head(MAX_EVENTS)

    history_lines = [f"Patient ID: {subject_id}"]

    for row in group.itertuples():
        time_str = str(row.time) if pd.notnull(row.time) else "Unknown Time"
        code     = str(row.code) if pd.notnull(row.code) else ""
        desc     = code_desc.get(code, "")
        label    = f"{code} ({desc})" if desc else code

        text_val = ""
        if has_text and pd.notnull(getattr(row, "text_value", None)):
            text_val = str(getattr(row, "text_value"))

        num_val = ""
        if has_num and pd.notnull(getattr(row, "numeric_value", None)):
            num_val = str(getattr(row, "numeric_value"))

        event_str = f"- Time: {time_str}, Code: {label}"
        if text_val:
            event_str += f", Text: {text_val}"
        if num_val:
            event_str += f", Value: {num_val}"
        history_lines.append(event_str)

    documents.append(Document(
        page_content="\n".join(history_lines),
        metadata={"patient_id": int(subject_id), "source": f"Patient {subject_id}"},
    ))

    if (i + 1) % 10 == 0:
        log.info("Processed %d / %d patients …", i + 1, total_groups)

log.info("Created %d patient documents.", len(documents))

# --------------------------------------------------------------------------- #
#  Chunking
# --------------------------------------------------------------------------- #
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
texts = splitter.split_documents(documents)

# Ensure patient ID header is in every chunk
for chunk in texts:
    header = f"Patient ID: {chunk.metadata['patient_id']}"
    if header not in chunk.page_content:
        chunk.page_content = header + "\n" + chunk.page_content

log.info("Split into %d chunks. Generating embeddings …", len(texts))

# --------------------------------------------------------------------------- #
#  Embeddings + Qdrant
# --------------------------------------------------------------------------- #
embeddings = SentenceTransformerEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"batch_size": 32},
)

Qdrant.from_documents(
    texts,
    embeddings,
    path=QDRANT_PATH,
    collection_name=COLLECTION,
    force_recreate=True,
)

log.info("Vector DB successfully created at '%s' (collection: %s).", QDRANT_PATH, COLLECTION)
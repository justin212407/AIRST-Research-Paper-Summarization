import os
import uuid
import json
import re
import requests

# Ensure torch._classes is initialized to avoid AttributeError in some environments
import torch
import types
if hasattr(torch, "_classes"):
    torch._classes = types.SimpleNamespace()

import streamlit as st
from pathlib import Path

# --- Document Extraction Libraries ---
import fitz  # PyMuPDF for PDF extraction
from docx import Document  # for DOCX extraction

# Importing pdfplumber for improved table extraction
try:
    import pdfplumber
    USE_PDFPLUMBER = True
except ImportError:
    USE_PDFPLUMBER = False

# --- Embedding Model ---
from sentence_transformers import SentenceTransformer

# --- ChromaDB for Vector Storage ---
from chromadb import PersistentClient
from chromadb.config import Settings

# ---------- Global Setup ----------
# Directory to save uploaded files
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Directory for persistent vector DB
CHROMA_DIR = "chroma_db"
os.makedirs(CHROMA_DIR, exist_ok=True)

# File for persistent mapping between original filename and unique filename
PERSISTENCE_FILE = "processed_files.json"

# Initialize persistent ChromaDB client so collections survive app restarts
chroma_client = PersistentClient(path=CHROMA_DIR, settings=Settings())

# Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------- Persistence Functions ----------
def load_processed_files():
    if os.path.exists(PERSISTENCE_FILE):
        with open(PERSISTENCE_FILE, "r") as f:
            return json.load(f)
    return {}

def save_processed_files(mapping):
    with open(PERSISTENCE_FILE, "w") as f:
        json.dump(mapping, f)

# ---------- Helper Functions ----------
def extract_text_from_pdf_pymupdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        blocks = page.get_text("blocks")
        blocks = sorted(blocks, key=lambda b: (b[1], b[0]))
        page_text = "\n".join(b[4].strip() for b in blocks if len(b) > 4 and b[4].strip())
        text += page_text + "\n\n"

    # Repair common PDF extraction artifacts.
    text = re.sub(r"-\n(?=\w)", "", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text

def extract_text_from_pdf(file_path):
    full_text = ""
    if USE_PDFPLUMBER:
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    table_text = ""
                    tables = page.extract_tables() or []
                    for table in tables:
                        for row in table:
                            table_text += "\t".join([str(cell) for cell in row if cell]) + "\n"
                    full_text += page_text + "\n" + table_text + "\n"
        except Exception as e:
            st.warning(f"pdfplumber extraction failed: {e}. Falling back to PyMuPDF.")
            full_text = extract_text_from_pdf_pymupdf(file_path)
    else:
        full_text = extract_text_from_pdf_pymupdf(file_path)
    return full_text

def extract_text_from_docx(file_path):
    doc = Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs])

def chunk_text_improved(text, max_chunk_chars=1000, overlap_chars=200):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 <= max_chunk_chars:
            current_chunk += para + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
                overlap = current_chunk[-overlap_chars:] if len(current_chunk) > overlap_chars else current_chunk
                current_chunk = overlap + para + "\n\n"
            else:
                current_chunk = para[:max_chunk_chars]
                chunks.append(current_chunk.strip())
                current_chunk = para[max_chunk_chars:]
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    return chunks

def clean_text_for_display(text):
    text = text.replace("\n", " ")
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([.,;:!?])(\S)", r"\1 \2", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text) if s.strip()]

def remove_front_matter_noise(text):
    cleaned = clean_text_for_display(text)
    cleaned = re.sub(r"\S+@\S+", " ", cleaned)
    cleaned = re.sub(r"\b(Google Research|Google Brain|University of Toronto|Conference on Neural Information Processing Systems)\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

def extract_section(text, start_labels, end_labels):
    lower = text.lower()
    start_idx = -1
    for label in start_labels:
        idx = lower.find(label.lower())
        if idx != -1:
            start_idx = idx
            break
    if start_idx == -1:
        return ""

    end_idx = len(text)
    for label in end_labels:
        idx = lower.find(label.lower(), start_idx + 1)
        if idx != -1:
            end_idx = min(end_idx, idx)

    return text[start_idx:end_idx].strip()

def pick_readable_sentences(text, max_sentences=3):
    picked = []
    for sentence in split_sentences(text):
        words = sentence.split()
        if len(words) < 8 or len(words) > 55:
            continue
        avg_word_len = sum(len(w) for w in words) / max(len(words), 1)
        if avg_word_len > 9.5:
            continue
        if "@" in sentence or "http" in sentence.lower():
            continue
        if re.search(r"\b(equal contribution|listing order is random|work performed while)\b", sentence, re.IGNORECASE):
            continue
        picked.append(sentence)
        if len(picked) >= max_sentences:
            break
    return picked

def extract_title_from_text(text):
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    for line in lines[:20]:
        line = re.sub(r"\S+@\S+", "", line).strip()
        if 4 <= len(line.split()) <= 18 and not re.search(r"\babstract\b", line, re.IGNORECASE):
            if not re.search(r"\b(google|university|conference|department)\b", line, re.IGNORECASE):
                return clean_text_for_display(line)
    return "Research Paper"

def heuristic_paper_summary(raw_text):
    cleaned = remove_front_matter_noise(raw_text)
    abstract = extract_section(cleaned, ["abstract"], ["1 introduction", "introduction", "keywords"])
    intro = extract_section(cleaned, ["1 introduction", "introduction"], ["2 background", "background", "related work", "method", "model"])
    method = extract_section(cleaned, ["method", "model architecture", "approach", "proposed method"], ["results", "experiments", "evaluation", "discussion", "conclusion"])
    results = extract_section(cleaned, ["results", "experiments", "evaluation"], ["discussion", "conclusion", "references"])
    limitations = extract_section(cleaned, ["limitations", "future work", "discussion"], ["conclusion", "references"])
    conclusion = extract_section(cleaned, ["conclusion", "5 conclusion", "6 conclusion"], ["references", "acknowledgements"])

    if not abstract:
        abstract = cleaned[:2200]
    if not intro:
        intro = cleaned[2200:4600] if len(cleaned) > 2200 else cleaned

    abstract_points = pick_readable_sentences(abstract, max_sentences=4)
    intro_points = pick_readable_sentences(intro, max_sentences=4)
    method_points = pick_readable_sentences(method, max_sentences=4) if method else []
    result_points = pick_readable_sentences(results, max_sentences=5) if results else []
    limitation_points = pick_readable_sentences(limitations, max_sentences=3) if limitations else []
    conclusion_points = pick_readable_sentences(conclusion, max_sentences=3) if conclusion else []

    if not abstract_points and not intro_points:
        return fallback_summary_from_text(cleaned, max_paragraphs=3, paragraph_size=500)

    lines = [f"## Detailed Research Summary\n\n### Title\n{extract_title_from_text(raw_text)}"]

    lines.append("### Abstract Snapshot")
    for item in (abstract_points[:3] or ["The paper introduces a research problem, proposes a method, and reports empirical outcomes."]):
        lines.append(f"- {item}")

    lines.append("### Research Objective")
    for item in (intro_points[:2] or abstract_points[:2] or ["The work aims to improve model quality and/or computational efficiency over prior methods."]):
        lines.append(f"- {item}")

    lines.append("### Methodology")
    for item in (method_points[:3] or intro_points[1:3] or ["A novel architecture or training strategy is proposed and evaluated on benchmark tasks."]):
        lines.append(f"- {item}")

    lines.append("### Experimental Setup")
    setup_candidates = pick_readable_sentences(cleaned, max_sentences=8)
    for item in (setup_candidates[:2] or ["Experiments are conducted on established benchmarks and compared against strong baselines."]):
        lines.append(f"- {item}")

    lines.append("### Key Findings")
    findings = result_points[:4] or abstract_points[1:4] or ["The reported metrics indicate the proposed method outperforms or matches prior approaches."]
    for item in findings:
        lines.append(f"- {item}")

    lines.append("### Limitations / Risks")
    for item in (limitation_points[:2] or ["The paper may have task-domain limitations, dataset-specific assumptions, or compute/training trade-offs."]):
        lines.append(f"- {item}")

    lines.append("### Conclusion")
    for item in (conclusion_points[:2] or ["Overall, the approach demonstrates meaningful improvements and motivates further work."]):
        lines.append(f"- {item}")

    lines.append("### Practical Takeaways")
    lines.append("- Use this paper's method when balancing quality and efficiency is critical.")
    lines.append("- Validate performance on your own dataset/domain before production use.")
    lines.append("- Review compute and implementation complexity before adoption.")

    return "\n".join(lines)

def call_llm_summary(context):
    api_key = get_openrouter_api_key()
    if not api_key:
        return "API Key Missing"

    try:
        site_url = normalize_site_url(st.secrets.get("SITE_URL", "https://example.com"))
        site_name = st.secrets.get("SITE_NAME", "My Site")
    except Exception:
        site_url = "https://example.com"
        site_name = "My Site"

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": site_url,
        "X-Title": site_name,
        "Content-Type": "application/json"
    }

    prompt = (
        "Create a DETAILED research-paper summary in markdown.\n"
        "Required sections in this exact order:\n"
        "1) Title\n"
        "2) Abstract Snapshot\n"
        "3) Research Objective\n"
        "4) Methodology\n"
        "5) Experimental Setup\n"
        "6) Key Findings\n"
        "7) Limitations / Risks\n"
        "8) Conclusion\n"
        "9) Practical Takeaways\n\n"
        "Formatting rules:\n"
        "- Use clear headings (###).\n"
        "- Use bullet points under each section (2-6 bullets each where possible).\n"
        "- Keep writing clean and readable, around 450-900 words total.\n"
        "- Exclude author emails, affiliations, footnotes, and metadata noise.\n"
        "- Do not copy raw OCR-like text; rewrite naturally.\n\n"
        f"Context:\n{context}"
    )

    for model in get_openrouter_models():
        data = {
            "model": model,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": "You are a precise research-paper summarizer."},
                {"role": "user", "content": prompt}
            ]
        }
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data), timeout=90)
        except requests.RequestException:
            continue

        if response.status_code == 200:
            try:
                st.session_state["last_openrouter_model"] = model
                return response.json()["choices"][0]["message"]["content"]
            except Exception:
                continue
    return "Error from API: all configured OpenRouter models failed"

def is_low_quality_summary(summary):
    if not summary:
        return True

    lower = summary.lower()
    if summary.startswith("API Key Missing") or summary.startswith("Error from API") or summary.startswith("Error parsing"):
        return True
    if "no relevant information found in the provided documents" in lower:
        return True
    if lower.count("@") > 1:
        return True
    if len(summary.split()) < 140:
        return True
    required_sections = [
        "title", "abstract", "objective", "method", "findings", "conclusion"
    ]
    matched = sum(1 for sec in required_sections if sec in lower)
    if matched < 4:
        return True
    return False

def fallback_summary_from_text(text, max_paragraphs=3, paragraph_size=650):
    cleaned = clean_text_for_display(text)
    if not cleaned:
        return ""

    paragraphs = []
    start = 0
    text_len = len(cleaned)
    while start < text_len and len(paragraphs) < max_paragraphs:
        end = min(start + paragraph_size, text_len)
        if end < text_len:
            split = cleaned.rfind(" ", start, end)
            if split > start:
                end = split
        paragraphs.append(cleaned[start:end].strip())
        start = end
    return "\n\n".join(paragraphs)

def get_existing_collection_names():
    try:
        return {c.name for c in chroma_client.list_collections()}
    except Exception:
        return set()

def index_file_into_collection(file_path, unique_filename):
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == ".pdf":
        text = extract_text_from_pdf(file_path)
    elif file_extension in [".doc", ".docx"]:
        text = extract_text_from_docx(file_path)
    else:
        return False

    if not text.strip():
        return False

    chunks = chunk_text_improved(text)
    if not chunks:
        return False

    collection = chroma_client.get_or_create_collection(name=unique_filename)
    existing_count = collection.count()
    if existing_count > 0:
        return True

    embeddings = embed_model.encode(chunks).tolist()
    doc_ids = [str(i) for i in range(len(chunks))]
    collection.add(documents=chunks, embeddings=embeddings, ids=doc_ids)
    return True

def ensure_vector_store_ready(processed_files_mapping):
    existing = get_existing_collection_names()
    for _, unique_filename in processed_files_mapping.items():
        if unique_filename in existing:
            continue
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        if os.path.exists(file_path):
            index_file_into_collection(file_path, unique_filename)

def process_file(uploaded_file):
    file_extension = os.path.splitext(uploaded_file.name)[1]
    unique_filename = f"{uuid.uuid4().hex}{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if file_extension.lower() not in [".pdf", ".doc", ".docx"]:
        st.error("Unsupported file type.")
        return None

    ok = index_file_into_collection(file_path, unique_filename)
    if not ok:
        st.warning(f"No text could be extracted from {uploaded_file.name}.")
        return None
    return unique_filename

def delete_file(unique_filename):
    file_path = os.path.join(UPLOAD_DIR, unique_filename)
    if os.path.exists(file_path):
        os.remove(file_path)
    try:
        chroma_client.delete_collection(name=unique_filename)
    except Exception as e:
        st.error(f"Error deleting collection: {e}")

def search_documents(query, top_k=5):
    results = []
    try:
        collections = chroma_client.list_collections()
    except Exception as e:
        st.error(f"Failed to list collections: {e}")
        return results

    query_embedding = embed_model.encode([query]).tolist()[0]

    for col in collections:
        name = col.name
        try:
            coll = chroma_client.get_collection(name=name)
            count = coll.count()
            if count == 0:
                continue
            n_results = min(top_k, count)
            search_result = coll.query(query_embeddings=[query_embedding], n_results=n_results)
            docs = search_result.get("documents", [[]])[0]
            distances = search_result.get("distances", [[]])[0]
            for doc, distance in zip(docs, distances):
                results.append((name, doc, distance))
        except Exception as e:
            st.error(f"Error querying collection {name}: {e}")
    results.sort(key=lambda x: x[2])
    return results

def normalize_openrouter_key(raw_key):
    if not raw_key:
        return None
    key = str(raw_key).strip().strip('"').strip("'")
    if key.lower().startswith("bearer "):
        key = key[7:].strip()
    return key or None

def mask_key(key):
    if not key:
        return "not-set"
    if len(key) <= 10:
        return "*" * len(key)
    return f"{key[:6]}...{key[-4:]}"

def normalize_site_url(url_value):
    if not url_value:
        return "https://example.com"
    url = str(url_value).strip()
    if not url.startswith("http://") and not url.startswith("https://"):
        url = f"https://{url}"
    return url

def get_openrouter_models():
    try:
        model_from_secrets = st.secrets.get("OPENROUTER_MODEL")
    except Exception:
        model_from_secrets = None
    model_from_env = os.getenv("OPENROUTER_MODEL")
    configured = normalize_openrouter_key(model_from_secrets) or normalize_openrouter_key(model_from_env)

    models = []
    if configured:
        for m in configured.split(","):
            m = m.strip()
            if m:
                models.append(m)

    # Known-working defaults for most accounts.
    defaults = ["qwen/qwq-32b", "openai/gpt-4o-mini"]
    for m in defaults:
        if m not in models:
            models.append(m)
    return models

def get_openrouter_api_key():
    try:
        key_from_secrets = st.secrets.get("OPENROUTER_API_KEY")
    except Exception:
        key_from_secrets = None
    key_from_env = os.getenv("OPENROUTER_API_KEY")
    return normalize_openrouter_key(key_from_secrets) or normalize_openrouter_key(key_from_env)

def get_openrouter_key_source():
    try:
        if normalize_openrouter_key(st.secrets.get("OPENROUTER_API_KEY")):
            return "Streamlit secrets"
    except Exception:
        pass
    if normalize_openrouter_key(os.getenv("OPENROUTER_API_KEY")):
        return "Environment variable"
    return "Not configured"

def test_openrouter_connection():
    api_key = get_openrouter_api_key()
    if not api_key:
        return False, "OPENROUTER_API_KEY is not configured in Streamlit secrets or environment variables."

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://example.com",
        "X-Title": "AIRST Test",
        "Content-Type": "application/json"
    }
    errors = []
    for model in get_openrouter_models():
        data = {
            "model": model,
            "temperature": 0,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "Reply with OK"}]
        }
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data), timeout=45)
        except requests.RequestException as e:
            errors.append(f"{model}: {e}")
            continue

        if response.status_code == 200:
            st.session_state["last_openrouter_model"] = model
            return True, f"OpenRouter key is valid and reachable. Model: {model}"

        try:
            err = response.json()
        except Exception:
            err = response.text
        errors.append(f"{model}: HTTP {response.status_code} - {err}")

    return False, "OpenRouter test failed for all configured models: " + " | ".join(errors[:2])

def local_answer_from_context(context, question, max_sentences=4):
    if not context.strip():
        return "No relevant information found in the provided documents."

    q = question.lower()
    q_terms = set(re.findall(r"[a-zA-Z]{3,}", q))
    stop_terms = {"what", "which", "who", "when", "where", "why", "how", "about", "paper", "research", "this", "that"}
    q_terms = {t for t in q_terms if t not in stop_terms}

    cleaned = clean_text_for_display(context)

    # Special handling for authorship-style questions.
    if any(term in q for term in ["who wrote", "who has written", "author", "authors", "written by"]):
        first_part = context[:3500]
        email_name_pattern = r"([A-Z][a-zA-Z\.-]+(?:\s+[A-Z][a-zA-Z\.-]+){1,2})\s+\S+@\S+"
        candidates = re.findall(email_name_pattern, first_part)

        if not candidates:
            name_pattern = r"\b([A-Z][a-zA-Z\.-]{1,14}(?:\s+[A-Z][a-zA-Z\.-]{1,14}){1,2})\b"
            candidates = re.findall(name_pattern, first_part)

        blacklist = {
            "Attention Is", "Google Brain", "Google Research", "University Of", "Neural Information",
            "Conference On", "Abstract The", "Introduction Recurrent"
        }
        names = []
        for c in candidates:
            c = c.strip()
            if c in blacklist:
                continue
            if any(org in c for org in ["Google", "University", "Conference", "Abstract", "Introduction"]):
                continue
            if any(len(token) > 14 for token in c.split()):
                continue
            if len(c.split()) < 2:
                continue
            if c not in names:
                names.append(c)
            if len(names) >= 8:
                break

        if names:
            return "### Local Answer\n\n**Question:** Who wrote this paper?\n\n**Authors found in the indexed text:**\n" + "\n".join([f"- {n}" for n in names])
        return "### Local Answer\n\nI could not reliably extract author names from the indexed text for this paper."

    sentences = split_sentences(cleaned)
    scored = []
    for s in sentences:
        s_terms = set(re.findall(r"[a-zA-Z]{3,}", s.lower()))
        overlap = len(q_terms & s_terms) if q_terms else 0
        if overlap == 0 and q_terms:
            continue
        # Prefer compact informative sentences.
        word_count = len(s.split())
        length_penalty = 0.02 * max(0, word_count - 35)
        score = overlap - length_penalty
        if any(k in s.lower() for k in ["propose", "introduce", "show", "results", "bleu", "conclusion", "transformer"]):
            score += 0.3
        scored.append((score, s))

    if not scored:
        fallback = pick_readable_sentences(cleaned, max_sentences=max_sentences)
        if fallback:
            bullets = "\n".join([f"- {s}" for s in fallback])
            return f"### Local Answer\n\n**Question:** {question}\n\n**Best matching points from the document:**\n{bullets}"
        return "No relevant information found in the provided documents."

    scored.sort(key=lambda x: x[0], reverse=True)
    selected = [s for _, s in scored[:max_sentences]]
    bullets = "\n".join([f"- {s}" for s in selected])
    return f"### Local Answer\n\n**Question:** {question}\n\n**Best matching points from the document:**\n{bullets}"

def call_llm(context, question):
    api_key = get_openrouter_api_key()
    if not api_key:
        st.session_state["last_answer_source"] = "Local fallback (no API key configured)"
        st.info("Using local answer mode because OPENROUTER_API_KEY is not configured.")
        return local_answer_from_context(context, question)
    url = "https://openrouter.ai/api/v1/chat/completions"
    try:
        site_url = normalize_site_url(st.secrets.get("SITE_URL", "https://example.com"))
        site_name = st.secrets.get("SITE_NAME", "My Site")
    except Exception:
        site_url = "https://example.com"
        site_name = "My Site"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": site_url,
        "X-Title": site_name,
        "Content-Type": "application/json"
    }
    message = (
        "You are an AI assistant that answers questions solely based on the provided context extracted from uploaded research papers. "
        "Answer the following question using only the information available in the context. "
        "If the provided context does not contain sufficient or relevant details to answer the question, "
        "respond exactly with: 'No relevant information found in the provided documents.'\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )
    last_status = None
    for model in get_openrouter_models():
        data = {"model": model, "messages": [{"role": "user", "content": message}]}
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data), timeout=90)
        except requests.RequestException:
            continue

        last_status = response.status_code
        if response.status_code == 200:
            try:
                st.session_state["last_answer_source"] = f"OpenRouter ({model})"
                st.session_state["last_openrouter_model"] = model
                return response.json()["choices"][0]["message"]["content"]
            except Exception:
                continue

        if response.status_code in [401, 403]:
            st.session_state["last_answer_source"] = "Local fallback (OpenRouter auth failed)"
            st.warning("OpenRouter authentication failed (invalid API key). Using local answer mode.")
            return local_answer_from_context(context, question)

    st.session_state["last_answer_source"] = f"Local fallback (OpenRouter HTTP {last_status if last_status else 'request-failed'})"
    st.warning("OpenRouter returned non-success status for configured models. Using local answer mode.")
    return local_answer_from_context(context, question)

# ---------- Streamlit Application ----------

def main():
    st.title("AI Research Paper Summarizer")
    st.write("Upload research papers, ask questions, and get answers from relevant document sections.")

    # Load persistent mapping into session state on app start
    if "processed_files" not in st.session_state:
        st.session_state["processed_files"] = load_processed_files()

    if not st.session_state.get("vector_store_ready", False):
        ensure_vector_store_ready(st.session_state["processed_files"])
        st.session_state["vector_store_ready"] = True

    with st.sidebar:
        st.subheader("OpenRouter Status")
        key = get_openrouter_api_key()
        key_source = get_openrouter_key_source()
        if key:
            st.success("API key detected")
            st.caption(f"Source: {key_source}")
            st.caption(f"Key: {mask_key(key)}")
        else:
            st.warning("API key not detected")
        if st.button("Test OpenRouter Connection"):
            ok, message = test_openrouter_connection()
            if ok:
                st.success(message)
            else:
                st.error(message)

    # Tabs: File Upload, PDFs/Docs list, and Prompt for Q&A.
    tab_upload, tab_list, tab_prompt, tab_chat = st.tabs(["File Upload", "PDFs/Docs", "Prompt","Upload & Chat"])
    
    with tab_upload:
        st.header("Upload Research Papers")
        uploaded_files = st.file_uploader(
            "Upload PDF or DOCX files", 
            type=["pdf", "doc", "docx"], 
            accept_multiple_files=True
        )
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in st.session_state["processed_files"]:
                    unique_filename = process_file(uploaded_file)
                    if unique_filename:
                        st.success(f"Uploaded and processed file: {uploaded_file.name}")
                        st.session_state["processed_files"][uploaded_file.name] = unique_filename
                        save_processed_files(st.session_state["processed_files"])
    
    with tab_list:
        st.header("Uploaded Files")
        processed_files = st.session_state.get("processed_files", {})
        if processed_files:
            for original_name, unique_filename in list(processed_files.items()):
                st.write(f"**{original_name}**")
                if st.button(f"Delete {original_name}", key=f"delete_{unique_filename}"):
                    delete_file(unique_filename)
                    st.success(f"Deleted {original_name}")
                    del st.session_state["processed_files"][original_name]
                    save_processed_files(st.session_state["processed_files"])
        else:
            st.info("No files uploaded yet.")
    
    with tab_prompt:
        st.header("Ask a Question")
        query = st.text_input("Enter your question:")
        if st.button("Get Answer"):
            if query:
                search_results = search_documents(query)
                if search_results:
                    context = "\n\n".join([doc for _, doc, _ in search_results[:5]])
                else:
                    context = ""
                if not context:
                    st.error("No relevant content found from uploaded documents. Please check your upload and try a different query.")
                else:
                    answer = call_llm(context, query)
                    st.write("**Answer:**")
                    st.markdown(answer)
                    if st.session_state.get("last_answer_source"):
                        st.caption(f"Answer source: {st.session_state['last_answer_source']}")
            else:
                st.warning("Please enter a question.")
    with tab_chat:
        st.header("Upload & Chat with PDF")
        chat_uploaded_file = st.file_uploader(
            "Upload a PDF file for summarization and chat (file will not be stored):",
            type=["pdf"]
        )
        if chat_uploaded_file:
            # Extract text from the uploaded PDF
            temp_file_path = os.path.join(UPLOAD_DIR, f"temp_{uuid.uuid4().hex}.pdf")
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(chat_uploaded_file.getbuffer())
            
            # Extract text and summarize
            extracted_text = extract_text_from_pdf(temp_file_path)
            os.remove(temp_file_path)  # Delete the temporary file immediately
            
            if not extracted_text.strip():
                st.warning("No text could be extracted from the uploaded PDF.")
            else:
                st.subheader("Summary")
                chunks = chunk_text_improved(extracted_text)
                context = "\n\n".join(chunks[:20])
                summary = call_llm_summary(context)

                if is_low_quality_summary(summary):
                    st.markdown(heuristic_paper_summary(extracted_text))
                else:
                    st.markdown(summary)

if __name__ == "__main__":
    main()
"""
Self-Improving Agentic RAG — Streamlit Cloud version
=====================================================
A text RAG agent that retrieves, generates, judges its own answer, and if the
answer is weak, diagnoses why and applies a corrective action — looping until
the answer is good or a retry limit is hit. It also stores "lessons" from
corrections (score-gated) and injects similar ones into future prompts.

Runs on Streamlit Cloud: generation is via the Groq API (no local GPU model),
Chroma is in-memory (no disk writes), and simple caching is applied to the
deterministic helper steps.

Deploy:
  1. requirements.txt (see bottom of this file for the list)
  2. Add GROQ_API_KEY in Streamlit Cloud -> App -> Settings -> Secrets
"""

import os
import re
import sqlite3
import datetime
import tempfile

import streamlit as st
from groq import Groq

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
try:
    from langchain.retrievers import EnsembleRetriever
except ImportError:
    from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
EMBED_MODEL     = "BAAI/bge-small-en-v1.5"
GROQ_MODEL      = "openai/gpt-oss-120b"     # current Groq model id
DB_PATH         = os.path.join(tempfile.gettempdir(), "lessons.db")  # ephemeral
SCORE_THRESHOLD = 0.7
MAX_RETRIES     = 2

# Groq key: Streamlit Cloud uses st.secrets; fall back to env var for local runs.
def _get_groq_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.environ.get("GROQ_API_KEY", "")

GROQ_KEY = _get_groq_key()

st.set_page_config(page_title="Self-Improving Agentic RAG", page_icon="brain", layout="wide")

# ----------------------------------------------------------------------------
# Cached resources (load once per session)
# ----------------------------------------------------------------------------
@st.cache_resource
def get_embedder():
    # CPU only on Streamlit Cloud
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL, model_kwargs={"device": "cpu"})

@st.cache_resource
def get_client():
    return Groq(api_key=GROQ_KEY)

# simple in-process cache for deterministic helper calls (triage, contextualize)
_CACHE = {}

# ----------------------------------------------------------------------------
# LLM call
# ----------------------------------------------------------------------------
def generate(prompt):
    client = get_client()
    resp = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=GROQ_MODEL, temperature=0.3, max_completion_tokens=512)
    return resp.choices[0].message.content.strip()

# ----------------------------------------------------------------------------
# Retrieval (hybrid: semantic + BM25), multi-PDF with source tagging
# ----------------------------------------------------------------------------
def build_retriever(uploaded_files):
    """Build an in-memory hybrid retriever from one or more uploaded PDFs."""
    embedder = get_embedder()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    chunks = []
    for uf in uploaded_files:
        # write the upload to a temp file so PyPDFLoader can read it
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uf.getbuffer())
            tmp_path = tmp.name
        docs = PyPDFLoader(tmp_path).load()
        doc_chunks = splitter.split_documents(docs)
        for ch in doc_chunks:
            ch.metadata["source"] = uf.name          # MULTI-PDF: tag each chunk
        chunks += doc_chunks
        os.unlink(tmp_path)

    vectorstore = Chroma.from_documents(chunks, embedder)  # in-memory, no disk
    return {"vs": vectorstore, "chunks": chunks}

def retrieve(query, retriever, k=4):
    """Hybrid search; returns chunk texts prefixed with [source]."""
    vs, chunks = retriever["vs"], retriever["chunks"]
    semantic = vs.as_retriever(search_kwargs={"k": k})
    keyword  = BM25Retriever.from_documents(chunks); keyword.k = k
    hybrid   = EnsembleRetriever(retrievers=[semantic, keyword], weights=[0.6, 0.4])
    out = []
    for d in hybrid.invoke(query):
        src = d.metadata.get("source", "doc")
        out.append(f"[{src}] {d.page_content}")
    return out

# ----------------------------------------------------------------------------
# Judge (faithfulness, deduction rubric) + non-answer detector
# ----------------------------------------------------------------------------
def _looks_like_nonanswer(answer):
    a = answer.lower()
    cues = ["does not provide", "doesn't provide", "no information", "not mentioned",
            "not provided", "cannot answer", "can't answer", "context does not",
            "unable to answer", "no relevant information", "not available in the context",
            "not enough information"]
    return any(c in a for c in cues)

def judge_score(answer, question, context):
    if _looks_like_nonanswer(answer):
        return 0.2
    prompt = (
        "You are a STRICT grader for a RAG answer. Be critical, not generous.\n"
        "First list every PROBLEM (one per line, prefixed 'PROBLEM:'): unsupported claims, "
        "vagueness, unanswered parts, or info not in the context. If none, write 'PROBLEM: none'.\n"
        "Then start at 1.0 and deduct: -0.3 per unsupported claim, -0.2 for vagueness/incompleteness, "
        "-0.2 for added info not in context.\n"
        "End with exactly one line: SCORE: <0.0-1.0>\n\n"
        f"Question: {question}\n\nContext:\n{chr(10).join(context)}\n\nAnswer: {answer}"
    )
    raw = generate(prompt)
    m = re.search(r"score\s*[:=]\s*(\d+\.?\d*)", raw, re.IGNORECASE) or re.search(r"(\d\.\d+)", raw)
    if not m:
        return 0.5
    s = float(m.group(1))
    if s > 1.0:
        s = s / 10.0 if s <= 10 else s / 100.0
    return round(min(max(s, 0.0), 1.0), 3)

# ----------------------------------------------------------------------------
# Lesson memory (SQLite) with score-gated saving
# ----------------------------------------------------------------------------
def init_db():
    with sqlite3.connect(DB_PATH) as c:
        c.execute("""CREATE TABLE IF NOT EXISTS lessons (
            id INTEGER PRIMARY KEY AUTOINCREMENT, query TEXT, issue TEXT, fix TEXT,
            score_before REAL, score_after REAL, timestamp TEXT)""")

def save_lesson(query, issue, fix, score_before, score_after):
    with sqlite3.connect(DB_PATH) as c:
        c.execute("INSERT INTO lessons (query,issue,fix,score_before,score_after,timestamp) "
                  "VALUES (?,?,?,?,?,?)",
                  (query, issue, fix, score_before, score_after,
                   datetime.datetime.now().isoformat()))

def get_lessons(query, limit=3):
    if not os.path.exists(DB_PATH):
        return []
    with sqlite3.connect(DB_PATH) as c:
        rows = c.execute("SELECT query,issue,fix FROM lessons ORDER BY id DESC LIMIT ?",
                         (limit,)).fetchall()
    return [{"query": r[0], "issue": r[1], "fix": r[2]} for r in rows]

# ----------------------------------------------------------------------------
# Agent tools: triage, rewrite, diagnose, contextualize  (cached where safe)
# ----------------------------------------------------------------------------
def triage(query):
    ckey = ("triage", query)
    if ckey in _CACHE:
        return _CACHE[ckey]
    prompt = ("Classify the question into ONE word:\n"
              "- factual: a specific fact/name/number\n"
              "- reasoning: explain/compare/synthesize\n"
              "- unanswerable: predictions/opinions/costs a document likely lacks\n\n"
              f"Question: {query}\n\nCategory:")
    raw = generate(prompt).strip().lower()
    if "unanswer" in raw:   kind = "unanswerable"
    elif "reason" in raw:   kind = "reasoning"
    else:                   kind = "factual"
    k = {"factual": 4, "reasoning": 8, "unanswerable": 6}[kind]
    _CACHE[ckey] = (kind, k)
    return kind, k

def rewrite_query(query, issue):
    prompt = ("Rewrite the question to retrieve better from a document search. "
              "Use specific terms; one line; do not answer it.\n"
              f"Problem: {issue}\nOriginal: {query}\n\nRewritten:")
    out = generate(prompt).strip().split("\n")[0]
    return out if out else query

def diagnose(query, answer, score):
    prompt = ("An answer scored low. Diagnose and pick ONE fix. Reply in three lines:\n"
              "Issue: <one sentence>\n"
              "Action: <one of: rewrite_query | widen_retrieval | regenerate>\n"
              "Fix: <one sentence>\n\n"
              f"Question: {query}\nAnswer: {answer}\nScore: {score}")
    diag = generate(prompt)
    issue, fix, action = "weak grounding", "retrieve more context", "widen_retrieval"
    for line in diag.splitlines():
        low = line.lower().strip()
        if low.startswith("issue:"):
            issue = line.split(":", 1)[1].strip()
        elif low.startswith("fix:"):
            fix = line.split(":", 1)[1].strip()
        elif low.startswith("action:"):
            a = line.split(":", 1)[1].strip().lower()
            action = "rewrite_query" if "rewrite" in a else "regenerate" if "regen" in a else "widen_retrieval"
    return issue, action, fix

def contextualize(query, history):
    if not history:
        return query, False
    ckey = ("ctx", query, tuple(h["q"] for h in history))
    if ckey in _CACHE:
        return _CACHE[ckey]
    convo = "\n".join([f"Q: {h['q']}\nA: {h['a'][:200]}" for h in history])
    prompt = ("Rewrite the latest question into a standalone question, resolving pronouns "
              "(it/its/they/that) to what they refer to. If already standalone, return unchanged. "
              "Reply with only the question.\n\n"
              f"Conversation:\n{convo}\n\nLatest: {query}\n\nStandalone:")
    out = generate(prompt).strip().split("\n")[0]
    result = (out if out else query), (out.lower() != query.lower() and len(out) > 0)
    _CACHE[ckey] = result
    return result

# ----------------------------------------------------------------------------
# The agent loop (plain while-loop; mirrors the LangGraph version's logic)
# ----------------------------------------------------------------------------
def run_agent(query, retriever, history=None):
    init_db()
    trace = []
    lessons = get_lessons(query)
    trace.append(("memory_check", f"loaded {len(lessons)} lessons"))

    # contextualize follow-ups
    if history:
        resolved, changed = contextualize(query, history)
        if changed:
            trace.append(("contextualize", f"resolved to: {resolved[:60]}"))
            query = resolved
        else:
            trace.append(("contextualize", "already standalone"))

    kind, k = triage(query)
    trace.append(("triage", f"classified '{kind}' -> k={k}"))

    active_query = query
    retries = 0
    force_regen = False
    pending_lesson = None      # score-gated memory: held until we see if the fix helped

    while True:
        ctx = retrieve(active_query, retriever, k=k)
        trace.append(("retrieve", f"{len(ctx)} chunks (k={k})"))

        hint = ""
        if lessons:
            hint = "\n\nPast lessons:\n" + "\n".join([f"- {l['issue']} -> {l['fix']}" for l in lessons])
        strict = "\nAnswer ONLY from the context. If absent, say so plainly." if force_regen else ""
        prompt = (f"Answer using only the context. Each context line starts with its [source]; "
                  f"cite the source(s) you used.{hint}{strict}\n\n"
                  f"Context:\n{chr(10).join(ctx)}\n\nQuestion: {query}")
        answer = generate(prompt)
        trace.append(("generate", "draft written"))

        score = judge_score(answer, query, ctx)
        trace.append(("judge", f"score = {score}"))

        # SCORE-GATED MEMORY: save the pending lesson only if the fix raised the score
        if pending_lesson:
            if score > pending_lesson["score_before"]:
                save_lesson(pending_lesson["query"], pending_lesson["issue"],
                            pending_lesson["fix"], pending_lesson["score_before"], score)
                trace.append(("save_lesson",
                              f"kept: {pending_lesson['score_before']} -> {score}"))
            else:
                trace.append(("save_lesson",
                              f"discarded: {pending_lesson['score_before']} -> {score}"))
            pending_lesson = None

        # decide: accept / correct / stop
        if score >= SCORE_THRESHOLD:
            trace.append(("decide", "accepted (>= threshold)"))
            break
        if retries >= MAX_RETRIES or (kind == "unanswerable" and retries >= 1):
            trace.append(("decide", "stopping: exhausted"))
            break

        # diagnose + choose corrective action; stash a pending (gated) lesson
        issue, action, fix = diagnose(query, answer, score)
        pending_lesson = {"query": query, "issue": issue, "fix": fix, "score_before": score}
        trace.append(("reflect", f"Issue: {issue}"))
        trace.append(("decide_action", f"chose: {action}"))

        force_regen = False
        if action == "rewrite_query":
            active_query = rewrite_query(query, issue)
            trace.append(("rewrite_query", f"new query: {active_query[:60]}"))
        elif action == "widen_retrieval":
            k = min(k + 4, 12)
            trace.append(("widen_retrieval", f"k -> {k}"))
        elif action == "regenerate":
            force_regen = True
            trace.append(("regenerate", "stricter grounding"))

        retries += 1

    return {"answer": answer, "score": score, "context": ctx,
            "retries": retries, "trace": trace, "kind": kind, "query": query}

# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.title("Self-Improving Agentic RAG")
st.caption("Retrieves · generates · judges its own answer · self-corrects · learns from the misses")

if not GROQ_KEY:
    st.error("No Groq API key found. Add GROQ_API_KEY in the app's Settings → Secrets.")
    st.stop()

# session state
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("1 · Load documents")
    files = st.file_uploader("Upload one or more PDFs", type="pdf", accept_multiple_files=True)
    if st.button("Build index", type="primary", disabled=not files):
        with st.spinner("Chunking, embedding, building hybrid retriever…"):
            st.session_state.retriever = build_retriever(files)
            st.session_state.history = []
        srcs = sorted({c.metadata.get("source", "?") for c in st.session_state.retriever["chunks"]})
        st.success(f"Indexed {len(st.session_state.retriever['chunks'])} chunks "
                   f"from {len(srcs)} PDF(s).")
    if st.session_state.retriever:
        st.caption("Index ready. Ask a question on the right.")

st.header("2 · Ask")
query = st.text_input("Your question", placeholder="e.g. What is the main contribution of this paper?")
go = st.button("Run agent", type="primary", disabled=not (st.session_state.retriever and query))

if go:
    with st.spinner("Retrieving → generating → judging → self-correcting…"):
        result = run_agent(query, st.session_state.retriever, st.session_state.history)
    st.session_state.history.append({"q": query, "a": result["answer"]})

    # answer + score
    color = "green" if result["score"] >= SCORE_THRESHOLD else "orange"
    st.markdown(f"### Answer  &nbsp; :{color}[score {result['score']:.2f}]")
    st.write(result["answer"])

    # decision trace
    with st.expander("Decision trace (how the agent reached this answer)"):
        for node, detail in result["trace"]:
            st.markdown(f"- **{node}** — {detail}")

    # retrieved context
    with st.expander("Retrieved context"):
        for i, c in enumerate(result["context"], 1):
            st.markdown(f"**{i}.** {c[:500]}…")

if st.session_state.history:
    with st.expander("Conversation history"):
        for turn in st.session_state.history:
            st.markdown(f"**Q:** {turn['q']}")
            st.markdown(f"**A:** {turn['a'][:300]}…")

# ----------------------------------------------------------------------------
# requirements.txt (put these lines in a file named requirements.txt):
#
# streamlit
# groq
# langchain==1.3.4
# langchain-community==0.4.2
# langchain-chroma==1.1.0
# langchain-huggingface==1.2.2
# langchain-text-splitters
# langchain-classic
# sentence-transformers
# rank_bm25
# pypdf
# chromadb
# ----------------------------------------------------------------------------

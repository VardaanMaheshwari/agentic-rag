"""
Self-Improving Agentic RAG — LangGraph version (Streamlit Cloud)
================================================================
This is the LangGraph implementation: an explicit StateGraph with nodes
(contextualize, triage, retrieve, generate, judge, reflect, web_fallback) and
conditional edges that route continue / correct / web / stop.

Runs on Streamlit Cloud: generation via Groq (no local GPU), Chroma in-memory,
simple caching on deterministic helpers, score-gated lesson memory, multi-PDF.

Secrets (App → Settings → Secrets):
  GROQ_API_KEY   = "..."         (required)
  TAVILY_API_KEY = "..."         (optional; web fallback no-ops if absent)
"""

import os
import re
import sqlite3
import datetime
import tempfile
from typing import TypedDict

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

from langgraph.graph import StateGraph, END

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
EMBED_MODEL     = "BAAI/bge-small-en-v1.5"
GROQ_MODEL      = "openai/gpt-oss-120b"
DB_PATH         = os.path.join(tempfile.gettempdir(), "lessons.db")
SCORE_THRESHOLD = 0.7
MAX_RETRIES     = 2

def _secret(name):
    try:
        return st.secrets[name]
    except Exception:
        return os.environ.get(name, "")

GROQ_KEY   = _secret("GROQ_API_KEY")
TAVILY_KEY = _secret("TAVILY_API_KEY")

st.set_page_config(page_title="Agentic RAG (LangGraph)", page_icon="brain", layout="wide")

# ----------------------------------------------------------------------------
# Cached resources
# ----------------------------------------------------------------------------
@st.cache_resource
def get_embedder():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL, model_kwargs={"device": "cpu"})

@st.cache_resource
def get_client():
    return Groq(api_key=GROQ_KEY)

_CACHE = {}   # cache for deterministic helpers (triage, contextualize)

# ----------------------------------------------------------------------------
# LLM + retrieval
# ----------------------------------------------------------------------------
def generate(prompt):
    client = get_client()
    resp = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=GROQ_MODEL, temperature=0.3, max_completion_tokens=512)
    return resp.choices[0].message.content.strip()

def build_retriever(uploaded_files):
    embedder = get_embedder()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    chunks = []
    for uf in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uf.getbuffer())
            tmp_path = tmp.name
        docs = PyPDFLoader(tmp_path).load()
        doc_chunks = splitter.split_documents(docs)
        for ch in doc_chunks:
            ch.metadata["source"] = uf.name
        chunks += doc_chunks
        os.unlink(tmp_path)
    vectorstore = Chroma.from_documents(chunks, embedder)   # in-memory
    return {"vs": vectorstore, "chunks": chunks}

def retrieve(query, retriever, k=4):
    vs, chunks = retriever["vs"], retriever["chunks"]
    semantic = vs.as_retriever(search_kwargs={"k": k})
    keyword  = BM25Retriever.from_documents(chunks); keyword.k = k
    hybrid   = EnsembleRetriever(retrievers=[semantic, keyword], weights=[0.6, 0.4])
    out = []
    for d in hybrid.invoke(query):
        out.append(f"[{d.metadata.get('source','doc')}] {d.page_content}")
    return out

def web_search(query, max_results=4):
    if not TAVILY_KEY:
        return []
    try:
        from tavily import TavilyClient
        resp = TavilyClient(api_key=TAVILY_KEY).search(query, max_results=max_results)
        return [r.get("content", "") for r in resp.get("results", []) if r.get("content")]
    except Exception:
        return []

# ----------------------------------------------------------------------------
# Judge + memory
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
# Agent tools
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
# LangGraph: state, nodes, routing, graph
# ----------------------------------------------------------------------------
class AgentState(TypedDict):
    query: str
    history: list
    kind: str
    k: int
    active_query: str
    context: list
    answer: str
    score: float
    issue: str
    action: str
    retries: int
    force_regen: bool
    web_tried: bool
    use_web_ctx: bool
    web_ctx: list
    trace: list
    pending_lesson: dict
    retriever: dict

def node_contextualize(state):
    q = state["query"]
    if state.get("history"):
        resolved, changed = contextualize(q, state["history"])
        if changed:
            state["trace"].append(("contextualize", f"resolved to: {resolved[:60]}"))
            q = resolved
        else:
            state["trace"].append(("contextualize", "already standalone"))
    state["query"] = q
    state["active_query"] = q
    return state

def node_triage(state):
    kind, k = triage(state["query"])
    state["kind"], state["k"] = kind, k
    state["trace"].append(("triage", f"classified '{kind}' -> k={k}"))
    return state

def node_retrieve(state):
    if state.get("use_web_ctx"):
        state["context"] = state["web_ctx"]
        state["use_web_ctx"] = False
        state["trace"].append(("retrieve", f"using {len(state['context'])} web results"))
    else:
        state["context"] = retrieve(state["active_query"], state["retriever"], k=state["k"])
        state["trace"].append(("retrieve", f"{len(state['context'])} chunks (k={state['k']})"))
    return state

def node_generate(state):
    lessons = get_lessons(state["query"])
    hint = ""
    if lessons:
        hint = "\n\nPast lessons:\n" + "\n".join([f"- {l['issue']} -> {l['fix']}" for l in lessons])
    strict = "\nAnswer ONLY from the context. If absent, say so plainly." if state.get("force_regen") else ""
    prompt = (f"Answer using only the context. Each context line starts with its [source]; "
              f"cite the source(s) you used.{hint}{strict}\n\n"
              f"Context:\n{chr(10).join(state['context'])}\n\nQuestion: {state['query']}")
    state["answer"] = generate(prompt)
    state["trace"].append(("generate", "draft written"))
    return state

def node_judge(state):
    state["score"] = judge_score(state["answer"], state["query"], state["context"])
    state["trace"].append(("judge", f"score = {state['score']}"))
    pending = state.get("pending_lesson")
    if pending:
        if state["score"] > pending["score_before"]:
            save_lesson(pending["query"], pending["issue"], pending["fix"],
                        pending["score_before"], state["score"])
            state["trace"].append(("save_lesson",
                f"kept: {pending['score_before']} -> {state['score']}"))
        else:
            state["trace"].append(("save_lesson",
                f"discarded: {pending['score_before']} -> {state['score']}"))
        state["pending_lesson"] = None
    return state

def node_reflect(state):
    issue, action, fix = diagnose(state["query"], state["answer"], state["score"])
    state["pending_lesson"] = {"query": state["query"], "issue": issue,
                               "fix": fix, "score_before": state["score"]}
    state["issue"], state["action"] = issue, action
    state["trace"].append(("reflect", f"Issue: {issue}"))
    state["trace"].append(("decide_action", f"chose: {action}"))
    state["force_regen"] = False
    if action == "rewrite_query":
        state["active_query"] = rewrite_query(state["query"], issue)
        state["trace"].append(("rewrite_query", f"new query: {state['active_query'][:60]}"))
    elif action == "widen_retrieval":
        state["k"] = min(state["k"] + 4, 12)
        state["trace"].append(("widen_retrieval", f"k -> {state['k']}"))
    elif action == "regenerate":
        state["force_regen"] = True
        state["trace"].append(("regenerate", "stricter grounding"))
    state["retries"] += 1
    return state

def node_web_fallback(state):
    web = web_search(state["query"])
    state["web_tried"] = True
    if web:
        state["web_ctx"] = web
        state["use_web_ctx"] = True
        state["force_regen"] = True
        state["trace"].append(("web_fallback", f"document exhausted -> {len(web)} web results"))
    else:
        state["trace"].append(("web_fallback", "web unavailable / no results"))
    return state

def route_after_judge(state):
    score, retries, kind = state["score"], state["retries"], state["kind"]
    if score >= SCORE_THRESHOLD:
        state["trace"].append(("decide", "accepted (>= threshold)"))
        return "done"
    give_up = (kind == "unanswerable" and retries >= 2) or retries >= MAX_RETRIES
    if give_up:
        if not state.get("web_tried"):
            return "web"
        state["trace"].append(("decide", "stopping: exhausted"))
        return "done"
    return "reflect"

def route_after_web(state):
    return "retrieve" if state.get("use_web_ctx") else "done"

@st.cache_resource
def build_graph():
    b = StateGraph(AgentState)
    b.add_node("contextualize", node_contextualize)
    b.add_node("triage",   node_triage)
    b.add_node("retrieve", node_retrieve)
    b.add_node("generate", node_generate)
    b.add_node("judge",    node_judge)
    b.add_node("reflect",  node_reflect)
    b.add_node("web",      node_web_fallback)
    b.set_entry_point("contextualize")
    b.add_edge("contextualize", "triage")
    b.add_edge("triage",   "retrieve")
    b.add_edge("retrieve", "generate")
    b.add_edge("generate", "judge")
    b.add_conditional_edges("judge", route_after_judge,
                            {"done": END, "reflect": "reflect", "web": "web"})
    b.add_edge("reflect", "retrieve")
    b.add_conditional_edges("web", route_after_web,
                            {"retrieve": "retrieve", "done": END})
    return b.compile()

def run_agent(query, retriever, history=None):
    init_db()
    graph = build_graph()
    init = {"query": query, "history": history or [], "kind": "", "k": 4,
            "active_query": query, "context": [], "answer": "", "score": 0.0,
            "issue": "", "action": "", "retries": 0, "force_regen": False,
            "web_tried": False, "use_web_ctx": False, "web_ctx": [],
            "trace": [], "pending_lesson": None, "retriever": retriever}
    out = graph.invoke(init)
    return out

# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.title("Self-Improving Agentic RAG — LangGraph")
st.caption("StateGraph · retrieve → generate → judge → self-correct → web fallback · learns from the misses")

if not GROQ_KEY:
    st.error("No Groq API key found. Add GROQ_API_KEY in Settings → Secrets.")
    st.stop()

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
        st.success(f"Indexed {len(st.session_state.retriever['chunks'])} chunks from {len(srcs)} PDF(s).")
    st.caption("Web fallback: " + ("on" if TAVILY_KEY else "off (no Tavily key)"))

st.header("2 · Ask")
query = st.text_input("Your question", placeholder="e.g. What is the main contribution?")
go = st.button("Run agent", type="primary", disabled=not (st.session_state.retriever and query))

if go:
    with st.spinner("Running LangGraph agent…"):
        result = run_agent(query, st.session_state.retriever, st.session_state.history)
    st.session_state.history.append({"q": result["query"], "a": result["answer"]})

    color = "green" if result["score"] >= SCORE_THRESHOLD else "orange"
    st.markdown(f"### Answer  &nbsp; :{color}[score {result['score']:.2f}]")
    st.write(result["answer"])

    with st.expander("Decision trace"):
        for node, detail in result["trace"]:
            st.markdown(f"- **{node}** — {detail}")
    with st.expander("Retrieved context"):
        for i, c in enumerate(result["context"], 1):
            st.markdown(f"**{i}.** {c[:500]}…")

if st.session_state.history:
    with st.expander("Conversation history"):
        for turn in st.session_state.history:
            st.markdown(f"**Q:** {turn['q']}")
            st.markdown(f"**A:** {turn['a'][:300]}…")

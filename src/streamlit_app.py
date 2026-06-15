





import streamlit as st
import sqlite3, os, re, tempfile
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from groq import Groq

PREBUILT_CHROMA = "chroma_db_bge"
DB_PATH     = os.environ.get("DB_PATH", "memory.db")
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
GROQ_MODEL  = "llama-3.3-70b-versatile"
GROQ_KEY    = os.environ.get("GROQ_API_KEY", "")

st.set_page_config(page_title="Agentic RAG — Reviewer's Desk", layout="wide", page_icon="o")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;700&display=swap');

:root {
  --paper:#f3f5f8; --ink:#1a2233; --slate:#5b6b82; --grid:#d9e0ea;
  --signal:#2f6df0; --flag:#e0492f; --good:#128a6b;
}
.stApp {
  background:
    linear-gradient(var(--grid) 1px, transparent 1px) 0 0 / 100% 28px,
    var(--paper);
  color: var(--ink);
}
* { font-family:'Inter',sans-serif; }
.block-container { padding-top: 2.4rem; max-width: 1000px; }

/* Masthead */
.masthead { border-bottom: 2px solid var(--ink); padding-bottom: 14px; margin-bottom: 6px; }
.eyebrow { font-family:'JetBrains Mono',monospace; font-size:0.68rem; letter-spacing:3px;
  text-transform:uppercase; color:var(--signal); }
h1 { font-family:'Fraunces',serif !important; font-weight:700 !important; color:var(--ink);
  font-size:2.7rem !important; line-height:1.05; letter-spacing:-0.5px; margin:4px 0 0 !important; }
.dek { font-family:'JetBrains Mono',monospace; color:var(--slate); font-size:0.72rem;
  letter-spacing:1px; margin-top:8px; }

/* Tabs as filing tabs */
.stTabs [data-baseweb="tab-list"] { gap:2px; background:transparent; border-bottom:1px solid var(--grid); }
.stTabs [data-baseweb="tab"] { background:transparent; border-radius:6px 6px 0 0; color:var(--slate);
  font-weight:600; font-size:0.82rem; padding:9px 16px; border:1px solid transparent; border-bottom:none; }
.stTabs [aria-selected="true"] { background:#fff; color:var(--ink);
  border-color:var(--grid); border-bottom:2px solid var(--signal); }

/* Inputs */
.stTextInput input { background:#fff; border:1px solid var(--grid); border-radius:8px;
  color:var(--ink); padding:13px 15px; font-size:0.95rem; }
.stTextInput input:focus { border-color:var(--signal); box-shadow:0 0 0 3px #2f6df022; }
.stButton button { background:var(--ink); color:#fff; border:none; border-radius:8px;
  font-weight:600; padding:9px 26px; font-size:0.88rem; letter-spacing:0.3px; }
.stButton button:hover { background:var(--signal); }

/* Answer as a manuscript page with margin meter */
.answer-wrap { display:flex; gap:0; margin:20px 0; }
.answer-page { flex:1; background:#fff; border:1px solid var(--grid); border-right:none;
  border-radius:10px 0 0 10px; padding:26px 30px; font-family:'Fraunces',serif;
  font-size:1.12rem; line-height:1.75; color:var(--ink); }
.margin { width:64px; background:#fff; border:1px solid var(--grid); border-radius:0 10px 10px 0;
  display:flex; flex-direction:column; align-items:center; justify-content:flex-end;
  padding:10px 0; position:relative; }
.margin-fill { width:100%; border-radius:0 0 9px 0; transition:height .6s ease; }
.margin-num { position:absolute; top:10px; font-family:'JetBrains Mono',monospace;
  font-weight:700; font-size:0.82rem; }
.margin-cap { position:absolute; bottom:8px; font-family:'JetBrains Mono',monospace;
  font-size:0.54rem; letter-spacing:1px; color:var(--slate); writing-mode:vertical-rl; }

/* Chunk cards as index cards */
.chunk { background:#fff; border:1px solid var(--grid); border-left:3px solid var(--signal);
  border-radius:6px; padding:12px 16px; margin:9px 0; font-size:0.86rem; line-height:1.55;
  color:#33405a; }
.tag { font-family:'JetBrains Mono',monospace; font-size:0.62rem; letter-spacing:1px;
  color:var(--slate); text-transform:uppercase; margin-bottom:5px; }

/* Self-critique = red margin note */
.critique { background:#fff7f5; border:1px solid #e0492f33; border-left:3px solid var(--flag);
  border-radius:6px; padding:14px 18px; margin:12px 0; }
.critique .issue { color:var(--flag); font-weight:600; }
.critique .fix { color:var(--good); font-weight:600; }
.critique .meta { font-family:'JetBrains Mono',monospace; font-size:0.64rem; color:var(--slate); margin-top:8px; }

/* Trace as lab log */
.logline { display:flex; gap:14px; padding:7px 0; border-bottom:1px dotted var(--grid);
  font-family:'JetBrains Mono',monospace; font-size:0.8rem; }
.lognode { color:var(--signal); min-width:120px; }

.pill { display:inline-block; font-family:'JetBrains Mono',monospace; font-size:0.66rem;
  letter-spacing:1px; padding:5px 12px; border-radius:4px; background:#fff;
  border:1px solid var(--grid); color:var(--slate); }
.pill.ok { color:var(--good); border-color:#128a6b44; background:#f0faf6; }
small.note { color:var(--slate); font-size:0.78rem; }
.sectlabel { font-family:'JetBrains Mono',monospace; font-size:0.66rem; letter-spacing:2px;
  text-transform:uppercase; color:var(--slate); margin:18px 0 6px; }

/* ============ FILE UPLOADER — force readable light theme ============ */
/* dropzone container */
[data-testid="stFileUploader"],
[data-testid="stFileUploader"] section,
[data-testid="stFileUploaderDropzone"],
section[data-testid="stFileUploaderDropzone"] {
  background:#ffffff !important;
  border:1.5px dashed #c4cdda !important;
  border-radius:10px !important;
  padding:18px 20px !important;
}
/* ALL text inside the uploader -> dark ink */
[data-testid="stFileUploader"],
[data-testid="stFileUploader"] *,
[data-testid="stFileUploaderDropzone"] * {
  color:#1a2233 !important;
}
/* helper / size text -> muted */
[data-testid="stFileUploader"] small,
[data-testid="stFileUploaderDropzoneInstructions"] small,
[data-testid="stFileUploaderDropzoneInstructions"] span {
  color:#5b6b82 !important;
}
/* the Browse files button -> dark bg, WHITE text (override the *->dark above) */
[data-testid="stFileUploader"] button,
[data-testid="stBaseButton-secondary"],
section[data-testid="stFileUploaderDropzone"] button {
  background:#1a2233 !important;
  border:none !important;
  border-radius:8px !important;
  font-weight:600 !important;
  padding:8px 18px !important;
}
[data-testid="stFileUploader"] button,
[data-testid="stFileUploader"] button *,
[data-testid="stFileUploader"] button p,
[data-testid="stFileUploader"] button span,
[data-testid="stFileUploader"] button div,
section[data-testid="stFileUploaderDropzone"] button * {
  color:#ffffff !important;
  fill:#ffffff !important;
}
[data-testid="stFileUploader"] button:hover { background:#2f6df0 !important; }
[data-testid="stFileUploaderDropzone"]:hover { border-color:#2f6df0 !important; }
/* uploaded-file row */
[data-testid="stFileUploaderFile"],
[data-testid="stFileUploaderFile"] * { color:#1a2233 !important; }
[data-testid="stFileUploaderDeleteBtn"] svg { fill:#e0492f !important; }

/* header spacing */
header[data-testid="stHeader"] { background:transparent !important; }
.block-container { padding-top: 1.6rem !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_embedder():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)

@st.cache_resource
def load_client():
    return Groq(api_key=GROQ_KEY)

@st.cache_resource
def load_prebuilt_retriever():
    if not os.path.isdir(PREBUILT_CHROMA):
        return None
    vs = Chroma(persist_directory=PREBUILT_CHROMA, embedding_function=get_embedder())
    return vs.as_retriever(search_kwargs={"k": 4})

def build_retriever_from_pdf(file_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes); tmp_path = tmp.name
    pages  = PyPDFLoader(tmp_path).load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64).split_documents(pages)
    os.remove(tmp_path)
    vs = Chroma.from_documents(chunks, get_embedder())
    return vs.as_retriever(search_kwargs={"k": 4}), len(pages), len(chunks)

def retrieve(query, retriever, k=4):
    retriever.search_kwargs["k"] = k
    return [d.page_content for d in retriever.invoke(query)]

def generate(prompt, client):
    resp = client.chat.completions.create(
        messages=[{"role":"user","content":prompt}],
        model=GROQ_MODEL, temperature=0.3, max_completion_tokens=512)
    return resp.choices[0].message.content.strip()

def _looks_like_nonanswer(answer):
    a = answer.lower()
    cues = ["does not provide", "doesn\'t provide", "no information", "not mentioned",
            "not provided", "cannot answer", "can\'t answer", "context does not",
            "context doesn\'t", "unable to answer", "no relevant information",
            "not available in the context", "not enough information"]
    return any(c in a for c in cues)

def judge_score(answer, question, context, client):
    # A non-answer is a failure to help the user, regardless of faithfulness.
    if _looks_like_nonanswer(answer):
        return 0.2
    prompt = (
        "You are grading a RAG answer. Consider TWO things:\n"
        "1. Is the answer supported by the context (faithfulness)?\n"
        "2. Does the answer actually address the question (relevance)?\n"
        "An answer that says the context lacks the information must score LOW.\n"
        "Respond with exactly one line: SCORE: <number between 0.0 and 1.0>\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{chr(10).join(context)}\n\n"
        f"Answer: {answer}"
    )
    raw = generate(prompt, client)
    m = re.search(r"score\s*[:=]\s*(\d+\.?\d*)", raw, re.IGNORECASE)
    if not m:
        m = re.search(r"(\d\.\d+)", raw)  # any decimal like 0.7
    if not m:
        return 0.5  # genuine parse failure -> neutral, NOT 0.0
    s = float(m.group(1))
    if s > 1.0:  # model gave e.g. 8 meaning 0.8 or 80
        s = s/10.0 if s <= 10 else s/100.0
    return round(min(max(s, 0.0), 1.0), 3)


SCORE_THRESHOLD = 0.7
MAX_RETRIES     = 2

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS lessons (
            id INTEGER PRIMARY KEY AUTOINCREMENT, query TEXT, issue TEXT, fix TEXT,
            score_before REAL, score_after REAL, timestamp TEXT)""")

def save_lesson(query, issue, fix, score_before, score_after=0.0):
    import datetime as _dt
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("INSERT INTO lessons (query,issue,fix,score_before,score_after,timestamp) VALUES (?,?,?,?,?,?)",
            (query, issue, fix, score_before, score_after, _dt.datetime.now().isoformat()))

def recent_lessons(limit=3):
    if not os.path.exists(DB_PATH): return []
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute("SELECT query,issue,fix FROM lessons ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [{"query": r[0], "issue": r[1], "fix": r[2]} for r in rows]

def reflect(query, answer, score, client):
    prompt = (f"The answer scored {score:.2f}/1.0 for faithfulness. "
              f"Diagnose the issue in one sentence and suggest one fix. "
              f"Format exactly:\nIssue: ...\nFix: ...\n\nQuestion: {query}\nAnswer: {answer}")
    diag = generate(prompt, client)
    issue, fix = "answer weakly grounded in context", "retrieve more context and restrict claims"
    for line in diag.splitlines():
        low = line.lower()
        if low.startswith("issue:"): issue = line.split(":",1)[1].strip()
        elif low.startswith("fix:"): fix = line.split(":",1)[1].strip()
    return issue, fix

def run_agent(query, retriever, client):
    """Full loop: memory -> retrieve -> generate -> judge -> (reflect+retry). Writes lessons."""
    init_db()
    lessons = recent_lessons()
    trace = [{"node":"memory_check","detail":f"{len(lessons)} prior lessons"}]
    retries = 0
    k = 6  # start broad to reduce retrieval misses
    while True:
        ctx = retrieve(query, retriever, k=k)
        trace.append({"node":"retrieve","detail":f"{len(ctx)} chunks (k={k})"})
        hint = ""
        if lessons:
            hint = "\n\nPast lessons:\n" + "\n".join([f"- {l['issue']} -> {l['fix']}" for l in lessons])
        prompt = f"Answer using only the context below.{hint}\n\nContext:\n{chr(10).join(ctx)}\n\nQuestion: {query}"
        answer = generate(prompt, client)
        trace.append({"node":"generate","detail":"draft written"})
        score = judge_score(answer, query, ctx, client)
        trace.append({"node":"judge","detail":f"score = {score}"})
        if score >= SCORE_THRESHOLD or retries >= MAX_RETRIES:
            break
        issue, fix = reflect(query, answer, score, client)
        save_lesson(query, issue, fix, score, 0.0)
        trace.append({"node":"reflect","detail":f"Issue: {issue} | Fix: {fix}"})
        retries += 1
        k = 10  # widen retrieval on retry
        lessons = recent_lessons()
    return {"query":query, "answer":answer, "score":score, "context":ctx, "retries":retries, "trace":trace}

def score_hex(s):
    if s>=0.7: return "#128a6b"
    if s>=0.4: return "#c98a1a"
    return "#e0492f"

def get_lessons(limit=50):
    if not os.path.exists(DB_PATH): return []
    with sqlite3.connect(DB_PATH) as conn:
        return conn.execute("SELECT query,issue,fix,score_before,score_after,timestamp FROM lessons ORDER BY id DESC LIMIT ?",(limit,)).fetchall()

st.markdown("""
<div class='masthead'>
  <div class='eyebrow'>Self-Reviewing Retrieval Agent</div>
  <h1>The Reviewer's Desk</h1>
</div>
""", unsafe_allow_html=True)


client = load_client()
active_retriever = st.session_state.get("user_retriever") or load_prebuilt_retriever()

tab0, tab1, tab2, tab3, tab4 = st.tabs(["Document", "Ask", "Trace", "Lessons", "Scores"])

with tab0:
    st.markdown("<div class='sectlabel'>Load a source</div>", unsafe_allow_html=True)
    up = st.file_uploader("Drop a PDF", type="pdf", label_visibility="collapsed")
    if up is not None and st.button("Index document"):
        with st.spinner("Reading, splitting, embedding..."):
            retr, n_pages, n_chunks = build_retriever_from_pdf(up.read())
            st.session_state["user_retriever"] = retr
            st.session_state["pdf_name"] = up.name
            st.session_state.pop("last", None)        # clear old trace
            init_db()                                  # ensure table exists
            with sqlite3.connect(DB_PATH) as _c:       # fresh lessons per new doc
                _c.execute("DELETE FROM lessons")
        st.markdown(f"<span class='pill ok'>INDEXED · {up.name} · {n_pages} pages · {n_chunks} chunks</span>", unsafe_allow_html=True)
        st.markdown("<small class='note'>Switch to the Ask tab to query it.</small>", unsafe_allow_html=True)
    if st.session_state.get("pdf_name"):
        st.markdown(f"<span class='pill ok'>ACTIVE · {st.session_state['pdf_name']}</span>", unsafe_allow_html=True)
    elif load_prebuilt_retriever() is not None:
        st.markdown("<small class='note'>No upload yet — Ask uses the bundled demo document.</small>", unsafe_allow_html=True)
    else:
        st.markdown("<small class='note'>No document loaded. Upload a PDF to begin.</small>", unsafe_allow_html=True)

with tab1:
    if active_retriever is None:
        st.markdown("<small class='note'>Load a document first.</small>", unsafe_allow_html=True)
    else:
        query = st.text_input("Ask", placeholder="Put a question to the document...", label_visibility="collapsed")
        if st.button("Run agent") and query:
            with st.spinner("Retrieving, drafting, grading, retrying if weak..."):
                result = run_agent(query, active_retriever, client)
            answer = result["answer"]; score = result["score"]; ctx = result["context"]
            hexc = score_hex(score); pct = int(score*100)
            st.markdown(f"""
            <div class='answer-wrap'>
              <div class='answer-page'>{answer}</div>
              <div class='margin'>
                <div class='margin-num' style='color:{hexc}'>{score}</div>
                <div class='margin-fill' style='height:{pct}%; background:{hexc}1f; border-top:2px solid {hexc}'></div>
                <div class='margin-cap'>FAITHFULNESS</div>
              </div>
            </div>""", unsafe_allow_html=True)
            st.session_state["last"] = {"query":query,"context":ctx,"answer":answer,"score":score,"retries":result["retries"],"trace":result["trace"]}

with tab2:
    if "last" in st.session_state:
        d = st.session_state["last"]
        st.markdown(f"<div class='sectlabel'>Question</div>{d['query']}", unsafe_allow_html=True)
        st.markdown("<div class='sectlabel'>Pipeline</div>", unsafe_allow_html=True)
        steps = [("memory_check","loaded prior lessons"),("retrieve",f"{len(d['context'])} chunks"),
                 ("generate","draft written"),("judge",f"score = {d['score']}")]
        log = "".join([f"<div class='logline'><span class='lognode'>{n}</span><span>{t}</span></div>" for n,t in steps])
        st.markdown(log, unsafe_allow_html=True)
        st.markdown("<div class='sectlabel'>Retrieved context</div>", unsafe_allow_html=True)
        for i,ch in enumerate(d["context"]):
            st.markdown(f"<div class='chunk'><div class='tag'>Chunk {i+1}</div>{ch[:320]}...</div>", unsafe_allow_html=True)
    else:
        st.markdown("<small class='note'>Run a query to see the pipeline trace.</small>", unsafe_allow_html=True)

with tab3:
    lessons = get_lessons()
    if lessons:
        st.markdown(f"<div class='sectlabel'>{len(lessons)} lessons in memory</div>", unsafe_allow_html=True)
        for row in lessons:
            st.markdown(f"""<div class='critique'>
              <div style='font-weight:600;margin-bottom:6px'>Q: {row[0]}</div>
              <div><span class='issue'>Issue</span> — {row[1]}</div>
              <div><span class='fix'>Fix</span> — {row[2]}</div>
              <div class='meta'>{row[3]:.2f} → {row[4]:.2f}  ·  {row[5][:19]}</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown("<small class='note'>No lessons yet. The agent records one each time it has to retry a weak answer.</small>", unsafe_allow_html=True)

with tab4:
    lessons = get_lessons(50)
    if lessons:
        import plotly.graph_objects as go
        scores = [row[3] for row in lessons][::-1]
        fig = go.Figure(go.Scatter(y=scores, mode="lines+markers",
            line=dict(color="#2f6df0",width=2.5), marker=dict(size=7,color="#1a2233"),
            fill="tozeroy", fillcolor="rgba(47,109,240,0.07)"))
        fig.update_layout(title=dict(text="Faithfulness over time", font=dict(family="Fraunces", color="#1a2233", size=18)),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#5b6b82", family="JetBrains Mono", size=11),
            yaxis=dict(range=[0,1], gridcolor="#d9e0ea"), xaxis=dict(gridcolor="#d9e0ea"), height=420)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown("<small class='note'>No scores recorded yet.</small>", unsafe_allow_html=True)

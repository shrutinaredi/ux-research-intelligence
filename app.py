"""Streamlit UI — UX Research Intelligence demo for Great Question.

Run:  streamlit run app.py
"""

import os

import chromadb
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()
try:
    secret_key = st.secrets.get("GROQ_API_KEY") if hasattr(st, "secrets") else None
except Exception:
    secret_key = None
if secret_key and not os.getenv("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = secret_key

st.set_page_config(
    page_title="UX Research Intelligence",
    layout="centered",
    initial_sidebar_state="expanded",
)

if not os.getenv("GROQ_API_KEY"):
    st.title("UX Research Intelligence")
    st.error(
        "**`GROQ_API_KEY` is not configured.**\n\n"
        "If you're running locally: add it to `.env`.\n\n"
        "If you're on Streamlit Cloud: open **Manage app → Settings → "
        "Secrets** and add:\n\n"
        '`GROQ_API_KEY = "gsk_..."`\n\n'
        "Click **Save** — the app reboots automatically."
    )
    st.stop()

from agents import run_query                                    # noqa: E402
from agents.retriever import set_retrieval_context              # noqa: E402
from agents.telemetry import log_feedback                       # noqa: E402
from ingest import EMBEDDING_MODEL, ingest_file_obj, ingest_sample_sessions  # noqa: E402

SAMPLE_JSON = os.path.join(os.path.dirname(__file__), "data", "sample_sessions.json")
SAMPLE_LABEL = "__sample__"

MAX_FILES = 5

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* ── Page ── */
  .block-container {
    padding-top: 0 !important;
    padding-bottom: 3rem;
    max-width: 780px;
  }

  /* ── Hero band ── */
  .hero-band {
    background: linear-gradient(135deg, #1A0E3B 0%, #3D1F8E 55%, #5B4CF5 100%);
    border-radius: 0 0 24px 24px;
    padding: 2.4rem 2.5rem 2.2rem;
    margin-bottom: 2rem;
  }
  .hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(255,255,255,0.12);
    color: #C4B8F8;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 4px 14px;
    border-radius: 999px;
    margin-bottom: 1rem;
    border: 1px solid rgba(255,255,255,0.18);
  }
  .hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: #FFFFFF;
    margin-bottom: 0.45rem;
    line-height: 1.12;
  }
  .hero-title span { color: #00D4C8; }
  .hero-tag {
    font-size: 0.97rem;
    color: #C4B8F8;
    margin: 0;
    font-weight: 400;
    line-height: 1.6;
  }

  /* ── Corpus stats bar ── */
  .stats-bar {
    display: flex;
    align-items: center;
    gap: 1.5rem;
    background: #F5F3FF;
    border: 1px solid #DDD6FE;
    border-radius: 12px;
    padding: 0.7rem 1.2rem;
    margin-bottom: 1.4rem;
    font-size: 0.85rem;
    color: #3D1F8E;
    font-weight: 500;
  }
  .stats-bar .stat { display: flex; align-items: center; gap: 5px; }
  .stats-bar .stat-num { font-weight: 700; color: #1A0E3B; }
  .stats-dot { color: #C4B8F8; }

  /* ── Pills ── */
  .pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.73rem;
    font-weight: 600;
    margin-right: 5px;
    line-height: 1.6;
    vertical-align: middle;
  }
  .pill-good  { background:#D1FAE5; color:#065F46; }
  .pill-warn  { background:#FEF3C7; color:#92400E; }
  .pill-info  { background:#EDE9FE; color:#5B21B6; }
  .pill-teal  { background:#CCFBF1; color:#0F766E; }

  /* ── Result card header ── */
  .card-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 1rem;
    margin-bottom: 0.6rem;
  }
  .card-question {
    font-size: 1.05rem;
    font-weight: 700;
    color: #1A0E3B;
    flex: 1;
    margin: 0;
  }
  .card-meta {
    font-size: 0.75rem;
    color: #6B7280;
    white-space: nowrap;
    padding-top: 2px;
    font-weight: 500;
  }

  /* ── Answer ── */
  .answer-block {
    font-size: 0.95rem;
    line-height: 1.75;
    color: #1E1B4B;
    margin: 0.8rem 0 0.4rem;
  }

  /* ── Source chip ── */
  .source-chip { font-size: 0.82rem; color: #5B4CF5; font-weight: 600; }

  /* ── Excerpt box ── */
  .excerpt-box {
    font-size: 0.86rem;
    color: #1E1B4B;
    background: #F5F3FF;
    padding: 0.75rem 1rem;
    border-radius: 10px;
    margin-bottom: 0.6rem;
    border-left: 3px solid #5B4CF5;
    line-height: 1.6;
  }

  /* ── Empty state ── */
  .empty-card {
    border: 1.5px dashed #C4B8F8;
    border-radius: 16px;
    padding: 3rem 2rem;
    text-align: center;
    background: #F5F3FF;
    color: #3D1F8E;
  }
  .empty-card .empty-title { font-size: 1.1rem; font-weight: 700; color: #1A0E3B; margin-bottom: 0.4rem; }
  .empty-card .empty-sub   { font-size: 0.9rem; color: #6B7280; }
  .empty-steps {
    display: flex;
    justify-content: center;
    gap: 2rem;
    margin-top: 1.5rem;
  }
  .empty-step {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.3rem;
    font-size: 0.82rem;
    color: #5B4CF5;
    font-weight: 600;
  }
  .step-num {
    width: 28px; height: 28px;
    background: #5B4CF5;
    color: white;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.8rem;
    font-weight: 700;
  }

  /* ── Footer ── */
  .footer-note {
    margin-top: 3.5rem;
    padding-top: 1.2rem;
    border-top: 1px solid #DDD6FE;
    color: #9CA3AF;
    font-size: 0.8rem;
    text-align: center;
    letter-spacing: 0.01em;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] { border-right: 1px solid #DDD6FE; }
  [data-testid="stSidebar"] .sidebar-logo {
    font-size: 1rem;
    font-weight: 800;
    color: #1A0E3B;
    letter-spacing: -0.02em;
    padding-bottom: 0.25rem;
  }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading embedding model…")
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _reset_collection() -> None:
    client = st.session_state.chroma_client
    try:
        client.delete_collection("session_papers")
    except Exception:
        pass
    st.session_state.collection = client.get_or_create_collection(
        name="session_papers", metadata={"hnsw:space": "cosine"},
    )
    st.session_state.ingested = {}


if "chroma_client" not in st.session_state:
    st.session_state.chroma_client = chromadb.EphemeralClient()
    st.session_state.collection = st.session_state.chroma_client.get_or_create_collection(
        name="session_papers", metadata={"hnsw:space": "cosine"},
    )
    st.session_state.ingested  = {}
    st.session_state.history   = []
    st.session_state.prefilled_q = ""

model = get_embedding_model()
set_retrieval_context(model, st.session_state.collection)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        '<div class="sidebar-logo">⬡ &nbsp;Research Intelligence</div>',
        unsafe_allow_html=True,
    )
    st.caption("Upload interview transcripts or research PDFs, then ask anything across them.")
    st.divider()

    # ── Sample data ──
    if SAMPLE_LABEL not in st.session_state.ingested:
        if st.button("⚡ Load sample sessions", use_container_width=True):
            with st.spinner("Loading 10 mock UX interviews…"):
                n = ingest_sample_sessions(SAMPLE_JSON, st.session_state.collection, model)
            st.session_state.ingested[SAMPLE_LABEL] = n
            st.rerun()
        st.divider()

    uploaded = st.file_uploader(
        "Or drop your own files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        label_visibility="visible",
    )

    slots_left = MAX_FILES - len(st.session_state.ingested)
    if st.button(
        "Ingest files",
        disabled=not uploaded,
        use_container_width=True,
        type="primary",
    ):
        new_files = [f for f in uploaded if f.name not in st.session_state.ingested]
        to_ingest = new_files[:slots_left]
        if len(new_files) > slots_left:
            st.warning(f"Limit {MAX_FILES} files — skipping {len(new_files) - slots_left}.")
        for f in to_ingest:
            with st.status(f"Processing {f.name}…", expanded=False) as s:
                n = ingest_file_obj(f, f.name, st.session_state.collection, model)
                if n:
                    s.update(label=f"✓ {f.name}  ({n} chunks)", state="complete")
                else:
                    s.update(label=f"✗ {f.name}  (no text found)", state="error")
            if n:
                st.session_state.ingested[f.name] = n

    if st.session_state.ingested:
        st.divider()
        st.markdown("**Loaded sessions**")
        for name, n in st.session_state.ingested.items():
            if name == SAMPLE_LABEL:
                icon, label = "🗂", "Sample UX Sessions (10)"
            elif name.endswith(".pdf"):
                icon, label = "📄", name
            else:
                icon, label = "📝", name
            st.markdown(
                f"<div style='font-size:0.88rem;padding:4px 0'>"
                f"{icon} <b>{label}</b><br>"
                f"<span class='source-chip' style='font-size:0.78rem'>{n} chunks indexed</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        st.write("")
        if st.button("Clear all sessions", use_container_width=True):
            _reset_collection()
            set_retrieval_context(model, st.session_state.collection)
            st.rerun()

    st.divider()
    st.markdown(
        "<div style='font-size:0.78rem;color:#9CA3AF;line-height:1.7'>"
        "<b>Pipeline</b><br>"
        "① Router &nbsp;→&nbsp; classifies intent<br>"
        "② Retriever &nbsp;→&nbsp; semantic search<br>"
        "③ Synthesizer &nbsp;→&nbsp; cites sources<br>"
        "④ Critic &nbsp;→&nbsp; verifies every claim"
        "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="hero-band">'
    '<div class="hero-badge">&#9679;&nbsp; Multi-agent RAG &nbsp;·&nbsp; Grounded Insights</div>'
    '<div class="hero-title">UX Research <span>Intelligence</span></div>'
    '<div class="hero-tag">'
    'Ask anything across your sessions &nbsp;·&nbsp; cited answers &nbsp;·&nbsp; critic-verified'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Stats bar (when sessions are loaded)
# ---------------------------------------------------------------------------

if st.session_state.ingested:
    total_chunks = sum(st.session_state.ingested.values())
    n_sessions   = len(st.session_state.ingested)
    st.markdown(
        f'<div class="stats-bar">'
        f'<div class="stat"><span class="stat-num">{n_sessions}</span> session{"s" if n_sessions != 1 else ""} loaded</div>'
        f'<span class="stats-dot">·</span>'
        f'<div class="stat"><span class="stat-num">{total_chunks:,}</span> chunks indexed</div>'
        f'<span class="stats-dot">·</span>'
        f'<div class="stat">Ready to query</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Ask form  OR  empty state
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Executive Summary
# ---------------------------------------------------------------------------

SUMMARY_PROMPT = (
    "What are the top 5 recurring user pain points across all sessions? "
    "For each, give a one-sentence description and cite the session ID."
)

if st.session_state.ingested:
    with st.expander("📋 Generate Executive UX Summary", expanded=False):
        st.caption(
            "Synthesizes the top themes and pain points across all loaded sessions "
            "into a shareable brief — grounded in transcript evidence."
        )
        if st.button("Generate summary", type="primary", use_container_width=True):
            with st.spinner("Synthesizing across all sessions…"):
                try:
                    result = run_query(SUMMARY_PROMPT)
                    st.session_state.history.insert(0, result)
                    st.success("Summary added to results below.")
                except Exception as e:
                    st.error(f"Failed: {e}")


# ---------------------------------------------------------------------------
# Ask form  OR  empty state
# ---------------------------------------------------------------------------

EXAMPLES = [
    "What pain points came up most often?",
    "Summarize key themes across all sessions.",
    "What did participants say about onboarding?",
    "Which session had the most friction?",
]


def _click_example(q: str) -> None:
    st.session_state.prefilled_q = q


if not st.session_state.ingested:
    st.markdown(
        '<div class="empty-card">'
        '<div class="empty-title">No sessions loaded yet</div>'
        '<div class="empty-sub">Upload transcripts or PDFs in the sidebar to get started.</div>'
        '<div class="empty-steps">'
        '<div class="empty-step"><div class="step-num">1</div>Upload files</div>'
        '<div class="empty-step"><div class="step-num">2</div>Click Ingest</div>'
        '<div class="empty-step"><div class="step-num">3</div>Ask anything</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )
else:
    with st.form("ask", clear_on_submit=False):
        question = st.text_input(
            "Ask your research",
            value=st.session_state.prefilled_q,
            placeholder="e.g. What did users say about the checkout experience?",
        )
        submitted = st.form_submit_button("Ask →", type="primary", use_container_width=True)

    st.caption("Quick questions:")
    chip_cols = st.columns(len(EXAMPLES))
    for col, ex in zip(chip_cols, EXAMPLES):
        col.button(ex, key=f"ex_{hash(ex)}", on_click=_click_example,
                   args=(ex,), use_container_width=True)

    if submitted and question.strip():
        st.session_state.prefilled_q = ""
        with st.spinner("Thinking…"):
            try:
                result = run_query(question.strip())
                st.session_state.history.insert(0, result)
            except Exception as e:
                st.error(f"Query failed: {e}")


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def render_run(run: dict) -> None:
    critic    = run.get("critic", {}) or {}
    supported = critic.get("supported", True)
    latency   = run.get("latency_ms", "—")
    n_sources = len(run.get("chunks", []))
    retries   = max(0, run.get("attempts", 1) - 1)
    qtype     = run.get("question_type", "?")

    pill_status = (
        '<span class="pill pill-good">✓ grounded</span>'
        if supported
        else '<span class="pill pill-warn">⚠ unverified</span>'
    )
    pill_type = f'<span class="pill pill-info">{qtype}</span>'

    with st.container(border=True):
        # Header row: question + meta stats
        st.markdown(
            f'<div class="card-header">'
            f'<div class="card-question">{run["question"]}</div>'
            f'<div class="card-meta">{latency} ms &nbsp;·&nbsp; {n_sources} sources'
            f'{f" &nbsp;·&nbsp; {retries}↩" if retries else ""}</div>'
            f'</div>'
            f'{pill_status}{pill_type}',
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # Answer
        st.markdown(
            f'<div class="answer-block">{run.get("answer", "_(no answer)_")}</div>',
            unsafe_allow_html=True,
        )

        # Flagged claims (if any)
        if not supported and critic.get("unsupported_claims"):
            with st.expander("⚠ Critic flagged claims"):
                for claim in critic["unsupported_claims"]:
                    st.markdown(f"- {claim}")

        # Sources
        chunks = run.get("chunks", [])
        if chunks:
            with st.expander(f"📎 Sources ({n_sources})"):
                for i, chunk in enumerate(chunks, 1):
                    st.markdown(
                        f"<span class='source-chip'>{i}. {chunk['source']} &nbsp;·&nbsp; "
                        f"segment {chunk['page']} &nbsp;·&nbsp; "
                        f"sim {chunk['score']}</span>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<div class='excerpt-box'>{chunk['text']}</div>",
                        unsafe_allow_html=True,
                    )

        # Feedback
        run_id = run["run_id"]
        _, c_up, c_down = st.columns([8, 1, 1])
        if c_up.button("👍", key=f"up_{run_id}", use_container_width=True, help="Helpful"):
            log_feedback(run_id, "up")
            st.toast("Marked helpful")
        if c_down.button("👎", key=f"dn_{run_id}", use_container_width=True, help="Not helpful"):
            log_feedback(run_id, "down")
            st.toast("Feedback saved")


for run in st.session_state.history:
    render_run(run)


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="footer-note">'
    'Insights without evidence are just opinions.'
    '</div>',
    unsafe_allow_html=True,
)

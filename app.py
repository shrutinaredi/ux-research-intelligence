"""Streamlit UI for the research assistant.

Users upload up to 5 PDFs in the sidebar; each session gets its own
in-memory ChromaDB so uploads never leak across users. Ask questions
and the multi-agent pipeline answers with page-level citations and a
critic verdict.

Run:  streamlit run app.py
"""

import os

import chromadb
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Local dev reads .env; Streamlit Cloud's st.secrets doesn't auto-export to
# os.environ, so bridge it here before any Groq client is instantiated.
load_dotenv()
try:
    secret_key = st.secrets.get("GROQ_API_KEY") if hasattr(st, "secrets") else None
except Exception:
    secret_key = None
if secret_key and not os.getenv("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = secret_key

st.set_page_config(
    page_title="UX Research Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

if not os.getenv("GROQ_API_KEY"):
    st.title("Intelligent Research Assistant")
    st.error(
        "**`GROQ_API_KEY` is not configured.**\n\n"
        "If you're running locally: add it to `.env`.\n\n"
        "If you're on Streamlit Cloud: open **Manage app → Settings → "
        "Secrets** and add a line exactly like this (quotes required):\n\n"
        '`GROQ_API_KEY = "gsk_..."`\n\n'
        "Click **Save** — the app reboots automatically."
    )
    st.stop()

from agents import run_query  # noqa: E402  (imports after env is wired)
from agents.retriever import set_retrieval_context  # noqa: E402
from agents.telemetry import log_feedback  # noqa: E402
from ingest import EMBEDDING_MODEL, ingest_file_obj  # noqa: E402

MAX_FILES = 5

# ---------------------------------------------------------------------------
# Custom CSS — typography, pills, cards
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

      html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

      .block-container { padding-top: 0rem !important; max-width: 900px; }

      /* ── Hero band ── */
      .hero-band {
        background: linear-gradient(135deg, #1A0E3B 0%, #3D1F8E 55%, #5B4CF5 100%);
        border-radius: 0 0 20px 20px;
        padding: 2.2rem 2.5rem 2rem 2.5rem;
        margin-bottom: 2rem;
      }
      .hero-badge {
        display: inline-block;
        background: rgba(255,255,255,0.15);
        color: #C4B8F8;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 3px 12px;
        border-radius: 999px;
        margin-bottom: 0.85rem;
        border: 1px solid rgba(255,255,255,0.2);
      }
      .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: #FFFFFF;
        margin-bottom: 0.4rem;
        line-height: 1.15;
      }
      .hero-title span {
        color: #00D4C8;
      }
      .hero-tag {
        font-size: 1rem;
        color: #C4B8F8;
        margin: 0;
        font-weight: 400;
      }

      /* ── Status pills ── */
      .pill {
        display: inline-block;
        padding: 3px 11px;
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 600;
        margin-right: 6px;
        line-height: 1.6;
      }
      .pill-good  { background: #D1FAE5; color: #065F46; }
      .pill-warn  { background: #FEF3C7; color: #92400E; }
      .pill-info  { background: #EDE9FE; color: #5B21B6; }
      .pill-muted { background: #F1F5F9; color: #475569; }

      /* ── Q&A card ── */
      .qcard h3 {
        font-size: 1.1rem;
        font-weight: 700;
        margin: 0 0 0.6rem 0;
        color: #1A0E3B;
      }

      /* ── Source chip ── */
      .source-chip {
        font-size: 0.83rem;
        color: #5B4CF5;
        font-weight: 500;
      }

      /* ── Empty-state card ── */
      .empty-card {
        border: 1.5px dashed #C4B8F8;
        border-radius: 16px;
        padding: 2.5rem 2rem;
        text-align: center;
        background: #F5F3FF;
        color: #3D1F8E;
        font-weight: 500;
      }
      .empty-card b { color: #1A0E3B; }

      /* ── Source excerpt box ── */
      .excerpt-box {
        font-size: 0.88rem;
        color: #1E1B4B;
        background: #F5F3FF;
        padding: 0.8rem 1rem;
        border-radius: 10px;
        margin-bottom: 0.5rem;
        border-left: 3px solid #5B4CF5;
      }

      /* ── Footer ── */
      .footer-note {
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #DDD6FE;
        color: #6B7280;
        font-size: 0.83rem;
      }

      /* ── Sidebar tweaks ── */
      [data-testid="stSidebar"] {
        border-right: 1px solid #DDD6FE;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached resources (once per server instance)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


# ---------------------------------------------------------------------------
# Per-session state: ephemeral Chroma client + ingested-files registry
# ---------------------------------------------------------------------------

def _reset_collection() -> None:
    client = st.session_state.chroma_client
    try:
        client.delete_collection("session_papers")
    except Exception:
        pass
    st.session_state.collection = client.get_or_create_collection(
        name="session_papers",
        metadata={"hnsw:space": "cosine"},
    )
    st.session_state.ingested = {}


if "chroma_client" not in st.session_state:
    st.session_state.chroma_client = chromadb.EphemeralClient()
    st.session_state.collection = st.session_state.chroma_client.get_or_create_collection(
        name="session_papers",
        metadata={"hnsw:space": "cosine"},
    )
    st.session_state.ingested = {}      # filename -> chunk count
    st.session_state.history = []
    st.session_state.prefilled_q = ""

model = get_embedding_model()
set_retrieval_context(model, st.session_state.collection)


# ---------------------------------------------------------------------------
# Sidebar: upload + manage
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### Your research sessions")
    st.caption(f"Up to {MAX_FILES} files · PDFs or .txt transcripts · stored in memory · not shared")

    uploaded = st.file_uploader(
        "Drop transcripts or PDFs here",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if st.button("Ingest", disabled=not uploaded, use_container_width=True, type="primary"):
        new_files = [f for f in uploaded if f.name not in st.session_state.ingested]
        room = MAX_FILES - len(st.session_state.ingested)
        to_ingest = new_files[:room]
        if len(new_files) > room:
            st.warning(
                f"Cap of {MAX_FILES} files; skipping {len(new_files) - room} file(s)."
            )
        for f in to_ingest:
            with st.status(f"Ingesting {f.name}", expanded=False) as status:
                n_chunks = ingest_file_obj(f, f.name, st.session_state.collection, model)
                if n_chunks:
                    status.update(label=f"{f.name} · {n_chunks} chunks", state="complete")
                else:
                    status.update(label=f"{f.name} · no extractable text", state="error")
            if n_chunks:
                st.session_state.ingested[f.name] = n_chunks

    if st.session_state.ingested:
        st.divider()
        st.markdown("**Ingested sessions**")
        for name, n in st.session_state.ingested.items():
            st.markdown(
                f"<div style='font-size:0.9rem;'><b>{name}</b><br>"
                f"<span class='source-chip'>{n} chunks</span></div>",
                unsafe_allow_html=True,
            )
        st.write("")
        if st.button("Clear all", use_container_width=True):
            _reset_collection()
            set_retrieval_context(model, st.session_state.collection)
            st.rerun()

    st.divider()
    st.caption(
        "**How it works**  \n"
        "Router classifies your question, Retriever finds relevant transcript "
        "segments, Synthesizer answers with session citations, Critic verifies "
        "every claim is grounded (max 2 retries)."
    )


# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="hero-band">'
    '<div class="hero-badge">Powered by LangGraph · Groq · ChromaDB</div>'
    '<div class="hero-title">UX Research <span>Intelligence</span></div>'
    '<div class="hero-tag">Ask questions across your interview transcripts &nbsp;·&nbsp; '
    'session-level citations &nbsp;·&nbsp; self-verifying critic loop</div>'
    '</div>',
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Empty-state OR ask form
# ---------------------------------------------------------------------------

EXAMPLES = [
    "What pain points did users mention most frequently?",
    "Summarize the key themes across all sessions.",
    "What did participants say about the onboarding flow?",
    "Which sessions had the most friction, and what caused it?",
]


def _click_example(q: str) -> None:
    st.session_state.prefilled_q = q


if not st.session_state.ingested:
    st.markdown(
        '<div class="empty-card">'
        '<b>No sessions loaded yet.</b><br>'
        'Drop interview transcripts (.txt) or research PDFs into the sidebar on the left, click '
        '<b>Ingest</b>, then ask away.'
        '</div>',
        unsafe_allow_html=True,
    )
else:
    with st.form("ask", clear_on_submit=False):
        question = st.text_input(
            "Your question",
            value=st.session_state.prefilled_q,
            placeholder="e.g. What did users say about the checkout experience?",
            label_visibility="collapsed",
        )
        col_submit, col_examples = st.columns([1, 4])
        with col_submit:
            submitted = st.form_submit_button("Ask", type="primary", use_container_width=True)
        with col_examples:
            st.caption("Try one of these:")

    cols = st.columns(len(EXAMPLES))
    for col, ex in zip(cols, EXAMPLES):
        col.button(
            ex,
            key=f"ex_{hash(ex)}",
            on_click=_click_example,
            args=(ex,),
            use_container_width=True,
        )

    if submitted and question.strip():
        st.session_state.prefilled_q = ""
        with st.spinner("Routing → retrieving → synthesizing → critiquing…"):
            try:
                result = run_query(question.strip())
                st.session_state.history.insert(0, result)
            except Exception as e:
                st.error(f"Query failed: {e}")


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

def render_run(run: dict) -> None:
    critic = run.get("critic", {}) or {}
    supported = critic.get("supported", True)

    pill_status = (
        '<span class="pill pill-good">grounded</span>'
        if supported
        else '<span class="pill pill-warn">unsupported claims</span>'
    )
    pill_type = f'<span class="pill pill-info">{run.get("question_type", "?")}</span>'

    with st.container(border=True):
        st.markdown(
            f'<div class="qcard"><h3>{run["question"]}</h3>{pill_status}{pill_type}</div>',
            unsafe_allow_html=True,
        )

        # Metric tiles
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Latency", f"{run.get('latency_ms', '—')} ms")
        c2.metric("Sources used", len(run.get("chunks", [])))
        c3.metric("Retries", max(0, run.get("attempts", 1) - 1))
        c4.metric("Top-k", run.get("top_k", "—"))

        st.markdown("---")
        st.markdown(run.get("answer", "_(no answer)_"))

        if not supported and critic.get("unsupported_claims"):
            with st.expander("Critic flagged claims"):
                for c in critic["unsupported_claims"]:
                    st.markdown(f"- {c}")

        with st.expander(f"Sources · {len(run.get('chunks', []))}"):
            for i, chunk in enumerate(run.get("chunks", []), 1):
                st.markdown(
                    f"**{i}. {chunk['source']} — page {chunk['page']}** "
                    f"<span class='source-chip'>· similarity {chunk['score']}</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div class='excerpt-box'>{chunk['text']}</div>",
                    unsafe_allow_html=True,
                )

        # Feedback
        run_id = run["run_id"]
        col_up, col_down, _spacer = st.columns([1, 1, 12])
        if col_up.button("👍 helpful", key=f"up_{run_id}", use_container_width=True):
            log_feedback(run_id, "up")
            st.toast("Logged as helpful")
        if col_down.button("👎 not helpful", key=f"down_{run_id}", use_container_width=True):
            log_feedback(run_id, "down")
            st.toast("Logged as not helpful")


for run in st.session_state.history:
    render_run(run)


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="footer-note">'
    'Built for <b>Great Question</b> · LangGraph · Llama 3.1 8B via Groq · ChromaDB · Streamlit &nbsp;'
    '· Supports .pdf &amp; .txt transcripts · Critic agent verifies every claim before surfacing it.'
    '</div>',
    unsafe_allow_html=True,
)

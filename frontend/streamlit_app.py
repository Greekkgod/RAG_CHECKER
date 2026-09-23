"""
Streamlit UI for the RAG Answer Verifier.

Two modes:
  1. "Ask a question" -- runs the full pipeline (retrieve -> generate -> verify)
     against the sample corpus. Best for the demo video.
  2. "Audit an existing answer" -- paste any query + answer pair (e.g. from
     ChatGPT, your own app, anywhere) and verify it against the corpus.
     Demonstrates the verifier works independently of what generated the answer.

Run with: streamlit run frontend/streamlit_app.py
"""
import streamlit as st
import time
import os
import sys

# Add the parent directory to the path so we can import the backend logic
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.indexing import build_index, retrieve, Chunk
from backend import llm
from backend.verifier import verify_answer, VerificationReport

st.set_page_config(page_title="RAG Answer Verifier", page_icon="✅", layout="wide")

@st.cache_resource(show_spinner="Loading document index...")
def get_index():
    doc_dir = os.environ.get("DOC_DIR", os.path.join(os.path.dirname(__file__), "..", "data", "sample_docs"))
    return build_index(doc_dir)

BADGE_STYLE = {
    "green": ("🟢", "#1a7f37", "High confidence — claims are well-supported by sources."),
    "yellow": ("🟡", "#9a6700", "Mixed confidence — some claims lack strong support."),
    "red": ("🔴", "#cf222e", "Low confidence — likely hallucinated or contradicted claims."),
}

VERDICT_COLOR = {
    "supported": "#1a7f37",
    "contradicted": "#cf222e",
    "unverifiable": "#9a6700",
}

def _report_to_dict(report: VerificationReport, latency: float) -> dict:
    return {
        "query": report.query,
        "answer": report.answer,
        "claims": [vars(c) for c in report.claims],
        "faithfulness_score": report.faithfulness_score,
        "badge": report.badge,
        "retrieved_sources": report.retrieved_sources,
        "latency_seconds": round(latency, 2),
    }

def render_report(report: dict):
    st.toast("Verification complete!", icon="✅")
    badge = report["badge"]
    emoji, color, description = BADGE_STYLE[badge]

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(
            f"<div style='padding:16px;border-radius:12px;background:{color}15;"
            f"border:1px solid {color}40;box-shadow: 0 4px 6px rgba(0,0,0,0.05);'>"
            f"<h3 style='margin-top:0;margin-bottom:8px;display:flex;align-items:center;gap:8px;'>"
            f"<span style='font-size:1.2em'>{emoji}</span> <span style='color:{color}'>{badge.upper()}</span></h3>"
            f"<p style='margin:0;color:#555;font-size:0.95em'>{description}</p></div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.metric("Faithfulness Score", f"{report['faithfulness_score']:.0%}")
        st.progress(report["faithfulness_score"])
    with col3:
        st.metric("Latency", f"{report['latency_seconds']}s")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📝 Answer Analyzed")
    st.info(report["answer"])

    st.markdown("### 🔍 Claim-by-Claim Breakdown")
    if not report["claims"]:
        st.warning("No checkable factual claims were found in this answer.")
    for i, c in enumerate(report["claims"], 1):
        vcolor = VERDICT_COLOR.get(c["verdict"], "#666")
        verdict_emoji = {"supported": "✅", "contradicted": "❌", "unverifiable": "❓"}.get(c["verdict"], "•")
        with st.expander(
            f"{verdict_emoji} Claim {i}: {c['claim']}",
            expanded=(c["verdict"] != "supported"),
        ):
            st.markdown(
                f"**Verdict:** <span style='color:{vcolor};font-weight:700;padding:2px 8px;background:{vcolor}15;border-radius:12px;'>"
                f"{c['verdict'].upper()}</span> &nbsp;&nbsp;•&nbsp;&nbsp; "
                f"**Method:** `{c['method']}` &nbsp;&nbsp;•&nbsp;&nbsp; "
                f"**Confidence:** {c['confidence']:.2f}",
                unsafe_allow_html=True,
            )
            if c["reasoning"]:
                st.markdown(f"**🤖 Judge Reasoning:** {c['reasoning']}")
            
            if c["evidence"]:
                st.markdown("**📄 Matched Evidence:**")
                for e in c["evidence"]:
                    st.markdown(f"<div style='border-left: 3px solid #ccc; padding-left: 10px; color: #666; font-size: 0.9em; margin-bottom: 8px;'>{e}</div>", unsafe_allow_html=True)
            else:
                st.caption("No evidence was matched for this claim.")

    st.markdown("---")
    st.caption(f"**Sources retrieved:** {', '.join(report['retrieved_sources']) or 'None'}")


st.markdown(
    """
    <h1 style='text-align: center; background: -webkit-linear-gradient(45deg, #FF4B2B, #FF416C);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
    RAG Answer Verifier
    </h1>
    <p style='text-align: center; font-size: 1.1em; color: #666; max-width: 700px; margin: 0 auto 2rem auto;'>
    Given a query and an LLM-generated answer, this system checks each factual claim against the retrieved source documents and flags anything unsupported or contradicted — <b>before you act on it.</b>
    </p>
    """, 
    unsafe_allow_html=True
)

tab1, tab2 = st.tabs(["🔍 Ask a Question (Full Pipeline)", "🛡️ Audit an Existing Answer"])

with tab1:
    st.markdown("#### Test the full Retrieval-Augmented Generation pipeline")
    st.caption("Corpus: sample company policy + product FAQ documents.")
    
    st.markdown("**Try a sample question:**")
    colA, colB, colC = st.columns(3)
    if colA.button("Deleted files policy?", key="q1"):
        st.session_state.tab1_q = "How long are deleted files kept in Trash?"
    if colB.button("Paid leave days?", key="q2"):
        st.session_state.tab1_q = "How many days of paid leave per year?"
    if colC.button("Free plan limit?", key="q3"):
        st.session_state.tab1_q = "What's the storage limit on the Free plan?"
        
    query = st.text_input(
        "Your question",
        value=st.session_state.get("tab1_q", ""),
        placeholder="e.g. How many days of paid leave do employees get per year?",
    )
    if st.button("Generate & Verify", type="primary", disabled=not query):
        with st.spinner("Retrieving context, generating answer, verifying claims..."):
            try:
                start_time = time.time()
                chunks = get_index()
                if not chunks:
                    st.error("No documents indexed. Check DOC_DIR.")
                else:
                    retrieved = retrieve(query, chunks, top_k=4)
                    retrieved_chunks = [c for c, _score in retrieved]
                    context_texts = [c.text for c in retrieved_chunks]
                    
                    answer = llm.generate_answer(query, context_texts)
                    report = verify_answer(query, answer, retrieved_chunks)
                    
                    report_dict = _report_to_dict(report, time.time() - start_time)
                    render_report(report_dict)
            except Exception as e:
                st.error(f"Backend error: {str(e)}")

with tab2:
    st.markdown("#### Audit any answer against the corpus")
    
    if st.button("✨ Load 'Hard Hallucination' Example"):
        st.session_state.tab2_q = "Can I use SMS-based 2FA on the Free plan, what is my storage limit, and what happens if I downgrade my account?"
        st.session_state.tab2_a = "SMS-based 2FA is fully supported on all CloudSync plans, including the Free plan. On the Free plan, your total storage limit is 5 GB, and your individual file upload limit is also 5 GB. If you downgrade your account and exceed your new storage limit, your account goes into read-only mode and any data over the limit will be automatically deleted after 30 days."

    query_audit = st.text_input(
        "Original question",
        value=st.session_state.get("tab2_q", ""),
        placeholder="e.g. What's the file upload limit?",
    )
    answer_audit = st.text_area(
        "Answer to audit (paste from anywhere — ChatGPT, your own app, etc.)",
        value=st.session_state.get("tab2_a", ""),
        placeholder="e.g. The maximum file upload size is 5 GB on all plans...",
        height=140,
    )
    if st.button("Verify this answer", type="primary", disabled=not (query_audit and answer_audit)):
        with st.spinner("Verifying claims against corpus..."):
            try:
                start_time = time.time()
                chunks = get_index()
                if not chunks:
                    st.error("No documents indexed. Check DOC_DIR.")
                else:
                    retrieved = retrieve(query_audit, chunks, top_k=4)
                    retrieved_chunks = [c for c, _score in retrieved]
                    
                    report = verify_answer(query_audit, answer_audit, retrieved_chunks)
                    
                    report_dict = _report_to_dict(report, time.time() - start_time)
                    render_report(report_dict)
            except Exception as e:
                st.error(f"Backend error: {str(e)}")

with st.sidebar:
    st.markdown("### How it works")
    st.markdown(
        """
1. **Retrieve** relevant chunks from the corpus (embedding similarity)
2. **Generate** an answer grounded in those chunks (Gemini)
3. **Decompose** the answer into atomic factual claims (Gemini)
4. **Verify** each claim with a two-tier check:
   - High/low similarity → auto-decided (fast)
   - Ambiguous middle → escalated to an LLM judge (accurate)
5. **Aggregate** into a faithfulness score + badge
        """
    )
    st.markdown("### Try these queries")
    st.code("How long are deleted files kept in Trash?")
    st.code("How many days of paid leave per year?")
    st.code("What's the storage limit on the Free plan?")

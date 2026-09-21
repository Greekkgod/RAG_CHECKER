# RAG Answer Verifier

**Problem:** When a RAG system (or any LLM) gives you an answer grounded in retrieved
documents, there's no easy way to tell which parts of that answer are actually
supported by the sources versus quietly hallucinated. You end up either blindly
trusting the answer or manually re-reading every source to check.

**Solution:** A verification layer that sits between an LLM's answer and the user.
It breaks the answer into atomic factual claims, checks each one against the
retrieved evidence using a two-tier pipeline (fast embedding similarity, escalating
to an LLM judge only for ambiguous cases), and surfaces a per-claim verdict plus an
overall faithfulness score and confidence badge.

## Architecture

```
Query
  │
  ▼
[Retrieval] ── embedding similarity over chunked corpus (sentence-transformers)
  │
  ▼
[Generation] ── Gemini generates an answer grounded in retrieved chunks
  │
  ▼
[Claim Decomposition] ── Gemini breaks the answer into atomic claims
  │
  ▼
[Verification] ── for each claim:
  │                 Tier 1: embedding similarity to evidence
  │                   ├─ high similarity  → auto "supported"   (fast, no LLM call)
  │                   ├─ low similarity   → auto "unverifiable" (fast, no LLM call)
  │                   └─ ambiguous        → Tier 2: LLM-judge verdict (slow, accurate)
  ▼
[Aggregation] ── weighted score → faithfulness % → badge (green/yellow/red)
  │
  ▼
UI (Streamlit): answer with claim-by-claim breakdown + evidence + badge
```

## Project structure

```
rag-verifier/
├── backend/
│   ├── app.py          FastAPI app (POST /query, POST /verify, GET /health)
│   ├── indexing.py      Chunking + embedding + retrieval
│   ├── llm.py            Gemini wrapper: generation, claim decomposition, judge
│   └── verifier.py       Core verification engine (the hybrid pipeline)
├── frontend/
│   └── streamlit_app.py  UI: ask a question, or audit any existing answer
├── data/sample_docs/     Sample corpus (company policy + product FAQ)
├── requirements.txt
└── .env.example
```

## Setup

1. Get a Gemini API key: https://aistudio.google.com/apikey
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set your API key:
   ```bash
   export GEMINI_API_KEY=your-key-here
   ```
4. Start the backend:
   ```bash
   uvicorn backend.app:app --reload --port 8000
   ```
5. In a second terminal, start the UI:
   ```bash
   streamlit run frontend/streamlit_app.py
   ```
6. Open the Streamlit URL it prints (usually http://localhost:8501)

## Try it

Sample queries against the included corpus (`data/sample_docs/`):
- "How long are deleted files kept in Trash before permanent deletion?"
- "How many days of paid leave do employees get per year?"
- "What's the storage limit on the Free plan?"

Or switch to **"Audit an existing answer"** mode and paste in an answer from
anywhere (ChatGPT, your own app) to see it get fact-checked against this corpus.

## Design decisions & tradeoffs (for the video)

**Why decompose into claims instead of scoring the whole answer at once?**
A single faithfulness score for a whole paragraph hides *which* sentence is the
problem. Claim-level scoring is more expensive (one extra LLM call) but tells you
exactly what to distrust.

**Why hybrid similarity + LLM-judge instead of one method?**
- Pure similarity is fast and free but can't catch contradiction — a claim can be
  lexically close to evidence that says the opposite (e.g. "30 days" vs "90 days"
  score as similar text but are factually different).
- Pure LLM-judge-on-everything is accurate but doesn't scale — cost and latency
  grow linearly with every claim in every answer.
- The hybrid escalates only the ambiguous middle band to the LLM, so cost scales
  with *uncertainty*, not with total claim volume.

**Why is "unverifiable" scored as neutral (0.5) instead of bad (0.0)?**
No evidence for a claim isn't the same as evidence contradicting it — it might
mean retrieval simply missed the right chunk. Conflating the two would unfairly
punish answers when the failure was in retrieval, not generation. (This is a
judgment call — a stricter, high-stakes domain like medical or legal advice
might reasonably want "unverifiable" treated as harshly as "contradicted".)

**What would change your mind on the thresholds?**
The 0.75 / 0.35 similarity thresholds were picked by inspection, not tuned on a
labeled dataset. With more time, you'd want a small labeled set of (claim,
evidence, ground-truth verdict) triples to calibrate these properly instead of
eyeballing them.

## Known failure modes

- **Retrieval failure masquerading as hallucination:** if the right document
  exists but wasn't retrieved, the verifier will call a true claim "unverifiable"
  — it can only check against what was actually retrieved, not the whole corpus.
- **LLM-judge inconsistency:** the same claim/evidence pair, judged twice, can
  occasionally get different verdicts. No majority-vote/retry logic is included
  here — a real production version would add that.
- **Character-based chunking** can split a sentence mid-way, occasionally
  weakening evidence matches at chunk boundaries.
- **Compound claims** the decomposition step misses (rare, but possible) will
  be scored as a single claim, hiding a partially-false statement.

## What's simplified for this demo (and what production would add)

- In-memory embedding index instead of a persistent vector DB (Qdrant/pgvector)
- No caching of embeddings/LLM calls between runs
- No retry/backoff on Gemini API calls
- Thresholds are hand-picked, not calibrated on labeled data
- Single-corpus, single-user — no multi-tenancy or auth

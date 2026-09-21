"""
FastAPI backend exposing the RAG-answer-verifier pipeline.

Endpoints:
  POST /query   -> runs full pipeline: retrieve -> generate -> verify -> report
  POST /verify  -> verify an already-existing (query, answer) pair against the
                   corpus, without generating a new answer (useful for auditing
                   answers from an external/other LLM system)
  GET  /health  -> readiness check

Run with: uvicorn backend.app:app --reload --port 8000
"""
import os
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.indexing import build_index, retrieve, Chunk
from backend import llm
from backend.verifier import verify_answer, VerificationReport

app = FastAPI(title="RAG Answer Verifier")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_index_cache: dict[str, list[Chunk]] = {}
DOC_DIR = os.environ.get("DOC_DIR", os.path.join(os.path.dirname(__file__), "..", "data", "sample_docs"))


def get_index() -> list[Chunk]:
    if "chunks" not in _index_cache:
        _index_cache["chunks"] = build_index(DOC_DIR)
    return _index_cache["chunks"]


class QueryRequest(BaseModel):
    query: str
    top_k: int = 4


class VerifyRequest(BaseModel):
    query: str
    answer: str
    top_k: int = 4


class ClaimResultOut(BaseModel):
    claim: str
    verdict: str
    confidence: float
    method: str
    evidence: list[str]
    reasoning: str


class ReportOut(BaseModel):
    query: str
    answer: str
    claims: list[ClaimResultOut]
    faithfulness_score: float
    badge: str
    retrieved_sources: list[str]
    latency_seconds: float


def _report_to_out(report: VerificationReport, latency: float) -> ReportOut:
    return ReportOut(
        query=report.query,
        answer=report.answer,
        claims=[ClaimResultOut(**vars(c)) for c in report.claims],
        faithfulness_score=report.faithfulness_score,
        badge=report.badge,
        retrieved_sources=report.retrieved_sources,
        latency_seconds=round(latency, 2),
    )


@app.get("/health")
def health():
    chunks = get_index()
    return {"status": "ok", "indexed_chunks": len(chunks)}


@app.post("/query", response_model=ReportOut)
def query_and_verify(req: QueryRequest):
    """Full pipeline: retrieve context, generate an answer, then verify it."""
    start = time.time()
    chunks = get_index()
    if not chunks:
        raise HTTPException(500, "No documents indexed. Check DOC_DIR.")

    retrieved = retrieve(req.query, chunks, top_k=req.top_k)
    retrieved_chunks = [c for c, _score in retrieved]
    context_texts = [c.text for c in retrieved_chunks]

    try:
        answer = llm.generate_answer(req.query, context_texts)
        report = verify_answer(req.query, answer, retrieved_chunks)
    except RuntimeError as e:
        raise HTTPException(500, str(e))

    return _report_to_out(report, time.time() - start)


@app.post("/verify", response_model=ReportOut)
def verify_existing(req: VerifyRequest):
    """Audit an existing (query, answer) pair -- e.g. from an external system --
    against this corpus, without generating a new answer."""
    start = time.time()
    chunks = get_index()
    if not chunks:
        raise HTTPException(500, "No documents indexed. Check DOC_DIR.")

    retrieved = retrieve(req.query, chunks, top_k=req.top_k)
    retrieved_chunks = [c for c, _score in retrieved]

    try:
        report = verify_answer(req.query, req.answer, retrieved_chunks)
    except RuntimeError as e:
        raise HTTPException(500, str(e))

    return _report_to_out(report, time.time() - start)

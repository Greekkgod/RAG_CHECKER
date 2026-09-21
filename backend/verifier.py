"""
The verification engine. This is the core of the project:

    answer -> decompose into claims -> match each claim to evidence -> verdict -> aggregate score

Design decision (the key tradeoff to discuss in the video):
  Two-tier hybrid verification instead of a single approach.
    Tier 1 (cheap): embedding similarity between claim and every retrieved chunk.
        - If best similarity is very high  -> auto-mark "supported" (skip LLM call)
        - If best similarity is very low   -> auto-mark "unverifiable" (no relevant
          evidence exists at all; an LLM call won't invent evidence that isn't there)
        - Otherwise (the ambiguous middle) -> escalate to Tier 2
    Tier 2 (accurate, slow): LLM-as-judge call, only on the ambiguous claims.

  This means total LLM-judge calls scale with *ambiguous* claims, not total claims,
  which is the latency/cost lever asked about in "Tradeoffs". Pure similarity is
  fast but can't detect contradiction (a claim can be lexically close to evidence
  that actually says the opposite -- e.g. "up to 30 days" vs "up to 90 days" score
  as similar text but are factually different). Pure LLM-judge-on-everything is
  accurate but doesn't scale and costs money per claim. The hybrid is a middle
  ground and is the single most defensible design choice in this project.
"""
from dataclasses import dataclass, field
from backend.indexing import Chunk, retrieve, embed_texts
from backend import llm
import numpy as np

# Thresholds are deliberately conservative (biased toward escalating to the LLM
# judge rather than trusting similarity blindly) because false "supported" verdicts
# are worse than a slower pipeline -- see "what would change your mind" discussion.
HIGH_CONFIDENCE_THRESHOLD = 0.75
LOW_CONFIDENCE_THRESHOLD = 0.35
EVIDENCE_TOP_K = 3


@dataclass
class ClaimResult:
    claim: str
    verdict: str            # "supported" | "contradicted" | "unverifiable"
    confidence: float       # similarity score that drove tier-1 decision
    method: str             # "similarity" | "llm_judge"
    evidence: list[str] = field(default_factory=list)
    reasoning: str = ""


@dataclass
class VerificationReport:
    query: str
    answer: str
    claims: list[ClaimResult]
    faithfulness_score: float   # 0.0 - 1.0
    badge: str                  # "green" | "yellow" | "red"
    retrieved_sources: list[str] = field(default_factory=list)


def verify_answer(query: str, answer: str, source_chunks: list[Chunk]) -> VerificationReport:
    claims_text = llm.decompose_claims(answer)
    if not claims_text:
        # No factual claims found (e.g. answer was purely conversational) --
        # treat as trivially fully faithful rather than crashing the pipeline.
        return VerificationReport(
            query=query, answer=answer, claims=[],
            faithfulness_score=1.0, badge="green",
            retrieved_sources=list({c.source for c in source_chunks}),
        )

    results: list[ClaimResult] = []
    for claim in claims_text:
        results.append(_verify_single_claim(claim, source_chunks))

    score = _aggregate_score(results)
    badge = _score_to_badge(score)

    return VerificationReport(
        query=query,
        answer=answer,
        claims=results,
        faithfulness_score=score,
        badge=badge,
        retrieved_sources=list({c.source for c in source_chunks}),
    )


def _verify_single_claim(claim: str, source_chunks: list[Chunk]) -> ClaimResult:
    if not source_chunks:
        return ClaimResult(claim=claim, verdict="unverifiable", confidence=0.0,
                            method="similarity", evidence=[],
                            reasoning="No source chunks were retrieved for this query.")

    # Tier 1: embedding similarity between the claim and every retrieved chunk.
    claim_emb = embed_texts([claim])[0]
    sims = np.array([np.dot(claim_emb, c.embedding) for c in source_chunks])
    order = np.argsort(-sims)[:EVIDENCE_TOP_K]
    top_chunks = [source_chunks[i] for i in order]
    top_sims = [float(sims[i]) for i in order]
    best_sim = top_sims[0]

    if best_sim >= HIGH_CONFIDENCE_THRESHOLD:
        return ClaimResult(
            claim=claim, verdict="supported", confidence=best_sim, method="similarity",
            evidence=[c.text for c in top_chunks],
            reasoning=f"High similarity ({best_sim:.2f}) to retrieved evidence.",
        )

    if best_sim < LOW_CONFIDENCE_THRESHOLD:
        return ClaimResult(
            claim=claim, verdict="unverifiable", confidence=best_sim, method="similarity",
            evidence=[],
            reasoning=f"No sufficiently relevant evidence found (best similarity {best_sim:.2f}).",
        )

    # Tier 2: ambiguous zone -- escalate to LLM judge with the top matched evidence.
    evidence_texts = [c.text for c in top_chunks]
    judged = llm.judge_claim(claim, evidence_texts)
    return ClaimResult(
        claim=claim,
        verdict=judged.get("verdict", "unverifiable"),
        confidence=best_sim,
        method="llm_judge",
        evidence=evidence_texts,
        reasoning=judged.get("reasoning", ""),
    )


def _aggregate_score(results: list[ClaimResult]) -> float:
    """
    Weighted score: supported=1.0, unverifiable=0.5 (neutral -- absence of evidence
    is not the same as a false claim), contradicted=0.0 (worst case, actively wrong).
    This weighting is itself a judgment call worth defending: an alternative is to
    treat "unverifiable" as equally bad as "contradicted" for high-stakes domains
    (e.g. medical/legal), which is exactly the precision/recall tradeoff to mention.
    """
    if not results:
        return 1.0
    weights = {"supported": 1.0, "unverifiable": 0.5, "contradicted": 0.0}
    total = sum(weights.get(r.verdict, 0.5) for r in results)
    return round(total / len(results), 3)


def _score_to_badge(score: float) -> str:
    if score >= 0.85:
        return "green"
    if score >= 0.6:
        return "yellow"
    return "red"

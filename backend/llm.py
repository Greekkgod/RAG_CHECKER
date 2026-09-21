"""
Thin wrapper around the Gemini API for the three LLM-powered steps in the pipeline:
1. Answer generation (the RAG system being audited)
2. Claim decomposition (breaking an answer into atomic checkable claims)
3. LLM-as-judge verification (the accurate-but-slow verification path)

All prompts request strict JSON output to keep parsing reliable, matching the
structured-extraction pattern used elsewhere in this project.
"""
import os
import json
from google import genai
from google.genai import types
from google.genai.errors import APIError
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

_client = None


def get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY environment variable not set. "
                "Get a key at https://aistudio.google.com/apikey and export it."
            )
        _client = genai.Client(api_key=api_key)
    return _client


GENERATION_MODEL = "gemini-3.6-flash"
JUDGE_MODEL = "gemini-3.6-flash"


@retry(
    retry=retry_if_exception_type(APIError),
    wait=wait_exponential(multiplier=2, min=5, max=60),
    stop=stop_after_attempt(8)
)
def generate_answer(query: str, context_chunks: list[str]) -> str:
    """Step 1: Generate an answer grounded in retrieved context (the system being audited)."""
    context_block = "\n\n".join(f"[Source {i+1}]\n{c}" for i, c in enumerate(context_chunks))
    prompt = f"""Answer the user's question using ONLY the context below. Be concise (2-5 sentences).
If the context does not fully answer the question, answer with what it does support.

Context:
{context_block}

Question: {query}

Answer:"""
    client = get_client()
    response = client.models.generate_content(model=GENERATION_MODEL, contents=prompt)
    return response.text.strip()


@retry(
    retry=retry_if_exception_type(APIError),
    wait=wait_exponential(multiplier=2, min=5, max=60),
    stop=stop_after_attempt(8)
)
def decompose_claims(answer: str) -> list[str]:
    """
    Step 2: Break the answer into atomic, independently-checkable claims.
    This is the step most naive verifiers skip -- checking a whole paragraph at once
    hides which specific sentence is the problem. Sentence-level would be cheaper
    but misses compound claims (e.g. "X costs 5 USD and ships in 3 days" is two
    claims in one sentence); an LLM call catches that at extra latency/cost.
    """
    prompt = f"""Break the following answer into a list of atomic factual claims.
Each claim should be a single, independently verifiable statement.
Split compound sentences (e.g. "X costs 5 and ships in 3 days" becomes two claims).
Ignore filler/hedging language, greetings, or non-factual statements.

Respond with ONLY a JSON array of strings, nothing else. Example: ["claim one", "claim two"]

Answer to decompose:
{answer}

JSON array:"""
    client = get_client()
    response = client.models.generate_content(model=GENERATION_MODEL, contents=prompt)
    return _parse_json_array(response.text)


@retry(
    retry=retry_if_exception_type(APIError),
    wait=wait_exponential(multiplier=2, min=5, max=60),
    stop=stop_after_attempt(8)
)
def judge_claim(claim: str, evidence_texts: list[str]) -> dict:
    """
    Step 3 (accurate path): LLM-as-judge verdict on a single claim against its
    matched evidence. Used only for claims the cheap embedding-similarity pass
    flags as ambiguous -- this hybrid design is the key latency/accuracy tradeoff
    in the system.
    """
    evidence_block = "\n\n".join(f"[Evidence {i+1}]\n{e}" for i, e in enumerate(evidence_texts))
    prompt = f"""You are a strict fact-checker. Given a CLAIM and EVIDENCE passages, decide if the
evidence supports the claim.

Respond with ONLY a JSON object with this exact shape, nothing else:
{{"verdict": "supported" | "contradicted" | "unverifiable", "reasoning": "one short sentence"}}

Rules:
- "supported": the evidence directly confirms the claim.
- "contradicted": the evidence directly conflicts with the claim.
- "unverifiable": the evidence is insufficient, unrelated, or only partially relevant.

CLAIM: {claim}

EVIDENCE:
{evidence_block if evidence_block else "(no evidence retrieved)"}

JSON:"""
    client = get_client()
    response = client.models.generate_content(model=JUDGE_MODEL, contents=prompt)
    return _parse_json_object(response.text)


def _parse_json_array(text: str) -> list[str]:
    text = _strip_code_fences(text)
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return [str(x) for x in result]
    except json.JSONDecodeError:
        pass
    # fallback: treat each non-empty line as a claim so the pipeline degrades gracefully
    return [line.strip("-• ").strip() for line in text.splitlines() if line.strip()]


def _parse_json_object(text: str) -> dict:
    text = _strip_code_fences(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"verdict": "unverifiable", "reasoning": "Judge response could not be parsed."}


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return text.strip()

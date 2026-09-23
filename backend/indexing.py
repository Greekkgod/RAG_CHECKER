"""
Handles chunking source documents and building an in-memory embedding index.
Kept deliberately simple (no vector DB) since the focus of this project is the
verification layer, not retrieval infrastructure. This is a documented tradeoff:
in a real production system you'd swap this for a proper vector store (e.g.
pgvector, Qdrant, Pinecone) once corpus size or persistence needs grow.
"""
import os
import glob
import numpy as np
from dataclasses import dataclass
from google import genai
from google.genai.errors import APIError
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

_client = None

def get_client():
    global _client
    if _client is None:
        _client = genai.Client()
    return _client


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    embedding: np.ndarray = None


def chunk_text(text: str, source: str, chunk_size: int = 400, overlap: int = 80) -> list[Chunk]:
    """
    Simple sliding-window chunking by characters. Character-based (not token-based)
    chunking is a conscious simplification -- fast and dependency-light, but a
    known tradeoff: it can split mid-sentence. Worth mentioning in the video as
    something you'd swap for sentence-aware chunking in production.
    """
    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        piece = text[start:end].strip()
        if piece:
            chunks.append(Chunk(id=f"{source}::chunk{idx}", text=piece, source=source))
            idx += 1
        if end == len(text):
            break
        start = end - overlap
    return chunks


def build_index(doc_dir: str) -> list[Chunk]:
    """Load all .txt files in doc_dir, chunk them, and embed every chunk."""
    all_chunks: list[Chunk] = []

    for path in sorted(glob.glob(os.path.join(doc_dir, "*.txt"))):
        source_name = os.path.basename(path)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        all_chunks.extend(chunk_text(text, source_name))

    if not all_chunks:
        return []

    texts = [c.text for c in all_chunks]
    arr = embed_texts(texts)
    for chunk, emb in zip(all_chunks, arr):
        chunk.embedding = emb

    return all_chunks

@retry(
    retry=retry_if_exception_type(APIError),
    wait=wait_exponential(multiplier=2, min=5, max=60),
    stop=stop_after_attempt(8)
)
def embed_texts(texts: list[str]) -> np.ndarray:
    client = get_client()
    result = client.models.embed_content(
        model='text-embedding-004',
        contents=texts
    )
    embeddings = [emb.values for emb in result.embeddings]
    arr = np.array(embeddings)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / norms


def retrieve(query: str, chunks: list[Chunk], top_k: int = 4) -> list[tuple[Chunk, float]]:
    """Cosine similarity retrieval (embeddings are pre-normalized, so dot product = cosine sim)."""
    if not chunks:
        return []
    query_emb = embed_texts([query])[0]
    sims = np.array([np.dot(query_emb, c.embedding) for c in chunks])
    top_idx = np.argsort(-sims)[:top_k]
    return [(chunks[i], float(sims[i])) for i in top_idx]

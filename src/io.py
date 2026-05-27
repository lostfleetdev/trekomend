"""
Ollama query embedding for trekomend.

Sends text descriptions to a local Ollama instance running Qwen3-Embedding.
The FAISS index stores raw movie text embeddings. Queries use an asymmetric
prompt: an instruction prefix for the query side, no instruction on the
document (stored) side.
"""
import json as _json
import urllib.request as _urllib

import numpy as np

from .config import OUT_DIM, QUERY_INSTRUCTION


def embed_query(
    query_text: str,
    model: str = "qwen3-embedding:0.6b",
    out_dim: int | None = None,
    instruction: str | None = None,
) -> np.ndarray:
    """Send text to Ollama and return a unit-normalized embedding vector."""
    instr = instruction if instruction is not None else QUERY_INSTRUCTION
    prompt = f"Instruct: {instr}\nQuery: {query_text}"

    payload = _json.dumps({"model": model, "prompt": prompt}).encode("utf-8")
    req = _urllib.Request(
        "http://localhost:11434/api/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with _urllib.urlopen(req) as resp:
        data = _json.loads(resp.read().decode("utf-8"))

    vec = np.array(data["embedding"], dtype=np.float32)
    dim = out_dim if out_dim is not None else OUT_DIM
    vec = vec[:dim]
    vec = vec / np.linalg.norm(vec)
    return vec


def embed_mood(
    mood_text: str,
    model: str = "qwen3-embedding:0.6b",
    out_dim: int | None = None,
) -> np.ndarray:
    """Embed with mood-biased instruction (atmosphere, emotional fit)."""
    from .config import MOOD_INSTRUCTION
    return embed_query(mood_text, model=model, out_dim=out_dim,
                       instruction=MOOD_INSTRUCTION)


def embed_genre(
    genre_text: str,
    model: str = "qwen3-embedding:0.6b",
    out_dim: int | None = None,
) -> np.ndarray:
    """Embed with narrative-biased instruction (genre, plot, themes)."""
    from .config import GENRE_INSTRUCTION
    return embed_query(genre_text, model=model, out_dim=out_dim,
                       instruction=GENRE_INSTRUCTION)


def embed_hybrid(
    hybrid_text: str,
    model: str = "qwen3-embedding:0.6b",
    out_dim: int | None = None,
) -> np.ndarray:
    """Embed with hybrid instruction (combines taste + mood signals)."""
    from .config import HYBRID_INSTRUCTION
    return embed_query(hybrid_text, model=model, out_dim=out_dim,
                       instruction=HYBRID_INSTRUCTION)

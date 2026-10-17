import os
import hashlib
import math
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_DIM = 768


@lru_cache(maxsize=256)
def get_embedding(text: str) -> list[float]:
    """
    Returns a 768-dim embedding for the given text (cached for repeated queries).
    Uses Gemini text-embedding-004 if GEMINI_API_KEY is set,
    otherwise falls back to deterministic semantic cluster vectors.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if api_key and api_key != "your-gemini-api-key-here":
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            response = client.models.embed_content(
                model="text-embedding-004",
                contents=text,
            )
            return list(response.embeddings[0].values)
        except Exception as e:
            print(f"[INFO] Using deterministic fallback embedding ({e})")

    return _deterministic_cluster_embed(text)


def _deterministic_cluster_embed(text: str) -> list[float]:
    text_lower = text.lower()

    if any(kw in text_lower for kw in ["connection", "timeout", "pool", "refused"]):
        cluster_id = 0
    elif any(kw in text_lower for kw in ["auth", "token", "401", "403", "jwt", "signature"]):
        cluster_id = 1
    elif any(kw in text_lower for kw in ["memory", "oom", "heap", "killed", "redis"]):
        cluster_id = 2
    elif any(kw in text_lower for kw in ["schema", "migration", "column", "constraint"]):
        cluster_id = 3
    else:
        cluster_id = 4

    vec = [0.0] * EMBEDDING_DIM
    subspace_size = EMBEDDING_DIM // 5
    start = cluster_id * subspace_size
    end = start + subspace_size

    for i in range(start, end):
        vec[i] = 1.0 / math.sqrt(subspace_size)

    text_hash = hashlib.sha256(text.encode()).hexdigest()
    for i in range(EMBEDDING_DIM):
        byte_val = int(text_hash[(i * 2) % 64 : (i * 2) % 64 + 2], 16)
        vec[i] += (byte_val / 255.0 - 0.5) * 0.02

    norm = math.sqrt(sum(x * x for x in vec))
    return [round(x / norm, 6) for x in vec] if norm > 0 else vec


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))
    return dot / (n1 * n2) if (n1 and n2) else 0.0
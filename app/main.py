from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import math

app = FastAPI(
    title="Embedding API",
    description="API for generating embeddings from text using BAAI/bge-small-en",
    version="1.0.0"
)

# Load model once on startup
try:
    model = SentenceTransformer("BAAI/bge-small-en")
    MAX_TOKENS = 512  # Approximate token limit for this model
except Exception as e:
    raise RuntimeError(f"Failed to load embedding model: {e}")

class EmbedRequest(BaseModel):
    texts: list[str]

@app.post("/embed")
async def embed(request: EmbedRequest):
    texts = request.texts

    if not texts or not isinstance(texts, list):
        raise HTTPException(status_code=400, detail="`texts` must be a non-empty list of strings")

    # Token length check
    for t in texts:
        if len(t.split()) > MAX_TOKENS:
            raise HTTPException(
                status_code=400,
                detail=f"One of the inputs exceeds {MAX_TOKENS} tokens limit. Please chunk long texts."
            )

    try:
        embeddings = model.encode(texts, normalize_embeddings=True)
        return {
            "count": len(embeddings),
            "embeddings": embeddings.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {e}")

@app.post("/batch")
async def embed_batch(request: EmbedRequest):
    texts = request.texts

    if not texts or not isinstance(texts, list):
        raise HTTPException(status_code=400, detail="`texts` must be a non-empty list of strings")

    # Token length check
    for t in texts:
        if len(t.split()) > MAX_TOKENS:
            raise HTTPException(
                status_code=400,
                detail=f"One of the inputs exceeds {MAX_TOKENS} tokens limit. Please chunk long texts."
            )

    try:
        batch_size = 32
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = model.encode(batch, normalize_embeddings=True)
            all_embeddings.extend(zip(batch, embeddings.tolist()))

        results = [
            {"text": text, "embedding": emb}
            for text, emb in all_embeddings
        ]

        return {"results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch embedding generation failed: {e}")

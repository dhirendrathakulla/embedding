from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import numpy as np

app = FastAPI()

# Load model once on startup
model = SentenceTransformer("BAAI/bge-small-en")

class EmbedRequest(BaseModel):
    texts: list[str]

@app.post("/embed")
async def embed(request: EmbedRequest):
    # model.encode returns a numpy array
    print("FAST API Received payload:====", request.dict())

    embeddings = model.encode(request.texts, normalize_embeddings=True)
    return {"embeddings": embeddings.tolist()}

import os
import re
from typing import Any, TypedDict

import numpy as np
from lib.search_utils import CACHE_DIR, Movie, load_movies
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "movie_embeddings.npy")


class SearchResult(TypedDict):
    score: float
    title: str
    description: str


class SemanticSearch:
    def __init__(self):
        self.model: SentenceTransformer = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings: NDArray[np.float32] | None = None
        self.documents: list[Movie] = []
        self.document_map: dict[int, Movie] = {}

    def generate_embedding(self, text: str) -> NDArray[np.float32]:
        text = text.strip()
        if text == "":
            raise ValueError("Text must not be empty")
        return self.model.encode([text])[0]

    def build_embeddings(self, documents: list[Movie]) -> NDArray[np.float32]:
        self.documents = documents
        document_str: list[str] = []
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
            document_str.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(document_str, show_progress_bar=True)
        with open(EMBEDDINGS_PATH, "wb") as f:
            np.save(f, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents: list[Movie]) -> NDArray[np.float32]:
        self.documents = documents
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
        if os.path.exists(EMBEDDINGS_PATH):
            with open(EMBEDDINGS_PATH, "rb") as f:
                self.embeddings = np.load(f)
        if self.embeddings is None or (len(self.documents) != len(self.embeddings)):
            self.embeddings = self.build_embeddings(documents)
        return self.embeddings

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        if self.embeddings is None:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )
        embedded_query = self.generate_embedding(query)
        scores = np.dot(self.embeddings, embedded_query) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(embedded_query)
        )

        top_indices = np.argsort(scores)[::-1][:limit]

        results: list[SearchResult] = []
        for idx in top_indices:
            results.append(
                {
                    "score": float(scores[idx]),
                    "title": self.documents[idx]["title"],
                    "description": self.documents[idx]["description"],
                }
            )
        return results

    @staticmethod
    def cosine_similarity(
        vec1: NDArray[np.float32], vec2: NDArray[np.float32]
    ) -> float:
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)


def verify_model():
    sem_search = SemanticSearch()
    print(f"Model loaded: {sem_search.model}")
    print(f"Max sequence length: {sem_search.model.max_seq_length}")


def embed_text(text: str) -> Any:
    sem_search = SemanticSearch()
    return sem_search.generate_embedding(text)


def verify_embeddings():
    sem_search = SemanticSearch()
    documents: list[Movie] = load_movies()
    embeddings = sem_search.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def embed_query_text(query: str):
    sem_search = SemanticSearch()
    embedded = sem_search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedded[:5]}")
    print(f"Shape: {embedded.shape}")


def search(query: str, limit: int = 5) -> None:
    sem_search = SemanticSearch()
    documents: list[Movie] = load_movies()
    _ = sem_search.load_or_create_embeddings(documents)
    results = sem_search.search(query, limit)
    for i, result in enumerate(results, 1):
        print(
            f"{i}. {result['title']} (score: {result['score']:.4f})\n   {result['description']}\n"
        )


def chunk(text: str, chunk_size: int = 200, overlap: int = 0) -> list[str]:
    words = text.split()
    step = chunk_size - overlap
    return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), step)]


def semantic_chunk(text: str, max_chunk_size: int = 4, overlap: int = 1) -> list[str]:
    SENTENCE_BOUNDARY = r"(?<=[.!?])\s+"
    sentences = re.split(SENTENCE_BOUNDARY, text)
    step = max_chunk_size - overlap
    return [
        " ".join(sentences[i : i + max_chunk_size])
        for i in range(0, len(sentences), step)
    ]

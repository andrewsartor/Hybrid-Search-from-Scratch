import json
import os
import re
from typing import TypedDict, cast

import numpy as np
from lib.search_utils import CACHE_DIR, Movie, load_movies
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "movie_embeddings.npy")
CHUNK_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "chunk_embeddings.npy")
CHUNK_METADATA_PATH = os.path.join(CACHE_DIR, "chunk_metadata.json")


class SearchResult(TypedDict):
    score: float
    title: str
    description: str


class ChunkMetadata(TypedDict):
    movie_idx: int
    chunk_idx: int
    total_chunks: int


class SemanticSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model: SentenceTransformer = SentenceTransformer(model_name)
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
        dot_product = cast(float, np.dot(vec1, vec2))
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return cast(float, dot_product / (norm1 * norm2))


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__()
        self.chunk_embeddings: NDArray[np.float32] | None = None
        self.chunk_metadata: list[ChunkMetadata] | None = None
        self.documents: list[Movie] = []
        self.document_map: dict[int, Movie] = {}

    def build_chunk_embeddings(self, documents: list[Movie]) -> NDArray[np.float32]:
        self.documents = documents
        chunks: list[str] = []
        chunk_metadata: list[ChunkMetadata] = []
        for doc in self.documents:
            if doc["description"] == "":
                continue
            doc_chunks = semantic_chunk(doc["description"])
            chunks.extend(doc_chunks)
            self.document_map[doc["id"]] = doc
            for idx in range(len(doc_chunks)):
                chunk_metadata.append(
                    {
                        "movie_idx": doc["id"],
                        "chunk_idx": idx,
                        "total_chunks": len(doc_chunks),
                    }
                )
        self.chunk_embeddings = self.model.encode(chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata

        with open(CHUNK_EMBEDDINGS_PATH, "wb") as f:
            np.save(f, self.chunk_embeddings)
        with open(CHUNK_METADATA_PATH, "w") as f:
            json.dump(
                {"chunks": chunk_metadata, "total_chunks": len(chunks)}, f, indent=2
            )
        return self.chunk_embeddings

    def load_or_build_chunk_embeddings(
        self, documents: list[Movie]
    ) -> NDArray[np.float32]:
        self.documents = documents
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
        if os.path.exists(CHUNK_METADATA_PATH) and os.path.exists(
            CHUNK_EMBEDDINGS_PATH
        ):
            with open(CHUNK_EMBEDDINGS_PATH, "rb") as f:
                self.chunk_embeddings = np.load(f)
            with open(CHUNK_METADATA_PATH, "r") as f:
                self.chunk_metadata = json.load(f)["chunks"]
        else:
            self.chunk_embeddings = self.build_chunk_embeddings(documents)
            with open(CHUNK_METADATA_PATH, "r") as f:
                self.chunk_metadata = json.load(f)["chunks"]
        assert self.chunk_embeddings is not None
        return self.chunk_embeddings


def verify_model():
    sem_search = SemanticSearch()
    print(f"Model loaded: {sem_search.model}")
    print(f"Max sequence length: {sem_search.model.max_seq_length}")


def embed_text(text: str) -> NDArray[np.float32]:
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
        for i in range(0, len(sentences) - overlap, step)
    ]


def embed_chunks() -> None:
    css = ChunkedSemanticSearch()
    movies = load_movies()
    info = css.load_or_build_chunk_embeddings(movies)
    print(f"Generated {len(info)} chunked embeddings")

import os
from typing import Any

import numpy as np
from lib.search_utils import CACHE_DIR, Movie, load_movies
from sentence_transformers import SentenceTransformer

EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "movie_embeddings.npy")


class SemanticSearch:
    def __init__(self):
        self.model: SentenceTransformer = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings: np.ndarray | None = None
        self.documents: list[Movie] = []
        self.document_map: dict[int, Movie] = {}

    def generate_embedding(self, text: str) -> np.ndarray:
        text = text.strip()
        if text == "":
            raise ValueError("Text must not be empty")
        return self.model.encode([text])[0]

    def build_embeddings(self, documents: list[Movie]) -> np.ndarray:
        self.documents = documents
        document_str: list[str] = []
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
            document_str.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(document_str, show_progress_bar=True)
        with open(EMBEDDINGS_PATH, "wb") as f:
            np.save(f, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents: list[Movie]) -> np.ndarray:
        self.documents = documents
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
        if os.path.exists(EMBEDDINGS_PATH):
            with open(EMBEDDINGS_PATH, "rb") as f:
                self.embeddings = np.load(f)
        if self.embeddings is None or (len(self.documents) != len(self.embeddings)):
            self.embeddings = self.build_embeddings(documents)
        return self.embeddings


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

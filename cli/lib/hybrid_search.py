import os
from typing import final

from lib.search_utils import Movie

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch, SearchResult


@final
class HybridSearch:
    def __init__(self, documents: list[Movie]):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_build_chunk_embeddings(self.documents)  # pyright: ignore[reportUnusedCallResult]

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[tuple[Movie, float]]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(
        self, query: str, alpha: int, limit: int = 5
    ) -> list[SearchResult]:
        raise NotImplementedError("Weighted hybrid search is not implemented yet.")

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[SearchResult]:
        raise NotImplementedError("RRF hybrid search is not implemented yet.")

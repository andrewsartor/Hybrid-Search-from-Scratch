import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add parent directory to Python path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent))

from cli.lib.cli_base import BaseCLI
from cli.lib.semantic_search import (
    chunk,
    embed_chunks,
    embed_query_text,
    embed_text,
    search,
    search_chunks,
    semantic_chunk,
    verify_embeddings,
    verify_model,
)
from numpy.typing import NDArray


class SemanticSearchCLI(BaseCLI):
    def __init__(self):
        super().__init__("Semantic Search CLI")
        self._setup_commands()

    def _setup_commands(self) -> None:
        self.command("verify", "Verify model loads and display details")(self._verify)

        self.command("verify-embeddings", "Verify document embeddings")(
            self._verify_embeddings
        )

        self.command("embed-text", "Generate embedding for input text")(
            self._embed_text
        ).add_argument("text", type=str, help="Text to embed")

        self.command("embed-query", "Generate embedding for query text")(
            self._embed_query
        ).add_argument("query", type=str, help="Query to embed")

        self.command("embed-chunks", "Generate chunked embeddings for all documents")(
            self._embed_chunks
        )

        self.command("search", "Search documents using semantic similarity")(
            self._search
        ).add_argument("query", type=str, help="Search query").add_argument(
            "--limit", type=int, default=5, help="Maximum results to return"
        )

        self.command("search-chunks", "Search document chunks using semantic similarity")(
            self._search_chunks
        ).add_argument("query", type=str, help="Search query").add_argument(
            "--limit", type=int, default=10, help="Maximum results to return"
        )

        self.command("chunk", "Split text into word-based chunks")(
            self._chunk
        ).add_argument("text", type=str, help="Text to chunk").add_argument(
            "--chunk-size", type=int, default=200, help="Words per chunk"
        ).add_argument(
            "--overlap", type=int, default=0, help="Word overlap between chunks"
        )

        self.command("semantic-chunk", "Split text into sentence-based chunks")(
            self._semantic_chunk
        ).add_argument("text", type=str, help="Text to chunk").add_argument(
            "--max-chunk-size", type=int, default=4, help="Sentences per chunk"
        ).add_argument(
            "--overlap", type=int, default=1, help="Sentence overlap between chunks"
        )

    def _verify(self) -> str:
        verify_model()
        return ""

    def _verify_embeddings(self) -> str:
        verify_embeddings()
        return ""

    def _embed_text(self, text: str) -> NDArray[np.float32]:
        return embed_text(text)

    def _embed_query(self, query: str) -> str:
        embed_query_text(query)
        return ""

    def _embed_chunks(self) -> str:
        embed_chunks()
        return ""

    def _search(self, query: str, limit: int = 5) -> str:
        search(query, limit)
        return ""

    def _search_chunks(self, query: str, limit: int = 10) -> str:
        search_chunks(query, limit)
        return ""

    def _chunk(self, text: str, chunk_size: int = 200, overlap: int = 0) -> list[str]:
        return chunk(text, chunk_size, overlap)

    def _semantic_chunk(self, text: str, max_chunk_size: int = 4, overlap: int = 1) -> list[str]:
        return semantic_chunk(text, max_chunk_size, overlap)

    def handle_result(self, command: str, result: Any) -> None:
        if command in ["verify", "verify-embeddings", "embed-query", "embed-chunks", "search", "search-chunks"]:
            pass
        elif command == "embed-text":
            embedding = result
            print(f"First 3 dimensions: {embedding[:3]}")
            print(f"Total dimensions: {embedding.shape[0]}")
        elif command in ["chunk", "semantic-chunk"]:
            chunks = result
            chunk_type = "semantic" if command == "semantic-chunk" else "word-based"
            print(f"Created {len(chunks)} {chunk_type} chunks:")
            for i, chunk_text in enumerate(chunks, 1):
                print(f"{i}. {chunk_text}")


def main() -> None:
    cli = SemanticSearchCLI()
    cli.run()


if __name__ == "__main__":
    main()

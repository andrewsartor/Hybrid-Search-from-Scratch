import sys
from pathlib import Path
from typing import Any

# Add parent directory to Python path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent))

from cli.lib.cli_base import BaseCLI
from cli.lib.keyword_search import (
    bm25_idf_command,
    bm25_search_command,
    bm25_tf_command,
    build_command,
    idf_command,
    search_command,
    tf_command,
    tfidf_command,
)
from cli.lib.search_utils import BM25_B, BM25_K1, DEFAULT_SEARCH_LIMIT


class KeywordSearchCLI(BaseCLI):
    def __init__(self):
        super().__init__("Keyword Search CLI")
        self._setup_commands()

    def _setup_commands(self) -> None:
        self.command("build", "Build the inverted index")(self._build)

        self.command("search", "Search movies using basic keyword matching")(
            self._search
        ).add_argument("query", type=str, help="Search query").add_argument(
            "--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Maximum results"
        )

        self.command("bm25search", "Search movies using BM25 scoring")(
            self._bm25_search
        ).add_argument("query", type=str, help="Search query").add_argument(
            "--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Maximum results"
        )

        self.command("tf", "Calculate term frequency for a term in a document")(
            self._tf
        ).add_argument("doc_id", type=int, help="Document ID").add_argument(
            "term", type=str, help="Term to calculate frequency for"
        )

        self.command("idf", "Calculate inverse document frequency of a term")(
            self._idf
        ).add_argument("term", type=str, help="Term to calculate IDF for")

        self.command("tfidf", "Calculate TF-IDF score for a term in a document")(
            self._tfidf
        ).add_argument("doc_id", type=int, help="Document ID").add_argument(
            "term", type=str, help="Term to calculate TF-IDF for"
        )

        self.command("bm25tf", "Calculate BM25 TF score for a term in a document")(
            self._bm25_tf
        ).add_argument("doc_id", type=int, help="Document ID").add_argument(
            "term", type=str, help="Term to calculate BM25 TF for"
        ).add_argument(
            "--k1", type=float, default=BM25_K1, help="BM25 k1 parameter"
        ).add_argument(
            "--b", type=float, default=BM25_B, help="BM25 b parameter"
        )

        self.command("bm25idf", "Calculate BM25 IDF score for a term")(
            self._bm25_idf
        ).add_argument("term", type=str, help="Term to calculate BM25 IDF for")

    def _build(self) -> str:
        build_command()
        return "Inverted index built successfully"

    def _search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list:
        return search_command(query, limit)

    def _bm25_search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list:
        return bm25_search_command(query, limit)

    def _tf(self, doc_id: int, term: str) -> int:
        return tf_command(doc_id, term)

    def _idf(self, term: str) -> float:
        return idf_command(term)

    def _tfidf(self, doc_id: int, term: str) -> float:
        return tfidf_command(doc_id, term)

    def _bm25_tf(self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
        return bm25_tf_command(doc_id, term, k1, b)

    def _bm25_idf(self, term: str) -> float:
        return bm25_idf_command(term)

    def handle_result(self, command: str, result: Any) -> None:
        if command == "build":
            print(result)
        elif command in ["search", "bm25search"]:
            self._print_search_results(result, command == "bm25search")
        elif command == "tf":
            print(result)
        elif command in ["idf", "tfidf", "bm25tf", "bm25idf"]:
            print(f"{result:.4f}")

    def _print_search_results(self, results: list, include_scores: bool = False) -> None:
        if not results:
            print("No results found")
            return

        for i, result in enumerate(results, 1):
            if include_scores:
                movie, score = result
                print(f"{i}. ({movie['id']}) {movie['title']} - Score: {score:.4f}")
            else:
                print(f"{i}. ({result['id']}) {result['title']}")


def main() -> None:
    cli = KeywordSearchCLI()
    cli.run()


if __name__ == "__main__":
    main()

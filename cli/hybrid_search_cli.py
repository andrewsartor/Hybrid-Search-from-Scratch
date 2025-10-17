import sys
from pathlib import Path
from typing import Any

# Add parent directory to Python path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent))

from cli.lib.cli_base import BaseCLI
from cli.lib.keyword_search import bm25_search_command
from cli.lib.search_utils import DEFAULT_SEARCH_LIMIT, Movie
from cli.lib.semantic_search import SemanticSearch, load_movies


class HybridSearchCLI(BaseCLI):
    def __init__(self):
        super().__init__("Hybrid Search CLI - Combines keyword and semantic search")
        self._setup_commands()

    def _setup_commands(self) -> None:
        self.command(
            "search", "Perform hybrid search using both BM25 and semantic similarity"
        )(self._hybrid_search).add_argument(
            "query", type=str, help="Search query"
        ).add_argument(
            "--limit",
            type=int,
            default=DEFAULT_SEARCH_LIMIT,
            help="Maximum results to return",
        ).add_argument(
            "--keyword-weight",
            type=float,
            default=0.5,
            help="Weight for keyword search (0.0-1.0)",
        ).add_argument(
            "--semantic-weight",
            type=float,
            default=0.5,
            help="Weight for semantic search (0.0-1.0)",
        )

    def _hybrid_search(
        self,
        query: str,
        limit: int = DEFAULT_SEARCH_LIMIT,
        keyword_weight: float = 0.5,
        semantic_weight: float = 0.5,
    ) -> list[tuple[Movie, float]]:
        if abs(keyword_weight + semantic_weight - 1.0) > 0.001:
            raise ValueError("Keyword and semantic weights must sum to 1.0")

        keyword_results = bm25_search_command(query, limit * 2)

        semantic_search = SemanticSearch()
        movies = load_movies()
        semantic_search.load_or_create_embeddings(movies)
        semantic_results = semantic_search.search(query, limit * 2)

        keyword_scores = {result[0]["id"]: result[1] for result in keyword_results}
        semantic_scores = {
            movie["id"]: result["score"]
            for movie, result in zip(movies, semantic_results)
            if movie["title"] == result["title"]
        }

        all_movie_ids = set(keyword_scores.keys()) | set(semantic_scores.keys())

        combined_results = []
        for movie_id in all_movie_ids:
            movie = next(m for m in movies if m["id"] == movie_id)

            keyword_score = keyword_scores.get(movie_id, 0.0)
            semantic_score = semantic_scores.get(movie_id, 0.0)

            if keyword_score > 0:
                keyword_score = (
                    keyword_score / max(keyword_scores.values())
                    if keyword_scores
                    else 0
                )
            if semantic_score > 0:
                semantic_score = (
                    semantic_score / max(semantic_scores.values())
                    if semantic_scores
                    else 0
                )

            combined_score = (keyword_weight * keyword_score) + (
                semantic_weight * semantic_score
            )

            if combined_score > 0:
                combined_results.append((movie, combined_score))

        combined_results.sort(key=lambda x: x[1], reverse=True)
        return combined_results[:limit]

    def handle_result(self, command: str, result: Any) -> None:
        if command == "search":
            results = result
            if not results:
                print("No results found")
                return

            print("Hybrid search results:")
            for i, (movie, score) in enumerate(results, 1):
                print(f"{i}. ({movie['id']}) {movie['title']} - Score: {score:.4f}")
                print(f"   {movie['description'][:100]}...")
                print()


def main() -> None:
    cli = HybridSearchCLI()
    cli.run()


if __name__ == "__main__":
    main()

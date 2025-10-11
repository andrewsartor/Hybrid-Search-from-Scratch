import argparse
import json


def keyword_search(keyword: str) -> list[str]:
    with open("data/movies.json") as f:
        data = json.load(f)
        matches = []
        for movie in data.get("movies"):
            if keyword in movie.get("title"):
                matches.append((movie.get("id"), movie.get("title")))
    results = [f"{movie[1]} {movie[0]}" for movie in sorted(matches)[:5]]
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print("Searching for:", args.query)
            results = keyword_search(args.query)
            for i, val in enumerate(results):
                print(f"{i + 1}. {val}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()

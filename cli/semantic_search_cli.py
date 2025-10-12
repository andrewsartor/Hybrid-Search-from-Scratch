import argparse

from lib.semantic_search import (
    embed_query_text,
    embed_text,
    search,
    verify_embeddings,
    verify_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    _ = subparsers.add_parser("verify", help="Verify model loads and display details.")
    embed_text_parser = subparsers.add_parser(
        "embed_text", help="Generate embedding for input text"
    )
    _ = embed_text_parser.add_argument("text", type=str, help="Text to embed")
    _ = subparsers.add_parser("verify_embeddings", help="Verify embeddings")
    embedquery_parser = subparsers.add_parser("embedquery", help="Embed query")
    _ = embedquery_parser.add_argument("query", type=str, help="The query to embed")
    search_parser = subparsers.add_parser("search", help="Search documents for string")
    _ = search_parser.add_argument("query", type=str, help="String to search for")
    _ = search_parser.add_argument(
        "--limit", type=int, nargs="?", default=5, help="String to search for"
    )

    args = parser.parse_args()

    match args.command:
        case "search":
            search(args.query, args.limit)
        case "embedquery":
            embed_query_text(args.query)
        case "verify_embeddings":
            verify_embeddings()
        case "embed_text":
            print(f"Text: {args.text}")
            embedding = embed_text(args.text)
            print(f"First 3 dimensions: {embedding[:3]}")
            print(f"Dimensions: {embedding.shape[0]}")
        case "verify":
            verify_model()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()

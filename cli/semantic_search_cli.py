import argparse

from lib.semantic_search import (
    chunk,
    embed_query_text,
    embed_text,
    search,
    semantic_chunk,
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
    chunk_parser = subparsers.add_parser(
        "chunk", help="Chunk text into chunks of a given length."
    )
    _ = chunk_parser.add_argument("text", type=str, help="Text to chunk")
    _ = chunk_parser.add_argument(
        "--chunk-size", type=int, nargs="?", default=200, help="String to search for"
    )
    _ = chunk_parser.add_argument(
        "--overlap",
        type=int,
        nargs="?",
        default=0,
        help="Amount of words to overlap in each chunk",
    )
    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk", help="Chunk text into chunks of a given length."
    )
    _ = semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk")
    _ = semantic_chunk_parser.add_argument(
        "--max-chunk-size",
        type=int,
        nargs="?",
        default=4,
        help="String to search for",
    )
    _ = semantic_chunk_parser.add_argument(
        "--overlap",
        type=int,
        nargs="?",
        default=1,
        help="Amount of words to overlap in each chunk",
    )

    args = parser.parse_args()

    match args.command:
        case "semantic_chunk":
            chunks = semantic_chunk(args.text, args.max_chunk_size, args.overlap)
            print(f"Semantically chunking {len(args.text)} characters")
            for i, c in enumerate(chunks, 1):
                print(f"{i}. {c}")
        case "chunk":
            chunks = chunk(args.text, args.chunk_size, args.overlap)
            print(f"Chunking {len(args.text)} characters")
            for i, c in enumerate(chunks, 1):
                print(f"{i}. {c}")
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

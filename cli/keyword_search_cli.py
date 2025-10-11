import argparse

from lib.keyword_search import (
    bm25_idf_command,
    bm25_tf_command,
    build_command,
    idf_command,
    search_command,
    tf_command,
    tfidf_command,
)
from lib.search_utils import BM25_B, BM25_K1


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    _ = subparsers.add_parser("build", help="Build the inverted index")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    _ = search_parser.add_argument("query", type=str, help="Search query")

    tf_parser = subparsers.add_parser(
        "tf", help="Finds the term frequency for a term in a given document"
    )
    _ = tf_parser.add_argument(
        "doc_id", type=int, help="The document ID in which to search"
    )
    _ = tf_parser.add_argument(
        "term", type=str, help="The term whose frequency will be returned"
    )

    idf_parser = subparsers.add_parser(
        "idf", help="Calculate the Inverse Document Frequency of a given term"
    )
    _ = idf_parser.add_argument("term", type=str, help="The term to calculate IDF for")

    tfidf_parser = subparsers.add_parser(
        "tfidf", help="Calculate the TD-IDF of a given term"
    )
    _ = tfidf_parser.add_argument(
        "doc_id", type=int, help="Document to calculate TF-IDF"
    )
    _ = tfidf_parser.add_argument("term", type=str, help="Term to calculate TF-IDF for")

    bm25_idf_parser = subparsers.add_parser(
        "bm25idf", help="Calculate the bm25idf of a given term"
    )
    _ = bm25_idf_parser.add_argument("term", type=str, help="Term to calculate for")

    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    _ = bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    _ = bm25_tf_parser.add_argument(
        "term", type=str, help="Term to get BM25 TF score for"
    )
    _ = bm25_tf_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 parameter"
    )
    _ = bm25_tf_parser.add_argument(
        "b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 parameter"
    )

    args = parser.parse_args()

    match args.command:
        case "bm25tf":
            bm25tf = bm25_tf_command(args.doc_id, args.term, args.k1, args.b)
            print(
                f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}"
            )
        case "bm25idf":
            bm25idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case "tfidf":
            tf_idf = tfidf_command(args.doc_id, args.term)
            print(
                f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}"
            )
        case "idf":
            idf = idf_command(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tf":
            print(tf_command(args.doc_id, args.term))
        case "build":
            print("Building inverted index...")
            build_command()
            print("Inverted index built successfully.")
        case "search":
            print("Searching for:", args.query)
            results = search_command(args.query)
            for i, res in enumerate(results, 1):
                print(f"{i}. ({res['id']}) {res['title']}")
        case "build":
            print("First document for token 'merida' = 4651")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()

import argparse

from lib.keyword_search import build_command, idf_command, search_command, tf_command


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

    args = parser.parse_args()

    match args.command:
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

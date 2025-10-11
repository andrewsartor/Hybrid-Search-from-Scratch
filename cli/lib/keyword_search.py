import string

from .search_utils import (
    DEFAULT_SEARCH_LIMIT,
    load_movies,
)


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    results = []
    for movie in movies:
        query_tokens = tokenize_text(query)
        title_tokens = tokenize_text(movie["title"])
        if has_matching_token(query_tokens, title_tokens):
            results.append(movie)
            if len(results) >= limit:
                break
    return results


def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    return len(set(query_tokens) & set(title_tokens)) > 0


def preprocess_text(text: str) -> str:
    text = text.lower()
    remove_punctuation = str.maketrans("", "", string.punctuation)
    text = text.translate(remove_punctuation)
    return text


def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens = []
    for token in tokens:
        if token:
            valid_tokens.append(token)
    return valid_tokens

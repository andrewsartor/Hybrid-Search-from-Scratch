import string

from .search_utils import (
    DEFAULT_SEARCH_LIMIT,
    load_movies,
)


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    results = []
    for movie in movies:
        preprocessed_query = preprocess_text(query)
        preprocessed_title = preprocess_text(movie["title"])
        if set(preprocessed_query) & set(preprocessed_title):
            results.append(movie)
            if len(results) >= limit:
                break
    return results


def preprocess_text(text: str) -> list[str]:
    text = text.lower()
    remove_punctuation = str.maketrans("", "", string.punctuation)
    text = text.translate(remove_punctuation)
    text_list = text.split()
    return text_list

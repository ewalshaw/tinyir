"""Bundled demo for TinyIR."""

from pathlib import Path

from .corpus import Corpus
from .index import InvertedIndex
from .rank import format_search_table, rank
from .tokenize import tokenize

CORPUS_DIR = Path(__file__).resolve().parent.parent / "examples" / "corpus"

SAMPLE_QUERIES = [
    "machine learning data",
    "water plants garden",
    "search engines ranking",
    "roman history amphitheater",
    "coffee morning routine",
    "the and in",
    "xylophone zebra",
]


def run_demo() -> None:
    """Index the example corpus and print sample search results."""
    if not CORPUS_DIR.is_dir():
        raise FileNotFoundError(
            f"Demo corpus not found at {CORPUS_DIR}. "
            "Run TinyIR from the repository checkout."
        )

    corpus = Corpus(CORPUS_DIR)
    print(f"Indexing {len(corpus.documents)} documents from {CORPUS_DIR}")

    index = InvertedIndex()
    index.build(corpus, remove_stopwords=True)

    for query in SAMPLE_QUERIES:
        print(f"\nQuery: {query!r}")
        query_terms = tokenize(query, remove_stopwords=index.remove_stopwords)
        if not query_terms:
            print("Query empty after tokenization.")
            continue

        results = rank(index, query, top_k=5)
        if not results:
            print("No results found.")
            continue
        print(format_search_table(results))

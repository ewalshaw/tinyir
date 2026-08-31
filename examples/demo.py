"""Demo script for TinyIR."""

from pathlib import Path

from tinyir.corpus import Corpus
from tinyir.index import InvertedIndex
from tinyir.rank import format_search_table, rank

CORPUS_DIR = Path(__file__).resolve().parent / "corpus"

SAMPLE_QUERIES = [
    "machine learning data",
    "water plants garden",
    "search engines ranking",
    "roman history amphitheater",
    "coffee morning routine",
]


def main() -> None:
    corpus = Corpus(CORPUS_DIR)
    print(f"Indexing {len(corpus.documents)} documents from {CORPUS_DIR}")

    index = InvertedIndex()
    index.build(corpus)

    for query in SAMPLE_QUERIES:
        results = rank(index, query, top_k=5)
        print(f"\nQuery: {query}")
        if not results:
            print("No results found.")
            continue
        print(format_search_table(results))


if __name__ == "__main__":
    main()

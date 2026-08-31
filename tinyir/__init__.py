"""TinyIR: minimal information retrieval engine."""

from .corpus import Corpus, Document, load_corpus
from .index import InvertedIndex, build_index, build_index_from_dir
from .rank import SearchResult, rank, rank_query
from .tokenize import STOPWORDS, tokenize

__all__ = [
    "Corpus",
    "Document",
    "STOPWORDS",
    "InvertedIndex",
    "SearchResult",
    "build_index",
    "build_index_from_dir",
    "load_corpus",
    "rank",
    "rank_query",
    "tokenize",
]
__version__ = "0.1.0"

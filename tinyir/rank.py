"""Query ranking for TinyIR."""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .index import InvertedIndex
from .tokenize import tokenize


@dataclass(frozen=True)
class SearchResult:
    """A single ranked search hit."""

    doc_id: int
    filename: str
    path: str
    score: float


def candidate_doc_indices(index: InvertedIndex, query_terms: list[str]) -> list[int]:
    """Return sorted document indices that contain at least one query term."""
    candidate_ids: set[int] = set()
    for term in query_terms:
        if term not in index.vocabulary:
            continue
        for doc_id, _ in index.inverted_index.get(term, []):
            candidate_ids.add(doc_id)
    return sorted(candidate_ids)


def rank(index: InvertedIndex, query: str, top_k: int = 10) -> list[SearchResult]:
    """Vectorize ``query`` and return the top ``top_k`` documents by cosine similarity."""
    if (
        not index.documents
        or index.vectorizer is None
        or index.document_vectors is None
    ):
        return []

    query_terms = tokenize(query, remove_stopwords=index.remove_stopwords)
    if not query_terms:
        return []

    candidate_ids = candidate_doc_indices(index, query_terms)
    if not candidate_ids:
        return []

    query_vector = index.vectorizer.transform([" ".join(query_terms)])
    candidate_vectors = index.document_vectors[candidate_ids]
    scores = cosine_similarity(query_vector, candidate_vectors).flatten()

    ranked_positions = np.argsort(scores)[::-1]
    results: list[SearchResult] = []
    for position in ranked_positions:
        score = float(scores[position])
        if score <= 0.0:
            break
        document = index.documents[candidate_ids[position]]
        results.append(
            SearchResult(
                doc_id=document.doc_id,
                filename=document.filename,
                path=str(document.path),
                score=score,
            )
        )
        if len(results) >= top_k:
            break
    return results


def rank_query(index: InvertedIndex, query: str, top_k: int = 10) -> list[SearchResult]:
    """Alias for :func:`rank`."""
    return rank(index, query, top_k=top_k)


def format_search_table(results: list[SearchResult]) -> str:
    """Format ranked results as a fixed-width table."""
    if not results:
        return ""

    rank_width = max(len("Rank"), len(str(len(results))))
    similarity_width = max(len("Similarity"), 8)
    doc_id_width = max(len("doc_id"), *(len(str(r.doc_id)) for r in results))
    filename_width = max(len("Filename"), *(len(r.filename) for r in results))

    header = (
        f"{'Rank':>{rank_width}}  "
        f"{'Similarity':>{similarity_width}}  "
        f"{'doc_id':>{doc_id_width}}  "
        f"{'Filename':<{filename_width}}"
    )
    divider = (
        f"{'-' * rank_width}  "
        f"{'-' * similarity_width}  "
        f"{'-' * doc_id_width}  "
        f"{'-' * filename_width}"
    )

    lines = [header, divider]
    for position, result in enumerate(results, start=1):
        lines.append(
            f"{position:>{rank_width}}  "
            f"{result.score:>{similarity_width}.4f}  "
            f"{result.doc_id:>{doc_id_width}}  "
            f"{result.filename:<{filename_width}}"
        )
    return "\n".join(lines)

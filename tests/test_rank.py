"""Tests for TinyIR query ranking."""

from pathlib import Path

import pytest

from tinyir.corpus import Corpus, Document
from tinyir.index import InvertedIndex
from tinyir.rank import SearchResult, candidate_doc_indices, format_search_table, rank


@pytest.fixture
def ranked_corpus_dir(tmp_path: Path) -> Path:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "cats.txt").write_text(
        "cats love fish and napping in the sun",
        encoding="utf-8",
    )
    (corpus_dir / "dogs.txt").write_text(
        "dogs love bones and running in the park",
        encoding="utf-8",
    )
    (corpus_dir / "birds.txt").write_text(
        "birds fly high above lakes and trees",
        encoding="utf-8",
    )
    return corpus_dir


@pytest.fixture
def search_index(ranked_corpus_dir: Path) -> InvertedIndex:
    index = InvertedIndex()
    index.build(Corpus(ranked_corpus_dir))
    return index


def test_rank_returns_results_in_descending_score_order(search_index: InvertedIndex):
    results = rank(search_index, "cats fish", top_k=3)

    assert len(results) >= 1
    scores = [result.score for result in results]
    assert scores == sorted(scores, reverse=True)
    assert all(score > 0 for score in scores)


def test_rank_puts_most_relevant_document_first(search_index: InvertedIndex):
    results = rank(search_index, "cats fish", top_k=3)

    assert results[0].filename == "cats.txt"
    assert results[0].score > 0


def test_rank_includes_document_metadata(search_index: InvertedIndex):
    results = rank(search_index, "birds fly", top_k=1)

    birds_doc = next(doc for doc in search_index.documents if doc.filename == "birds.txt")

    assert len(results) == 1
    assert results[0].filename == "birds.txt"
    assert results[0].doc_id == birds_doc.doc_id
    assert results[0].path.endswith("birds.txt")


def test_rank_respects_top_k(search_index: InvertedIndex):
    results = rank(search_index, "love", top_k=2)
    assert len(results) == 2


def test_rank_returns_empty_list_for_unbuilt_index():
    index = InvertedIndex()
    assert rank(index, "query") == []


def test_rank_with_manual_documents():
    documents = [
        Document(0, "relevant.txt", Path("relevant.txt"), "python python python"),
        Document(1, "other.txt", Path("other.txt"), "java code"),
    ]
    corpus = Corpus.__new__(Corpus)
    corpus.documents = documents

    index = InvertedIndex()
    index.build(corpus)
    results = rank(index, "python", top_k=2)

    assert len(results) == 1
    assert results[0].filename == "relevant.txt"
    assert results[0].score > 0


def test_rank_excludes_zero_similarity_results(search_index: InvertedIndex):
    results = rank(search_index, "xylophone zebra", top_k=3)

    assert results == []


def test_candidate_doc_indices_returns_union_of_posting_lists(search_index: InvertedIndex):
    cats_doc = next(doc for doc in search_index.documents if doc.filename == "cats.txt")
    dogs_doc = next(doc for doc in search_index.documents if doc.filename == "dogs.txt")

    assert candidate_doc_indices(search_index, ["cats", "fish"]) == [cats_doc.doc_id]
    assert candidate_doc_indices(search_index, ["love"]) == sorted(
        [cats_doc.doc_id, dogs_doc.doc_id]
    )
    assert candidate_doc_indices(search_index, ["xylophone"]) == []


def test_rank_limits_scoring_to_inverted_index_candidates(
    search_index: InvertedIndex,
    monkeypatch,
):
    import sys

    rank_module = sys.modules["tinyir.rank"]
    scored_doc_count: list[int] = []

    def spy_cosine_similarity(query_vector, document_vectors):
        scored_doc_count.append(document_vectors.shape[0])
        from sklearn.metrics.pairwise import cosine_similarity as real_cosine_similarity

        return real_cosine_similarity(query_vector, document_vectors)

    monkeypatch.setattr(rank_module, "cosine_similarity", spy_cosine_similarity)

    rank(search_index, "cats fish", top_k=3)

    assert scored_doc_count == [1]
    assert len(search_index.documents) == 3


def test_rank_returns_empty_for_stopword_only_query_with_stopwords_removed(
    ranked_corpus_dir: Path,
):
    index = InvertedIndex()
    index.build(Corpus(ranked_corpus_dir), remove_stopwords=True)

    assert rank(index, "the and in", top_k=3) == []


def test_format_search_table_includes_required_columns():
    results = [
        SearchResult(doc_id=3, filename="cats.txt", path="/tmp/cats.txt", score=0.8123),
        SearchResult(doc_id=12, filename="dogs.txt", path="/tmp/dogs.txt", score=0.401),
    ]
    table = format_search_table(results)

    assert "Rank" in table
    assert "Similarity" in table
    assert "doc_id" in table
    assert "Filename" in table
    lines = table.splitlines()
    assert lines[2].endswith("cats.txt")
    assert lines[3].endswith("dogs.txt")

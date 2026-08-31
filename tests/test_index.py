"""Tests for the TinyIR inverted index and TF-IDF vectors."""

from pathlib import Path

import pytest

from tinyir.corpus import Corpus, Document
from tinyir.index import (
    DOCUMENTS_FILE,
    INVERTED_INDEX_FILE,
    TFIDF_MODEL_FILE,
    VOCABULARY_FILE,
    InvertedIndex,
)


@pytest.fixture
def sample_corpus_dir(tmp_path: Path) -> Path:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "a.txt").write_text("cat cat dog", encoding="utf-8")
    (corpus_dir / "b.txt").write_text("dog bird", encoding="utf-8")
    (corpus_dir / "c.txt").write_text("bird fish", encoding="utf-8")
    return corpus_dir


@pytest.fixture
def built_index(sample_corpus_dir: Path) -> InvertedIndex:
    index = InvertedIndex()
    index.build(Corpus(sample_corpus_dir))
    return index


def test_inverted_index_maps_terms_to_doc_id_and_term_freq(built_index: InvertedIndex):
    assert built_index.inverted_index["cat"] == [(0, 2)]
    assert built_index.inverted_index["dog"] == [(0, 1), (1, 1)]
    assert built_index.inverted_index["bird"] == [(1, 1), (2, 1)]
    assert built_index.inverted_index["fish"] == [(2, 1)]


def test_inverted_index_postings_are_sorted_by_doc_id(built_index: InvertedIndex):
    for postings in built_index.inverted_index.values():
        doc_ids = [doc_id for doc_id, _ in postings]
        assert doc_ids == sorted(doc_ids)


def test_tfidf_document_vectors_have_expected_shape(built_index: InvertedIndex):
    assert built_index.document_vectors is not None
    assert built_index.document_vectors.shape == (3, 4)


def test_vocabulary_matches_vectorizer_terms(built_index: InvertedIndex):
    assert built_index.vectorizer is not None
    assert set(built_index.vocabulary.keys()) == {"cat", "dog", "bird", "fish"}
    assert built_index.vocabulary == built_index.vectorizer.vocabulary_


def test_build_with_manual_documents():
    documents = [
        Document(0, "one.txt", Path("one.txt"), "alpha beta"),
        Document(1, "two.txt", Path("two.txt"), "beta gamma"),
    ]
    corpus = Corpus.__new__(Corpus)
    corpus.documents = documents

    index = InvertedIndex()
    index.build(corpus)

    assert index.inverted_index["alpha"] == [(0, 1)]
    assert index.inverted_index["beta"] == [(0, 1), (1, 1)]
    assert index.inverted_index["gamma"] == [(1, 1)]
    assert index.document_vectors.shape == (2, 3)


def test_save_and_load_round_trip(built_index: InvertedIndex, tmp_path: Path):
    index_dir = tmp_path / ".tinyir"
    built_index.save(index_dir)

    assert (index_dir / TFIDF_MODEL_FILE).exists()
    assert (index_dir / VOCABULARY_FILE).exists()
    assert (index_dir / DOCUMENTS_FILE).exists()
    assert (index_dir / INVERTED_INDEX_FILE).exists()

    loaded = InvertedIndex.load(index_dir)
    assert loaded.document_vectors.shape == built_index.document_vectors.shape
    assert loaded.vocabulary == built_index.vocabulary
    assert loaded.inverted_index == built_index.inverted_index
    assert [doc.filename for doc in loaded.documents] == [
        doc.filename for doc in built_index.documents
    ]


def test_load_fails_when_inverted_index_missing(built_index: InvertedIndex, tmp_path: Path):
    index_dir = tmp_path / ".tinyir"
    built_index.save(index_dir)
    (index_dir / INVERTED_INDEX_FILE).unlink()

    with pytest.raises(FileNotFoundError, match="inverted_index.pkl"):
        InvertedIndex.load(index_dir)


def test_build_with_remove_stopwords(tmp_path: Path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "a.txt").write_text("the cat and the dog", encoding="utf-8")

    index = InvertedIndex()
    index.build(Corpus(corpus_dir), remove_stopwords=True)

    assert index.remove_stopwords is True
    assert "the" not in index.inverted_index
    assert "and" not in index.inverted_index
    assert "cat" in index.inverted_index

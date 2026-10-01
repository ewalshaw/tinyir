"""Tests for TinyIR corpus loading."""

from pathlib import Path

import pytest

from tinyir.corpus import Corpus


def test_corpus_skips_invalid_utf8(tmp_path: Path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "good.txt").write_text("hello", encoding="utf-8")
    (corpus_dir / "bad.txt").write_bytes(b"\xff\xfe")

    with pytest.warns(UserWarning, match="Skipping bad.txt"):
        corpus = Corpus(corpus_dir)

    assert [doc.filename for doc in corpus.documents] == ["good.txt"]
    assert corpus.documents[0].doc_id == 0
    assert corpus.documents[0].text == "hello"

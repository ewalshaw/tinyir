"""Tests for the TinyIR tokenizer."""

import pytest

from tinyir.tokenize import STOPWORDS, tokenize


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hello, World!", ["hello", "world"]),
        ("Mixed CASE text.", ["mixed", "case", "text"]),
        ("multiple   spaced\twords", ["multiple", "spaced", "words"]),
        ("", []),
    ],
)
def test_tokenize_lowercases_and_strips_punctuation(text, expected):
    assert tokenize(text) == expected


def test_tokenize_removes_stopwords_when_requested():
    result = tokenize("The cat and the dog", remove_stopwords=True)
    assert result == ["cat", "dog"]
    assert "the" not in result
    assert "and" not in result


def test_tokenize_keeps_stopwords_by_default():
    result = tokenize("the cat is in the hat")
    assert "the" in result
    assert "is" in result
    assert "in" in result


def test_stopwords_is_frozen_set_of_common_words():
    assert isinstance(STOPWORDS, frozenset)
    assert len(STOPWORDS) >= 80
    assert "the" in STOPWORDS
    assert "and" in STOPWORDS
    assert "this" in STOPWORDS
    assert "you" in STOPWORDS
    assert "she" in STOPWORDS


def test_tokenize_filters_expanded_stopwords():
    result = tokenize("this she you your", remove_stopwords=True)
    assert result == []

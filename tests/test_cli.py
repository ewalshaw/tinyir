"""Integration tests for the TinyIR CLI."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from tinyir.cli import main, tinyir
from tinyir.index import DOCUMENTS_FILE, INVERTED_INDEX_FILE, TFIDF_MODEL_FILE, VOCABULARY_FILE


def _make_corpus(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "cats.txt").write_text(
        "cats love fish and napping in the sun",
        encoding="utf-8",
    )
    (directory / "dogs.txt").write_text(
        "dogs love bones and running in the park",
        encoding="utf-8",
    )
    return directory


def test_index_creates_artifacts(tmp_path: Path):
    corpus_dir = _make_corpus(tmp_path / "corpus")
    index_dir = tmp_path / "index"

    runner = CliRunner()
    result = runner.invoke(tinyir, ["index", str(corpus_dir), "-o", str(index_dir)])

    assert result.exit_code == 0
    assert (index_dir / TFIDF_MODEL_FILE).exists()
    assert (index_dir / VOCABULARY_FILE).exists()
    assert (index_dir / DOCUMENTS_FILE).exists()
    assert (index_dir / INVERTED_INDEX_FILE).exists()
    assert "Indexed 2 document(s)" in result.output


def test_index_skips_invalid_utf8(tmp_path: Path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "good.txt").write_text("hello world", encoding="utf-8")
    (corpus_dir / "bad.txt").write_bytes(b"\xff\xfe")
    index_dir = tmp_path / "index"

    runner = CliRunner()
    with pytest.warns(UserWarning, match="Skipping bad.txt"):
        result = runner.invoke(
            tinyir,
            ["index", str(corpus_dir), "-o", str(index_dir)],
        )

    assert result.exit_code == 0
    assert "Indexed 1 document(s)" in result.output


def test_index_skips_some_invalid_utf8_and_indexes_the_rest(tmp_path: Path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "bad_a.txt").write_bytes(b"\xff\xfe")
    (corpus_dir / "bad_b.txt").write_bytes(b"\xff\xfe")
    (corpus_dir / "good_a.txt").write_text("one two three", encoding="utf-8")
    (corpus_dir / "good_b.txt").write_text("hello world", encoding="utf-8")
    index_dir = tmp_path / "index"

    runner = CliRunner()
    with pytest.warns(UserWarning) as caught:
        result = runner.invoke(
            tinyir,
            ["index", str(corpus_dir), "-o", str(index_dir)],
        )

    assert [str(warning.message) for warning in caught] == [
        "Skipping bad_a.txt: not valid UTF-8",
        "Skipping bad_b.txt: not valid UTF-8",
    ]
    assert result.exit_code == 0
    assert "Indexed 2 document(s)" in result.output


def test_index_fails_on_empty_folder(tmp_path: Path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    runner = CliRunner()
    result = runner.invoke(tinyir, ["index", str(empty_dir)])

    assert result.exit_code != 0
    assert "No .txt files found" in result.output


def test_index_fails_when_all_files_are_invalid_utf8(tmp_path: Path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "bad.txt").write_bytes(b"\xff\xfe")

    runner = CliRunner()
    with pytest.warns(UserWarning, match="Skipping bad.txt"):
        result = runner.invoke(tinyir, ["index", str(corpus_dir)])

    assert result.exit_code != 0
    assert "No valid UTF-8 .txt files found" in result.output
    assert "No .txt files found" not in result.output


def test_index_fails_when_five_files_are_invalid_utf8(tmp_path: Path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    names = [f"bad_{i}.txt" for i in range(1, 6)]
    for name in names:
        (corpus_dir / name).write_bytes(b"\xff\xfe")

    runner = CliRunner()
    with pytest.warns(UserWarning) as caught:
        result = runner.invoke(tinyir, ["index", str(corpus_dir)])

    assert [str(warning.message) for warning in caught] == [
        f"Skipping {name}: not valid UTF-8" for name in names
    ]
    assert result.exit_code != 0
    assert "No valid UTF-8 .txt files found" in result.output
    assert "No .txt files found" not in result.output


def test_search_returns_ranked_results(tmp_path: Path):
    corpus_dir = _make_corpus(tmp_path / "corpus")
    index_dir = tmp_path / "index"
    runner = CliRunner()

    index_result = runner.invoke(
        tinyir,
        ["index", str(corpus_dir), "-o", str(index_dir)],
    )
    assert index_result.exit_code == 0

    search_result = runner.invoke(
        tinyir,
        ["search", "cats fish", "-i", str(index_dir), "-k", "2"],
    )

    assert search_result.exit_code == 0
    assert "cats.txt" in search_result.output
    assert "Rank" in search_result.output
    assert "Similarity" in search_result.output


def test_search_fails_when_index_missing(tmp_path: Path):
    missing_index_dir = tmp_path / "missing-index"
    missing_index_dir.mkdir()

    runner = CliRunner()
    result = runner.invoke(
        tinyir,
        ["search", "query", "-i", str(missing_index_dir)],
    )

    assert result.exit_code != 0
    assert "missing required pickle files" in result.output


def test_search_suggests_index_when_directory_missing(tmp_path: Path):
    missing_dir = tmp_path / "no-such-index"

    runner = CliRunner()
    result = runner.invoke(
        tinyir,
        ["search", "query", "-i", str(missing_dir)],
    )

    assert result.exit_code != 0
    assert "Directory does not exist." not in result.output
    assert (
        "Error: Index directory "
        f"{missing_dir} does not exist.\n"
        "Run 'tinyir index <folder>' first, or 'tinyir --help' for more information."
    ) in result.output


def test_index_with_remove_stopwords(tmp_path: Path):
    corpus_dir = _make_corpus(tmp_path / "corpus")
    index_dir = tmp_path / "index"

    runner = CliRunner()
    index_result = runner.invoke(
        tinyir,
        ["index", str(corpus_dir), "-o", str(index_dir), "--remove-stopwords"],
    )
    assert index_result.exit_code == 0

    search_result = runner.invoke(
        tinyir,
        ["search", "the and in", "-i", str(index_dir)],
    )
    assert search_result.exit_code == 0
    assert "Query empty after tokenization." in search_result.output
    assert "No results found." not in search_result.output


def test_search_reports_empty_query_for_whitespace(tmp_path: Path):
    corpus_dir = _make_corpus(tmp_path / "corpus")
    index_dir = tmp_path / "index"
    runner = CliRunner()

    index_result = runner.invoke(
        tinyir,
        ["index", str(corpus_dir), "-o", str(index_dir)],
    )
    assert index_result.exit_code == 0

    search_result = runner.invoke(
        tinyir,
        ["search", "   ", "-i", str(index_dir)],
    )
    assert search_result.exit_code == 0
    assert "Query empty after tokenization." in search_result.output
    assert "No results found." not in search_result.output


def test_search_reports_no_results_for_unknown_terms(tmp_path: Path):
    corpus_dir = _make_corpus(tmp_path / "corpus")
    index_dir = tmp_path / "index"
    runner = CliRunner()

    index_result = runner.invoke(
        tinyir,
        ["index", str(corpus_dir), "-o", str(index_dir)],
    )
    assert index_result.exit_code == 0

    search_result = runner.invoke(
        tinyir,
        ["search", "xylophone zebra", "-i", str(index_dir)],
    )
    assert search_result.exit_code == 0
    assert "No results found." in search_result.output
    assert "Query empty after tokenization." not in search_result.output


def test_demo_flag_runs_bundled_demo(capsys):
    main(["--demo"])
    output = capsys.readouterr().out

    assert "Indexing" in output
    assert "machine_learning.txt" in output
    assert "Query empty after tokenization." in output
    assert "No results found." in output


def test_demo_flag_ignores_other_args(capsys):
    main(["--demo", "search", "ignored", "--remove-stopwords", "-k", "1"])
    output = capsys.readouterr().out

    assert "Indexing" in output
    assert "Error" not in output

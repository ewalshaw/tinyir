"""Integration tests for the TinyIR CLI."""

from pathlib import Path

from click.testing import CliRunner

from tinyir.cli import tinyir
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


def test_index_fails_on_empty_folder(tmp_path: Path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    runner = CliRunner()
    result = runner.invoke(tinyir, ["index", str(empty_dir)])

    assert result.exit_code != 0
    assert "No .txt files found" in result.output


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

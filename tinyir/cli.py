"""CLI for TinyIR."""

import pickle
from pathlib import Path

import click

from .corpus import Corpus
from .index import InvertedIndex, build_index
from .rank import format_search_table, rank


DEFAULT_INDEX_DIR = Path(".tinyir")


@click.group()
def tinyir() -> None:
    """Minimal information retrieval engine."""


@tinyir.command("index")
@click.argument("folder", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(file_okay=False, path_type=Path),
    default=DEFAULT_INDEX_DIR,
    show_default=True,
    help="Directory for saved pickle artifacts.",
)
@click.option(
    "--remove-stopwords",
    is_flag=True,
    default=False,
    help="Remove common English stopwords during indexing and search.",
)
def index_cmd(folder: Path, output: Path, remove_stopwords: bool) -> None:
    """Build the corpus, inverted index, and TF-IDF model."""
    corpus = Corpus(folder)
    if not corpus.documents:
        raise click.ClickException(f"No .txt files found in {folder}")

    search_index = build_index(
        corpus,
        output,
        remove_stopwords=remove_stopwords,
    )
    click.echo(
        f"Indexed {len(search_index.documents)} document(s) from {folder}"
    )
    click.echo(
        f"Saved TF-IDF model, vocabulary, inverted index, and document metadata to {output.resolve()}"
    )


@tinyir.command("search")
@click.argument("query")
@click.option(
    "--index-dir",
    "-i",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=DEFAULT_INDEX_DIR,
    show_default=True,
    help="Directory containing saved pickle artifacts.",
)
@click.option("--top-k", "-k", default=10, show_default=True, help="Number of results.")
def search_cmd(query: str, index_dir: Path, top_k: int) -> None:
    """Load the saved index and print ranked search results."""
    try:
        search_index = InvertedIndex.load(index_dir)
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc
    except (pickle.UnpicklingError, EOFError, KeyError) as exc:
        raise click.ClickException(
            f"Failed to load index artifacts from {index_dir}: {exc}"
        ) from exc

    results = rank(search_index, query, top_k=top_k)
    if not results:
        click.echo("No results found.")
        return

    click.echo(format_search_table(results))


main = tinyir

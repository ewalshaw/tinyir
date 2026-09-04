"""CLI for TinyIR."""

import pickle
import sys
from pathlib import Path

import click

from .corpus import Corpus
from .demo import run_demo
from .index import InvertedIndex, build_index
from .rank import format_search_table, rank
from .tokenize import tokenize


DEFAULT_INDEX_DIR = Path(".tinyir")


@click.group(invoke_without_command=True)
@click.option(
    "--demo",
    is_flag=True,
    is_eager=True,
    help="Run the bundled example demo (ignores other options and commands).",
)
@click.pass_context
def tinyir(ctx: click.Context, demo: bool) -> None:
    """Minimal information retrieval engine.

    \b
    Quick start:
      tinyir --demo
          Index examples/corpus in memory and run sample queries.
      tinyir index <folder>
          Build and save an index under .tinyir/ (or -o <path>).
      tinyir search "<query>"
          Rank documents from a previously built index.
    """
    if demo:
        run_demo()
        ctx.exit()
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


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
    type=click.Path(file_okay=False, path_type=Path),
    default=DEFAULT_INDEX_DIR,
    show_default=True,
    help="Directory containing saved pickle artifacts.",
)
@click.option("--top-k", "-k", default=10, show_default=True, help="Number of results.")
def search_cmd(query: str, index_dir: Path, top_k: int) -> None:
    """Load the saved index and print ranked search results."""
    if not index_dir.is_dir():
        raise click.ClickException(
            f"Index directory {index_dir} does not exist.\n"
            "Run 'tinyir index <folder>' first, or 'tinyir --help' for more information."
        )

    try:
        search_index = InvertedIndex.load(index_dir)
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc
    except (pickle.UnpicklingError, EOFError, KeyError) as exc:
        raise click.ClickException(
            f"Failed to load index artifacts from {index_dir}: {exc}"
        ) from exc

    query_terms = tokenize(query, remove_stopwords=search_index.remove_stopwords)
    if not query_terms:
        click.echo("Query empty after tokenization.")
        return

    results = rank(search_index, query, top_k=top_k)
    if not results:
        click.echo("No results found.")
        return

    click.echo(format_search_table(results))


def main(argv: list[str] | None = None) -> None:
    """CLI entry point. ``--demo`` short-circuits and ignores all other args."""
    args = list(sys.argv[1:] if argv is None else argv)
    if "--demo" in args:
        run_demo()
        return
    tinyir.main(args=args, prog_name="tinyir")

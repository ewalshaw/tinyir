# TinyIR

Command-line information retrieval over plain-text files. Indexes a folder of `.txt` documents and ranks them with tokenization, an inverted index, TF-IDF, and cosine similarity.

## Why TinyIR?

Search engines combine tokenization, inverted indexes, term weighting, and similarity scoring. Full systems like Elasticsearch wrap that in a large API and distributed setup, which makes the basics harder to study.

TinyIR is a small Python pipeline you can run locally: load files, build an index, run queries. No database or web server required. Useful for learning IR or prototyping keyword search on a file corpus.

## Overview

Each `.txt` file in a folder is one document. Indexing:

1. Assigns a `doc_id` and stores filename/path metadata.
2. Tokenizes text (lowercase, strip punctuation, optional stopword removal).
3. Builds an inverted index: `term -> [(doc_id, term_freq), ...]`.
4. Fits `sklearn.feature_extraction.text.TfidfVectorizer` and stores document vectors.
5. Writes pickle artifacts to `.tinyir/` by default.

Search vectorizes the query with the saved model, scores documents with cosine similarity, and prints a table of rank, similarity score, `doc_id`, and filename.

Modules: `corpus`, `tokenize`, `index`, `rank`, `cli`.

## Installation

Python 3.10+.

```bash
git clone https://github.com/ewalshaw/tinyir
cd tinyir
python -m pip install -e .
```

Dependencies: `scikit-learn`, `numpy`, `click`.

## Running the CLI

After install, you can run commands either way:

| Method  | Example                                  |
| ------- | ---------------------------------------- |
| Module  | `python -m tinyir index examples/corpus` |
| Command | `tinyir index examples/corpus`           |


Both accept the same arguments. Use `python -m tinyir` if you prefer not to change `PATH`. Use `tinyir` directly if Python's script directory is already on `PATH`.

`pip` installs the `tinyir` executable into Python's scripts directory. If the command is not found, add that directory to `PATH`.

Verify either method works:

```bash
python -m tinyir --help
tinyir --help
```

## Usage

Examples below use `tinyir`. Replace with `python -m tinyir` if needed.

### Index

```bash
tinyir index examples/corpus
```

Custom output directory:

```bash
tinyir index examples/corpus -o /path/to/my-index
```

Remove common English stopwords during indexing and search:

```bash
tinyir index examples/corpus --remove-stopwords
```

Writes to the index directory:

- `tfidf_model.pkl`: vectorizer and document TF-IDF matrix
- `vocabulary.pkl`: term to column index
- `inverted_index.pkl`: term to `(doc_id, term_freq)` postings
- `documents.pkl`: `doc_id`, `filename`, `path`

Re-run `tinyir index` after upgrading TinyIR if an older index directory is missing `inverted_index.pkl`.

### Search

```bash
tinyir search "machine learning data"
```

Options:

```bash
tinyir search "water plants garden" -i .tinyir -k 5
```

Example output for `tinyir search "machine learning data"`:

```
Rank  Similarity  doc_id  Filename
----  ----------  ------  --------------------
1     0.5413      7       machine_learning.txt
2     0.0591      10      python_lists.txt
3     0.0568      9       public_library.txt
```

### Demo script

```bash
python examples/demo.py
```

Indexes `examples/corpus/` in memory and runs sample queries.

## Tests

```bash
python -m pip install -e ".[dev]"
pytest
```

Covers tokenization, inverted-index postings, TF-IDF shapes, ranking order, and CLI integration.

## Layout

```
tinyir/        library and CLI
examples/
  corpus/      sample documents
  demo.py
tests/
design.md      architecture and data flow
```

## Design

See [design.md](design.md) for module layout, indexing/search flow, and planned extensions.

## License

MIT
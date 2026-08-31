"""Corpus loading utilities for TinyIR."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Document:
    """A single document in the corpus."""

    doc_id: int
    filename: str
    path: Path
    text: str


class Corpus:
    """A collection of text documents loaded from a directory."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory.resolve()
        self.documents: list[Document] = self._load_documents()

    def _load_documents(self) -> list[Document]:
        documents: list[Document] = []
        for doc_id, path in enumerate(sorted(self.directory.glob("*.txt"))):
            documents.append(
                Document(
                    doc_id=doc_id,
                    filename=path.name,
                    path=path.resolve(),
                    text=path.read_text(encoding="utf-8"),
                )
            )
        return documents

    def __len__(self) -> int:
        return len(self.documents)

    def __iter__(self):
        return iter(self.documents)


def load_corpus(directory: Path) -> list[Document]:
    """Load all ``.txt`` files from ``directory`` and assign document IDs."""
    return Corpus(directory).documents

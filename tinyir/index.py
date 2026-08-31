"""Inverted index and TF-IDF model for TinyIR."""

import pickle
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

from .corpus import Corpus, Document
from .tokenize import tokenize

TFIDF_MODEL_FILE = "tfidf_model.pkl"
VOCABULARY_FILE = "vocabulary.pkl"
DOCUMENTS_FILE = "documents.pkl"
INVERTED_INDEX_FILE = "inverted_index.pkl"
INDEX_ARTIFACTS = (
    TFIDF_MODEL_FILE,
    VOCABULARY_FILE,
    DOCUMENTS_FILE,
    INVERTED_INDEX_FILE,
)


class InvertedIndex:
    """Inverted index and TF-IDF vectors for a document corpus."""

    def __init__(self) -> None:
        self.documents: list[Document] = []
        self.inverted_index: dict[str, list[tuple[int, int]]] = {}
        self.vectorizer: TfidfVectorizer | None = None
        self.document_vectors = None
        self.vocabulary: dict[str, int] = {}
        self.remove_stopwords = False

    def build(self, corpus: Corpus, remove_stopwords: bool = False) -> None:
        """Build the inverted index and TF-IDF vectors from ``corpus``."""
        self.remove_stopwords = remove_stopwords
        self.documents = list(corpus.documents)

        tokenized_docs = [
            tokenize(doc.text, remove_stopwords=remove_stopwords)
            for doc in self.documents
        ]
        self.inverted_index = self._build_inverted_index(tokenized_docs)

        pseudo_texts = [" ".join(tokens) for tokens in tokenized_docs]
        self.vectorizer = TfidfVectorizer(analyzer=str.split)
        self.document_vectors = self.vectorizer.fit_transform(pseudo_texts)
        self.vocabulary = dict(self.vectorizer.vocabulary_)

    def _build_inverted_index(
        self,
        tokenized_docs: list[list[str]],
    ) -> dict[str, list[tuple[int, int]]]:
        inverted: dict[str, list[tuple[int, int]]] = defaultdict(list)

        for doc_index, doc in enumerate(self.documents):
            term_freqs = Counter(tokenized_docs[doc_index])
            for term, freq in term_freqs.items():
                inverted[term].append((doc.doc_id, freq))

        return dict(inverted)

    def save(self, directory: Path) -> None:
        """Persist index artifacts to ``directory``."""
        if self.vectorizer is None or self.document_vectors is None:
            raise ValueError("Cannot save an index that has not been built.")

        directory.mkdir(parents=True, exist_ok=True)

        with (directory / TFIDF_MODEL_FILE).open("wb") as model_file:
            pickle.dump(
                {
                    "vectorizer": self.vectorizer,
                    "document_vectors": self.document_vectors,
                    "remove_stopwords": self.remove_stopwords,
                },
                model_file,
            )

        with (directory / VOCABULARY_FILE).open("wb") as vocabulary_file:
            pickle.dump(self.vocabulary, vocabulary_file)

        with (directory / INVERTED_INDEX_FILE).open("wb") as inverted_index_file:
            pickle.dump(self.inverted_index, inverted_index_file)

        document_metadata = [
            {
                "doc_id": doc.doc_id,
                "filename": doc.filename,
                "path": str(doc.path),
            }
            for doc in self.documents
        ]
        with (directory / DOCUMENTS_FILE).open("wb") as documents_file:
            pickle.dump(document_metadata, documents_file)

    @classmethod
    def load(cls, directory: Path) -> "InvertedIndex":
        """Load index artifacts from ``directory``."""
        missing = [name for name in INDEX_ARTIFACTS if not (directory / name).exists()]
        if missing:
            missing_list = ", ".join(missing)
            raise FileNotFoundError(
                f"Index directory {directory} is missing required pickle files: {missing_list}"
            )

        index = cls()

        with (directory / TFIDF_MODEL_FILE).open("rb") as model_file:
            model_payload = pickle.load(model_file)

        with (directory / VOCABULARY_FILE).open("rb") as vocabulary_file:
            index.vocabulary = pickle.load(vocabulary_file)

        with (directory / INVERTED_INDEX_FILE).open("rb") as inverted_index_file:
            index.inverted_index = pickle.load(inverted_index_file)

        with (directory / DOCUMENTS_FILE).open("rb") as documents_file:
            document_metadata = pickle.load(documents_file)

        index.vectorizer = model_payload["vectorizer"]
        index.document_vectors = model_payload["document_vectors"]
        index.remove_stopwords = bool(model_payload.get("remove_stopwords", False))
        index.documents = [
            Document(
                doc_id=item["doc_id"],
                filename=item["filename"],
                path=Path(item["path"]),
                text="",
            )
            for item in document_metadata
        ]
        return index


def build_index(
    corpus: Corpus,
    output_dir: Path,
    *,
    remove_stopwords: bool = False,
) -> InvertedIndex:
    """Build an index from ``corpus`` and save artifacts to ``output_dir``."""
    index = InvertedIndex()
    index.build(corpus, remove_stopwords=remove_stopwords)
    index.save(output_dir)
    return index


def build_index_from_dir(
    corpus_dir: Path,
    output_dir: Path,
    *,
    remove_stopwords: bool = False,
) -> InvertedIndex:
    """Load a corpus from ``corpus_dir``, build an index, and save artifacts."""
    return build_index(
        Corpus(corpus_dir),
        output_dir,
        remove_stopwords=remove_stopwords,
    )

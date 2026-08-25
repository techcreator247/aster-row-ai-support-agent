from pathlib import Path
import pickle
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer


class RAGIndex:
    """TF-IDF index for knowledge-base chunks."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
        )

        self.chunks: list[dict[str, Any]] = []
        self.matrix = None

    def build(self, chunks: list[dict[str, Any]]) -> None:
        """Build the TF-IDF index."""

        if not chunks:
            raise ValueError("Cannot build an index with no chunks.")

        self.chunks = chunks

        texts = [
            self._build_search_text(chunk)
            for chunk in chunks
        ]

        self.matrix = self.vectorizer.fit_transform(texts)

    @staticmethod
    def _build_search_text(chunk: dict[str, Any]) -> str:
        """Combine useful fields for retrieval."""

        metadata = chunk.get("metadata", {})

        title = metadata.get("title", "")
        heading = chunk.get("heading", "")
        content = chunk.get("content", "")

        return f"{title} {heading} {content}"

    def save(self, path: str = "rag_index.pkl") -> None:
        """Save the index to disk."""

        if self.matrix is None:
            raise ValueError("Build the index before saving.")

        data = {
            "vectorizer": self.vectorizer,
            "chunks": self.chunks,
            "matrix": self.matrix,
        }

        Path(path).write_bytes(pickle.dumps(data))

    @classmethod
    def load(cls, path: str = "rag_index.pkl") -> "RAGIndex":
        """Load an existing index."""

        data = pickle.loads(Path(path).read_bytes())

        index = cls()
        index.vectorizer = data["vectorizer"]
        index.chunks = data["chunks"]
        index.matrix = data["matrix"]

        return index
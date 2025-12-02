"""Simple TF‑IDF retriever for demonstration purposes.

The retrieval component of a RAG system is responsible for selecting
documents from a knowledge base that are most relevant to a user
query.  For the purposes of this learning framework we implement a
lightweight TF‑IDF based ranker using scikit‑learn.  While vector
embedding retrievers (e.g. with FAISS or Chroma) often perform
better, TF‑IDF has the benefit of being easy to understand and
requires no external dependencies.

The :class:`SimpleTFIDFRetriever` exposes two methods:

* :meth:`fit` – build the internal index from an iterable of
  documents.
* :meth:`retrieve` – return the top‑k most similar documents to a
  query along with cosine similarity scores.
"""

from typing import Iterable, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class SimpleTFIDFRetriever:
    """A very basic retriever using TF‑IDF.

    The TF‑IDF representation is built from the provided documents and
    stored in memory.  Cosine similarity is used to rank documents
    against a query.
    """

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.doc_matrix = None
        self.documents: List[str] = []

    def fit(self, documents: Iterable[str]) -> None:
        """Fit the TF‑IDF index on the given documents."""
        self.documents = list(documents)
        self.doc_matrix = self.vectorizer.fit_transform(self.documents)

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Return the top_k documents ranked by cosine similarity to the query."""
        if self.doc_matrix is None:
            raise ValueError("Retriever has not been fitted. Call fit() first.")
        query_vec = self.vectorizer.transform([query])
        scores = linear_kernel(query_vec, self.doc_matrix).flatten()
        indices = np.argsort(scores)[::-1][:top_k]
        return [(self.documents[i], float(scores[i])) for i in indices]
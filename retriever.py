"""
retriever.py
-------------
A small vector-database retriever for the Local LLaMA Chatbot.

How it works:
1. Reads text chunks from docs/knowledge.txt (one paragraph per chunk)
2. Converts each chunk into a vector using TF-IDF (no internet/API needed —
   swap this for sentence-transformers embeddings later for better quality,
   see the note at the bottom of this file)
3. Stores those vectors in a FAISS index (a vector database built for
   fast similarity search)
4. Given a user question, finds the most relevant chunk(s) to use as
   context for the LLM

Install once:
    pip install faiss-cpu scikit-learn
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
import os

DOCS_PATH = os.path.join(os.path.dirname(__file__), "docs", "knowledge.txt")


class Retriever:
    def __init__(self, docs_path: str = DOCS_PATH):
        # Load and split the knowledge base into chunks (by blank line)
        with open(docs_path, "r", encoding="utf-8") as f:
            raw = f.read()
        self.chunks = [c.strip() for c in raw.split("\n\n") if c.strip()]

        # Build TF-IDF vectors for all chunks
        self.vectorizer = TfidfVectorizer()
        matrix = self.vectorizer.fit_transform(self.chunks).toarray().astype("float32")
        faiss.normalize_L2(matrix)

        # Build the FAISS index
        self.dimension = matrix.shape[1]
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(matrix)

    def retrieve(self, query: str, top_k: int = 2) -> list[str]:
        """Return the top_k most relevant chunks for a given query.
        If the query shares no vocabulary at all with the knowledge base
        (e.g. "hi", "hello", "thanks"), its TF-IDF vector is all zeros —
        in that case we return no chunks, so the chatbot replies normally
        instead of forcing in irrelevant context."""
        raw_vector = self.vectorizer.transform([query]).toarray().astype("float32")

        if raw_vector.sum() == 0:
            return []   # no shared vocabulary — nothing relevant to retrieve

        query_vector = raw_vector.copy()
        faiss.normalize_L2(query_vector)
        _, indices = self.index.search(query_vector, top_k)
        return [self.chunks[i] for i in indices[0]]


# Quick manual test: run `python retriever.py`
if __name__ == "__main__":
    r = Retriever()
    test_q = "How does the chatbot remember previous messages?"
    print(f"Query: {test_q}")
    for chunk in r.retrieve(test_q):
        print(f" - {chunk}")

# ---------------------------------------------------------------
# UPGRADE PATH (do this later, needs internet on your own machine):
#   pip install sentence-transformers
#   from sentence_transformers import SentenceTransformer
#   model = SentenceTransformer("all-MiniLM-L6-v2")
#   vectors = model.encode(chunks)
# This gives you real semantic embeddings instead of keyword-based
# TF-IDF — better retrieval quality, same FAISS indexing code below.
# ---------------------------------------------------------------
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


class DenseRetriever:
    def __init__(self, documents):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.documents = documents

        # Create embeddings
        self.embeddings = self.model.encode(documents)

        # Build FAISS index
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(self.embeddings))

    def retrieve(self, query, top_k=5):
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(
            np.array(query_embedding), top_k
        )

        results = []

        for idx, dist in zip(indices[0], distances[0]):
            # Convert L2 distance → similarity score
            similarity = 1 / (1 + dist)
            results.append((self.documents[idx], float(similarity)))

        return results
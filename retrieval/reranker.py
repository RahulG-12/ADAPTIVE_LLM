from sentence_transformers import CrossEncoder
import numpy as np


class Reranker:
    def __init__(self):
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def rerank(self, query, documents, top_k=5):
        pairs = [(query, doc) for doc, _ in documents]
        raw_scores = self.model.predict(pairs)

        # Convert logits to probabilities using sigmoid
        scores = 1 / (1 + np.exp(-raw_scores))

        ranked = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [(doc[0], float(score)) for doc, score in ranked[:top_k]]
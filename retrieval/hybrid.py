import time
import numpy as np
from retrieval.dense_retriever import DenseRetriever
from retrieval.sparse_retriever import SparseRetriever
from retrieval.reranker import Reranker


def normalize(scores):
    scores = np.array(scores)

    if scores.max() == scores.min():
        return np.ones_like(scores)

    return (scores - scores.min()) / (scores.max() - scores.min())


class HybridRetriever:
    def __init__(self, documents, alpha=0.5):
        """
        alpha controls weight between:
        dense score = alpha
        sparse score = (1 - alpha)
        """
        self.dense = DenseRetriever(documents)
        self.sparse = SparseRetriever(documents)
        self.reranker = Reranker()
        self.alpha = alpha

    def retrieve(self, query, top_k=5):
        start_time = time.time()

        # Step 1: Retrieve separately
        dense_results = self.dense.retrieve(query, top_k)
        sparse_results = self.sparse.retrieve(query, top_k)

        dense_docs, dense_scores = zip(*dense_results)
        sparse_docs, sparse_scores = zip(*sparse_results)

        # Step 2: Normalize scores
        dense_scores = normalize(dense_scores)
        sparse_scores = normalize(sparse_scores)

        # Step 3: Weighted Fusion
        combined = {}

        for doc, score in zip(dense_docs, dense_scores):
            combined[doc] = self.alpha * score

        for doc, score in zip(sparse_docs, sparse_scores):
            if doc in combined:
                combined[doc] += (1 - self.alpha) * score
            else:
                combined[doc] = (1 - self.alpha) * score

        combined_list = list(combined.items())

        # Step 4: Cross-Encoder Reranking
        reranked = self.reranker.rerank(query, combined_list, top_k)

        total_time = time.time() - start_time
        print(f"Retrieval latency: {total_time:.3f} sec")

        return reranked
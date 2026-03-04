from retrieval.hybrid import HybridRetriever
from fine_tuning.inference import generate
import joblib
import numpy as np


class AdaptiveLLMSystem:
    def __init__(self, documents):
        print("Initializing Hybrid Retrieval...")
        self.retriever = HybridRetriever(documents)

        print("Loading Hallucination Model...")
        self.halluc_model = joblib.load("hallucination/model.pkl")
        self.vectorizer = joblib.load("hallucination/vectorizer.pkl")

    def detect_hallucination(self, response):
        X = self.vectorizer.transform([response])
        prediction = self.halluc_model.predict(X)[0]
        return prediction  # 0 = grounded, 1 = hallucinated

    def run(self, query):
        print("\n🔎 Retrieving context...")
        retrieved_docs = self.retriever.retrieve(query, top_k=3)

        context = "\n".join([doc for doc, _ in retrieved_docs])

        prompt = f"""
Use the following context to answer the question.

Context:
{context}

Question:
{query}

Answer:
"""

        print("🤖 Generating answer...")
        response = generate(prompt)

        print("🧠 Checking hallucination...")
        halluc_flag = self.detect_hallucination(response)

        return {
            "query": query,
            "response": response,
            "hallucination": halluc_flag,
            "retrieved_docs": retrieved_docs
        }
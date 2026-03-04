from integrated_system import AdaptiveLLMSystem

def load_documents():
    return [
        "Retrieval-Augmented Generation (RAG) combines retrieval systems with large language models.",
        "BM25 is a ranking function used in information retrieval.",
        "Vector embeddings convert text into numerical representations.",
        "Cross-encoders score query-document pairs jointly."
    ]

def main():
    documents = load_documents()
    system = AdaptiveLLMSystem(documents)

    while True:
        query = input("\nEnter your question (or type exit): ")
        if query.lower() == "exit":
            break

        result = system.run(query)

        print("\n==============================")
        print("📌 FINAL RESPONSE:\n")
        print(result["response"])

        if result["hallucination"] == 1:
            print("\n⚠️ WARNING: Potential Hallucination Detected!")
        else:
            print("\n✅ Response appears grounded.")

        print("\n==============================")

if __name__ == "__main__":
    main()
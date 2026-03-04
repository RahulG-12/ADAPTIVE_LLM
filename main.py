print("Program started")

from retrieval.hybrid import HybridRetriever

def load_documents(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]

if __name__ == "__main__":
    print("Loading documents...")
    documents = load_documents("data/documents.txt")
    retriever = HybridRetriever(documents)

    print("Ready for query")
    query = input("Enter query: ")
    results = retriever.retrieve(query)

    print("\nTop Results:\n")
    for doc, score in results:
        print(f"Score: {score:.4f}")
        print(doc)
        print("-" * 50)

    input("Press Enter to exit...")
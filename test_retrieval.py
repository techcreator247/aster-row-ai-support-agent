from app.rag.index import RAGIndex
from app.rag.retriever import Retriever


index = RAGIndex.load("rag_index.pkl")
retriever = Retriever(index)

query = "What is the return window?"

results = retriever.search(query, top_k=5)

print("\nQUERY:", query)
print("=" * 70)

for result in results:
    metadata = result["metadata"]

    print(f"\nScore: {result['score']}")
    print(f"File: {result['filename']}")
    print(f"Heading: {result['heading']}")
    print(f"Status: {metadata.get('status')}")
    print(f"Authority: {metadata.get('policy_authority')}")
    print(f"Audience: {metadata.get('audience')}")
    print(f"Content:\n{result['content'][:500]}")
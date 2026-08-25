from app.rag.parser import DocumentParser
from app.rag.chunker import DocumentChunker
from app.rag.index import RAGIndex


def main():
    print("Parsing knowledge base...")

    parser = DocumentParser("knowledge-base")
    documents = parser.parse_all()

    print(f"Documents found: {len(documents)}")

    print("Chunking documents...")

    chunker = DocumentChunker()
    chunks = chunker.chunk_documents(documents)

    print(f"Chunks created: {len(chunks)}")

    print("Building TF-IDF index...")

    index = RAGIndex()
    index.build(chunks)

    index.save("rag_index.pkl")

    print("Index saved to rag_index.pkl")


if __name__ == "__main__":
    main()
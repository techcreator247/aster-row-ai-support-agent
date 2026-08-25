from app.rag.parser import DocumentParser


parser = DocumentParser("knowledge-base")

documents = parser.parse_all()

print(f"\nFound {len(documents)} documents\n")

for document in documents:
    print("=" * 60)
    print("FILE:", document["filename"])
    print("METADATA:", document["metadata"])
    print("HEADINGS:", document["headings"])
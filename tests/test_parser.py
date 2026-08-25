from app.rag.parser import DocumentParser


def test_parse_knowledge_base():
    parser = DocumentParser("knowledge-base")

    documents = parser.parse_all()

    assert len(documents) > 0

    for document in documents:
        assert "filename" in document
        assert "metadata" in document
        assert "headings" in document
        assert "content" in document

        assert document["filename"].endswith(".md")
        assert isinstance(document["metadata"], dict)
        assert isinstance(document["headings"], list)
        assert isinstance(document["content"], str)
from app.rag.parser import DocumentParser
from app.rag.chunker import DocumentChunker


def test_documents_are_chunked():
    parser = DocumentParser("knowledge-base")
    documents = parser.parse_all()

    chunker = DocumentChunker()
    chunks = chunker.chunk_documents(documents)

    assert len(chunks) > 0

    for chunk in chunks:
        assert "filename" in chunk
        assert "heading" in chunk
        assert "content" in chunk
        assert "metadata" in chunk

        assert chunk["filename"].endswith(".md")
        assert chunk["heading"]
        assert chunk["content"]
        assert isinstance(chunk["metadata"], dict)
from typing import Any


class DocumentChunker:
    """Split parsed Markdown documents into useful heading-based chunks."""

    def chunk_document(self, document: dict[str, Any]) -> list[dict[str, Any]]:
        content = document["content"]
        filename = document["filename"]
        metadata = document["metadata"]

        lines = content.splitlines()

        chunks = []
        current_heading = None
        current_content = []

        for line in lines:
            stripped = line.strip()

            # Detect Markdown headings
            if stripped.startswith("#"):
                # Save previous section only if it has actual content
                self._save_chunk(
                    chunks=chunks,
                    filename=filename,
                    metadata=metadata,
                    heading=current_heading,
                    content=current_content,
                )

                current_heading = stripped.lstrip("#").strip()
                current_content = []

            else:
                current_content.append(line)

        # Save the final section
        self._save_chunk(
            chunks=chunks,
            filename=filename,
            metadata=metadata,
            heading=current_heading,
            content=current_content,
        )

        return chunks

    def chunk_documents(
        self,
        documents: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Chunk all parsed documents."""

        all_chunks = []

        for document in documents:
            all_chunks.extend(self.chunk_document(document))

        return all_chunks

    @staticmethod
    def _save_chunk(
        chunks: list[dict[str, Any]],
        filename: str,
        metadata: dict[str, Any],
        heading: str | None,
        content: list[str],
    ) -> None:
        """Save a chunk only when heading and content are present."""

        if not heading:
            return

        cleaned_content = "\n".join(content).strip()

        if not cleaned_content:
            return

        chunks.append(
            {
                "filename": filename,
                "heading": heading,
                "content": cleaned_content,
                "metadata": metadata.copy(),
            }
        )
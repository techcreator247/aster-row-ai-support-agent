from pathlib import Path
from typing import Any
import re
import yaml


class DocumentParser:
    """Parse Markdown files from the Aster & Row knowledge base."""

    def __init__(self, knowledge_base_path: str = "knowledge-base"):
        self.knowledge_base_path = Path(knowledge_base_path)

    def parse_file(self, file_path: Path) -> dict[str, Any]:
        """Parse a single Markdown document."""

        text = file_path.read_text(encoding="utf-8")

        metadata, content = self._extract_front_matter(text)
        headings = self._extract_headings(content)

        return {
            "filename": file_path.name,
            "metadata": metadata,
            "headings": headings,
            "content": content.strip(),
        }

    def parse_all(self) -> list[dict[str, Any]]:
        """Parse all Markdown files in the knowledge base."""

        if not self.knowledge_base_path.exists():
            raise FileNotFoundError(
                f"Knowledge base not found: {self.knowledge_base_path}"
            )

        documents = []

        for file_path in sorted(self.knowledge_base_path.glob("*.md")):
            document = self.parse_file(file_path)
            documents.append(document)

        return documents

    @staticmethod
    def _extract_front_matter(
        text: str,
    ) -> tuple[dict[str, Any], str]:
        """Extract YAML front matter from a Markdown document."""

        pattern = r"^---\s*\n(.*?)\n---\s*\n(.*)$"

        match = re.match(pattern, text, re.DOTALL)

        if not match:
            return {}, text

        front_matter_text = match.group(1)
        content = match.group(2)

        metadata = yaml.safe_load(front_matter_text) or {}

        if not isinstance(metadata, dict):
            metadata = {}

        return metadata, content

    @staticmethod
    def _extract_headings(content: str) -> list[str]:
        """Extract Markdown headings."""

        headings = []

        for line in content.splitlines():
            match = re.match(r"^(#{1,6})\s+(.+)$", line.strip())

            if match:
                headings.append(match.group(2).strip())

        return headings
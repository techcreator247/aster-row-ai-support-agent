from typing import Any

from sklearn.metrics.pairwise import cosine_similarity

from app.rag.index import RAGIndex


class Retriever:
    """Retrieve relevant knowledge-base chunks safely."""

    def __init__(self, index: RAGIndex):
        self.index = index

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Retrieve relevant knowledge-base chunks.

        Relevance is the primary ranking signal, while document
        authority provides a safety-aware secondary signal.

        Superseded documents are excluded from the final results
        because they must never be used as current customer policy.
        """

        if not query.strip():
            return []

        if self.index.matrix is None:
            raise ValueError("RAG index has not been built.")

        # Add lightweight retrieval vocabulary for common
        # customer-language queries.
        search_query = self._expand_query(query)

        query_vector = self.index.vectorizer.transform(
            [search_query]
        )

        similarities = cosine_similarity(
            query_vector,
            self.index.matrix,
        )[0]

        candidates = []

        for index, similarity in enumerate(similarities):
            score = float(similarity)

            if score <= 0:
                continue

            chunk = self.index.chunks[index].copy()

            metadata = chunk.get("metadata", {})

            status = str(
                metadata.get("status", "")
            ).lower()

            # Superseded documents are historical information.
            # They should not enter the final customer-answer
            # retrieval set.
            if status == "superseded":
                continue

            chunk["score"] = round(score, 4)

            chunk["precedence_score"] = (
                self._precedence_score(metadata)
            )

            chunk["ranking_score"] = round(
                score
                + self._authority_bonus(
                    chunk["precedence_score"]
                ),
                6,
            )

            candidates.append(chunk)

        # Relevance first.
        # Authority acts as a secondary safety signal.
        candidates.sort(
            key=lambda item: (
                item["ranking_score"],
                item["score"],
                item["precedence_score"],
            ),
            reverse=True,
        )

        return candidates[:top_k]

    @staticmethod
    def _expand_query(query: str) -> str:
        """
        Add lightweight retrieval hints for common customer
        terminology.

        This only affects retrieval. It does not modify the
        user's actual question.
        """

        normalized = query.lower()

        expansions: list[str] = []

        # ---------------------------------------------------------
        # International shipping
        # ---------------------------------------------------------

        shipping_terms = (
            "ship",
            "shipping",
            "shipment",
            "deliver",
            "delivery",
            "country",
            "international",
        )

        country_terms = (
            "germany",
            "canada",
            "france",
            "india",
            "australia",
            "uk",
            "united kingdom",
            "japan",
            "mexico",
        )

        if (
            any(
                term in normalized
                for term in shipping_terms
            )
            and any(
                term in normalized
                for term in country_terms
            )
        ):
            expansions.extend(
                [
                    "international shipping",
                    "supported countries",
                    "shipping destinations",
                    "country availability",
                ]
            )

        if "international" in normalized:
            expansions.extend(
                [
                    "international shipping",
                    "supported countries",
                    "shipping destinations",
                ]
            )

        # ---------------------------------------------------------
        # Returns
        # ---------------------------------------------------------

        if (
            "return" in normalized
            or "send back" in normalized
            or "refund" in normalized
        ):
            expansions.extend(
                [
                    "return policy",
                    "return window",
                    "calendar days",
                    "delivery",
                ]
            )

        # ---------------------------------------------------------
        # Warranty
        # ---------------------------------------------------------

        if "warranty" in normalized:
            expansions.extend(
                [
                    "warranty coverage",
                    "bags",
                    "drinkware",
                    "travel accessories",
                ]
            )

        # ---------------------------------------------------------
        # Product care
        # ---------------------------------------------------------

        if (
            "dishwasher" in normalized
            or "wash" in normalized
            or "care" in normalized
        ):
            expansions.extend(
                [
                    "product care",
                    "dishwasher safe",
                    "hand wash",
                ]
            )

        if not expansions:
            return query

        return query + " " + " ".join(expansions)

    @staticmethod
    def _authority_bonus(
        precedence_score: int,
    ) -> float:
        """
        Convert document precedence into a modest ranking bonus.

        Relevance remains the dominant signal.
        """

        if precedence_score >= 140:
            return 0.15

        if precedence_score >= 100:
            return 0.10

        if precedence_score >= 50:
            return 0.05

        if precedence_score >= 0:
            return 0.0

        if precedence_score <= -100:
            return -0.15

        return -0.05

    @staticmethod
    def _precedence_score(
        metadata: dict[str, Any],
    ) -> int:
        """
        Rank documents according to their authority.

        Higher score = more appropriate for customer answers.
        """

        status = str(
            metadata.get("status", "")
        ).lower()

        authority = str(
            metadata.get("policy_authority", "")
        ).lower()

        audience = str(
            metadata.get("audience", "")
        ).lower()

        customer_answering = metadata.get(
            "customer_answering",
            True,
        )

        score = 0

        # Active documents are preferred.
        if status == "active":
            score += 100

        # Superseded documents are strongly deprioritized.
        elif status == "superseded":
            score -= 100

        # Draft documents should not outrank active policies.
        elif status == "draft":
            score -= 150

        # Official sources are preferred.
        if authority == "official":
            score += 30

        elif authority == "none":
            score -= 50

        # Customer-facing documents are preferred.
        if audience == "customer":
            score += 20

        elif audience == "internal":
            score -= 20

        # Documents explicitly marked as unsuitable for
        # customer answering receive a strong penalty.
        if customer_answering is False:
            score -= 100

        return score
from app.rag.index import RAGIndex
from app.rag.retriever import Retriever


def create_retriever():
    index = RAGIndex.load("rag_index.pkl")
    return Retriever(index)


def test_superseded_return_policy_is_not_top_result():
    retriever = create_retriever()

    results = retriever.search(
        "What is the return window?",
        top_k=5,
    )

    assert results

    top_filenames = [
        result["filename"]
        for result in results
    ]

    # Legacy policy must not be the first result.
    assert top_filenames[0] != "02-returns-policy-legacy.md"


def test_current_return_policy_is_retrieved():
    retriever = create_retriever()

    results = retriever.search(
        "What is the standard return window?",
        top_k=5,
    )

    filenames = [
        result["filename"]
        for result in results
    ]

    assert "01-returns-policy-current.md" in filenames


def test_legacy_policy_is_deprioritized():
    retriever = create_retriever()

    results = retriever.search(
        "What is the return window?",
        top_k=5,
    )

    legacy_results = [
        result
        for result in results
        if result["filename"]
        == "02-returns-policy-legacy.md"
    ]

    if legacy_results:
        assert legacy_results[0]["precedence_score"] < 0
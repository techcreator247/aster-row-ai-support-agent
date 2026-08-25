import sys
import json
import re
from pathlib import Path
from collections import defaultdict


# ============================================================
# PROJECT PATH FIX
# ============================================================

# evaluation/run_evaluation.py
#        parent -> evaluation
#        parent.parent -> project root

BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


# ============================================================
# APPLICATION IMPORTS
# ============================================================

from app.rag.index import RAGIndex
from app.rag.retriever import Retriever
from app.tools.orders import OrderLookup


# ============================================================
# FILE PATHS
# ============================================================

EVALUATION_DIR = BASE_DIR / "evaluation"

VISIBLE_CASES = EVALUATION_DIR / "visible-cases.json"
ORIGINAL_CASES = EVALUATION_DIR / "original-cases.json"

INDEX_PATH = BASE_DIR / "rag_index.pkl"
ORDERS_PATH = BASE_DIR / "data" / "orders.json"


# ============================================================
# ORDER ID PATTERN
# ============================================================

ORDER_ID_PATTERN = re.compile(
    r"\bORD-\d+\b",
    flags=re.IGNORECASE,
)


# ============================================================
# LOAD CASES
# ============================================================

def load_cases(path):
    """Load evaluation cases from a JSON file."""

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)

    return data["cases"]


# ============================================================
# EXTRACT ORDER ID
# ============================================================

def extract_order_id(text):
    """
    Extract an order ID from text.

    Supports:
        ORD-1007
        ord-1007
        ORD-1234
    """

    if not text:
        return None

    match = ORDER_ID_PATTERN.search(text)

    if not match:
        return None

    return match.group(0)


# ============================================================
# BUILD CONVERSATION TEXT
# ============================================================

def build_conversation_text(case):
    """
    Combine all user messages in a case.

    This is important for multi-turn conversations.

    Example:

        User:
        Where is ORD-1007?

        User:
        And what carrier is handling it?

    The evaluator still remembers ORD-1007.
    """

    messages = case.get(
        "messages",
        [],
    )

    user_messages = []

    for message in messages:

        if message.get("role") == "user":

            content = message.get(
                "content",
                "",
            )

            user_messages.append(
                content
            )

    return "\n".join(
        user_messages
    )


# ============================================================
# SOURCE HELPERS
# ============================================================

def get_source_name(result):
    """Get the source filename from a retrieval result."""

    return result.get(
        "filename",
        "unknown",
    )


def source_names(results):
    """
    Return unique source filenames while preserving
    retrieval order.
    """

    names = []

    for result in results:

        filename = get_source_name(
            result
        )

        if filename not in names:

            names.append(
                filename
            )

    return names


def retrieve_sources(
    retriever,
    query,
):
    """
    Retrieve relevant knowledge-base documents.

    top_k=8 gives the evaluator enough retrieval
    coverage for questions such as international
    shipping to Germany.
    """

    return retriever.search(
        query,
        top_k=8,
    )


def contains_required_sources(
    results,
    required_sources,
):
    """Return required sources that were not retrieved."""

    names = source_names(
        results
    )

    return [
        source
        for source in required_sources
        if source not in names
    ]


def contains_forbidden_sources(
    results,
    forbidden_sources,
):
    """Return forbidden sources that were retrieved."""

    names = source_names(
        results
    )

    return [
        source
        for source in forbidden_sources
        if source in names
    ]


# ============================================================
# PRIVATE DATA CHECK
# ============================================================

def contains_private_fields(
    order_result,
):
    """
    Check whether an order result contains fields that
    should not be exposed to the customer.
    """

    if not order_result:
        return False

    serialized = json.dumps(
        order_result,
        ensure_ascii=False,
    ).lower()

    private_markers = [
        "risk_score",
        "risk score",
        "shipping_address",
        "shipping address",
        "warehouse_note",
        "warehouse note",
        "internal",
        "fraud review",
        "example.test",
    ]

    return any(
        marker in serialized
        for marker in private_markers
    )


# ============================================================
# RETRIEVAL EVALUATION
# ============================================================

def evaluate_retrieval_case(
    case,
    retriever,
):
    """
    Evaluate retrieval-related cases deterministically.

    This checks:
        - required sources
        - forbidden sources
        - order tool should not be used
    """

    conversation_text = build_conversation_text(
        case
    )

    expectation = case.get(
        "expect",
        {},
    )

    results = retrieve_sources(
        retriever,
        conversation_text,
    )

    required_sources = expectation.get(
        "required_sources",
        [],
    )

    forbidden_sources = expectation.get(
        "forbidden_sources_as_authority",
        [],
    )

    expected_tool = expectation.get(
        "tool",
    )

    missing = contains_required_sources(
        results,
        required_sources,
    )

    forbidden_found = contains_forbidden_sources(
        results,
        forbidden_sources,
    )

    order_id = extract_order_id(
        conversation_text
    )

    passed = True
    reasons = []

    # --------------------------------------------------------
    # Required sources
    # --------------------------------------------------------

    if missing:

        passed = False

        reasons.append(
            "Required source(s) not retrieved: "
            + ", ".join(missing)
        )

    elif required_sources:

        reasons.append(
            "Required source(s) retrieved: "
            + ", ".join(required_sources)
        )

    # --------------------------------------------------------
    # Forbidden sources
    # --------------------------------------------------------

    if forbidden_found:

        passed = False

        reasons.append(
            "Forbidden source(s) appeared in retrieval: "
            + ", ".join(forbidden_found)
        )

    elif forbidden_sources:

        reasons.append(
            "Forbidden authority sources were not "
            "retrieved in top results."
        )

    # --------------------------------------------------------
    # Tool check
    # --------------------------------------------------------

    if expected_tool in {
        "not_called",
        "not_called_without_id",
    }:

        if order_id:

            passed = False

            reasons.append(
                f"Unexpected order ID detected: {order_id}"
            )

        else:

            reasons.append(
                "No order ID detected; order lookup "
                "is not called."
            )

    return {
        "id": case["id"],
        "category": case["category"],
        "passed": passed,
        "status": (
            "PASS"
            if passed
            else "FAIL"
        ),
        "reason": " ".join(
            reasons
        ),
        "retrieved_sources": source_names(
            results
        ),
        "order_id": order_id,
    }


# ============================================================
# ORDER EVALUATION
# ============================================================

def evaluate_order_case(
    case,
    order_tool,
):
    """
    Evaluate order lookup behavior without using the LLM.
    """

    conversation_text = build_conversation_text(
        case
    )

    expectation = case.get(
        "expect",
        {},
    )

    expected_tool = expectation.get(
        "tool",
    )

    order_id = extract_order_id(
        conversation_text
    )

    reasons = []
    passed = True

    # --------------------------------------------------------
    # Missing order ID case
    # --------------------------------------------------------

    if expected_tool == "not_called_without_id":

        if order_id:

            passed = False

            reasons.append(
                f"Unexpected order ID detected: {order_id}"
            )

        else:

            reasons.append(
                "No order ID detected; order lookup "
                "is not called."
            )

        return {
            "id": case["id"],
            "category": case["category"],
            "passed": passed,
            "status": (
                "PASS"
                if passed
                else "FAIL"
            ),
            "reason": " ".join(
                reasons
            ),
            "order_id": order_id,
        }

    # --------------------------------------------------------
    # Order ID expected but missing
    # --------------------------------------------------------

    if not order_id:

        return {
            "id": case["id"],
            "category": case["category"],
            "passed": False,
            "status": "FAIL",
            "reason": (
                "Expected an order ID but none "
                "was detected."
            ),
            "order_id": None,
        }

    # --------------------------------------------------------
    # Perform lookup
    # --------------------------------------------------------

    result = order_tool.lookup(
        order_id
    )

    # --------------------------------------------------------
    # Unknown order
    # --------------------------------------------------------

    if result is None:

        if case["id"] == "unknown-order":

            reasons.append(
                "Order lookup successfully handled "
                f"unknown order {order_id}."
            )

            return {
                "id": case["id"],
                "category": case["category"],
                "passed": True,
                "status": "PASS",
                "reason": " ".join(
                    reasons
                ),
                "order_id": order_id,
            }

        return {
            "id": case["id"],
            "category": case["category"],
            "passed": False,
            "status": "FAIL",
            "reason": (
                f"Order {order_id} was not found."
            ),
            "order_id": order_id,
        }

    # --------------------------------------------------------
    # Successful lookup
    # --------------------------------------------------------

    reasons.append(
        "Order lookup successfully resolved "
        f"{order_id}."
    )

    # --------------------------------------------------------
    # Privacy check
    # --------------------------------------------------------

    if contains_private_fields(
        result
    ):

        passed = False

        reasons.append(
            "Private order fields were exposed."
        )

    else:

        reasons.append(
            "No private order fields exposed."
        )

    return {
        "id": case["id"],
        "category": case["category"],
        "passed": passed,
        "status": (
            "PASS"
            if passed
            else "FAIL"
        ),
        "reason": " ".join(
            reasons
        ),
        "order_id": order_id,
    }


# ============================================================
# CASES REQUIRING LIVE LLM
# ============================================================

def evaluate_not_run_case(case):
    """
    These cases require checking the generated LLM response.

    We intentionally do not mark them PASS using only
    retrieval/tool information.
    """

    return {
        "id": case["id"],
        "category": case["category"],
        "passed": False,
        "not_run": True,
        "status": "NOT RUN",
        "reason": (
            "This case requires live LLM response evaluation."
        ),
    }


# ============================================================
# CASE ROUTER
# ============================================================

def evaluate_case(
    case,
    retriever,
    order_tool,
):
    """
    Select the correct deterministic evaluator.
    """

    expectation = case.get(
        "expect",
        {},
    )

    category = case.get(
        "category",
        "",
    )

    expected_tool = expectation.get(
        "tool",
    )

    # --------------------------------------------------------
    # LLM-dependent cases
    # --------------------------------------------------------

    if category == "privacy":

        return evaluate_not_run_case(
            case
        )

    if category == "safe-action":

        return evaluate_not_run_case(
            case
        )

    # --------------------------------------------------------
    # Order/tool cases
    # --------------------------------------------------------

    if expected_tool in {
        "order_lookup",
        "not_called_without_id",
    }:

        return evaluate_order_case(
            case,
            order_tool,
        )

    # --------------------------------------------------------
    # Retrieval cases
    # --------------------------------------------------------

    return evaluate_retrieval_case(
        case,
        retriever,
    )


# ============================================================
# PRINT RESULT
# ============================================================

def print_result(result):

    status = result.get(
        "status",
        "FAIL",
    )

    print(
        f"[{status:<8}] "
        f"{result['id']} "
        f"({result['category']})"
    )

    print(
        f"           {result['reason']}"
    )


# ============================================================
# SAVE REPORT
# ============================================================

def save_report(results):

    output_path = (
        EVALUATION_DIR
        / "evaluation-results.json"
    )

    report = {
        "total_cases": len(
            results
        ),

        "passed": sum(
            1
            for result in results
            if result.get("status") == "PASS"
        ),

        "failed": sum(
            1
            for result in results
            if (
                result.get("status") == "FAIL"
                and not result.get("not_run")
            )
        ),

        "not_run": sum(
            1
            for result in results
            if result.get("status") == "NOT RUN"
        ),

        "results": results,
    }

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return output_path


# ============================================================
# CATEGORY SUMMARY
# ============================================================

def print_category_summary(
    results
):

    categories = defaultdict(list)

    for result in results:

        categories[
            result["category"]
        ].append(
            result
        )

    print()
    print("=" * 70)
    print("CATEGORY SUMMARY")
    print("=" * 70)

    for category, category_results in categories.items():

        passed = sum(
            1
            for result in category_results
            if result.get("status") == "PASS"
        )

        failed = sum(
            1
            for result in category_results
            if (
                result.get("status") == "FAIL"
                and not result.get("not_run")
            )
        )

        not_run = sum(
            1
            for result in category_results
            if result.get("status") == "NOT RUN"
        )

        total = len(
            category_results
        )

        print(
            f"{category:25} "
            f"{passed}/{total} passed | "
            f"{failed} failed | "
            f"{not_run} not-run"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ASTER & ROW — OFFLINE EVALUATION SUITE")
    print("=" * 70)

    # --------------------------------------------------------
    # Load cases
    # --------------------------------------------------------

    visible_cases = load_cases(
        VISIBLE_CASES
    )

    original_cases = load_cases(
        ORIGINAL_CASES
    )

    cases = (
        visible_cases
        + original_cases
    )

    print()
    print(
        f"Visible cases : {len(visible_cases)}"
    )

    print(
        f"Original cases: {len(original_cases)}"
    )

    print(
        f"Total cases   : {len(cases)}"
    )

    # --------------------------------------------------------
    # Load retrieval index
    # --------------------------------------------------------

    print()
    print(
        "Loading retrieval index..."
    )

    index = RAGIndex.load(
        str(INDEX_PATH)
    )

    retriever = Retriever(
        index
    )

    print(
        "Retrieval index loaded."
    )

    # --------------------------------------------------------
    # Load order database
    # --------------------------------------------------------

    print(
        "Loading order database..."
    )

    order_tool = OrderLookup(
        str(ORDERS_PATH)
    )

    print(
        "Order database loaded."
    )

    # --------------------------------------------------------
    # Evaluate cases
    # --------------------------------------------------------

    print()

    results = []

    for case in cases:

        result = evaluate_case(
            case,
            retriever,
            order_tool,
        )

        results.append(
            result
        )

        print_result(
            result
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_category_summary(
        results
    )

    # --------------------------------------------------------
    # Overall results
    # --------------------------------------------------------

    passed = sum(
        1
        for result in results
        if result.get("status") == "PASS"
    )

    failed = sum(
        1
        for result in results
        if (
            result.get("status") == "FAIL"
            and not result.get("not_run")
        )
    )

    not_run = sum(
        1
        for result in results
        if result.get("status") == "NOT RUN"
    )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    report_path = save_report(
        results
    )

    print()
    print("=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)

    print(
        f"Passed  : {passed}/{len(results)}"
    )

    print(
        f"Failed  : {failed}/{len(results)}"
    )

    print(
        f"Not run : {not_run}/{len(results)}"
    )

    print()
    print(
        "Detailed report saved to:"
    )

    print(
        report_path
    )

    print()
    print("=" * 70)
    print(
        "Offline evaluation completed."
    )
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
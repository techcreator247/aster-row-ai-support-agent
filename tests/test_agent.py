from app.agent import SupportAgent


def create_agent():
    """
    Create an agent without making an OpenAI API request.
    """
    agent = object.__new__(SupportAgent)

    from app.rag.index import RAGIndex
    from app.rag.retriever import Retriever
    from app.memory.session import SessionMemory
    from app.tools.orders import OrderLookup

    agent.index = RAGIndex.load("rag_index.pkl")
    agent.retriever = Retriever(agent.index)
    agent.orders = OrderLookup("data/orders.json")
    agent.memory = SessionMemory()

    return agent


def test_order_id_is_detected():
    agent = create_agent()

    result = agent._handle_order_lookup(
        "Where is ORD-1007?"
    )

    assert result is not None
    assert result["success"] is True


def test_lowercase_order_id_is_detected():
    agent = create_agent()

    result = agent._handle_order_lookup(
        "where is ord-1007?"
    )

    assert result is not None
    assert result["success"] is True


def test_no_order_id_means_no_lookup():
    agent = create_agent()

    result = agent._handle_order_lookup(
        "What is your return policy?"
    )

    assert result is None


def test_order_result_does_not_expose_private_fields():
    agent = create_agent()

    result = agent._handle_order_lookup(
        "Where is ORD-1007?"
    )

    assert result is not None
    assert result["success"] is True

    order = result["order"]

    assert "customer_email" not in order
    assert "address" not in order
    assert "internal_notes" not in order
    assert "risk_score" not in order


def test_context_contains_source_information():
    agent = create_agent()

    results = agent.retriever.search(
        "What is the return window?",
        top_k=3,
    )

    context = agent._build_context(
        results,
        None,
    )

    assert "KNOWLEDGE BASE RESULTS" in context
    assert ".md" in context
    assert "Source:" in context


def test_context_contains_order_result():
    agent = create_agent()

    order_result = agent._handle_order_lookup(
        "Where is ORD-1007?"
    )

    context = agent._build_context(
        [],
        order_result,
    )

    assert "ORDER TOOL RESULT" in context
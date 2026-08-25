from app.tools.orders import OrderLookup


def get_lookup():
    return OrderLookup("data/orders.json")


def test_missing_order_id():
    lookup = get_lookup()

    result = lookup.lookup("")

    assert result["success"] is False
    assert result["error"] == "missing_order_id"


def test_invalid_order_id():
    lookup = get_lookup()

    result = lookup.lookup("hello")

    assert result["success"] is False
    assert result["error"] == "invalid_order_id"


def test_unknown_order_id():
    lookup = get_lookup()

    result = lookup.lookup("ORD-999999")

    assert result["success"] is False
    assert result["error"] == "order_not_found"


def test_order_id_is_case_and_whitespace_insensitive():
    lookup = get_lookup()

    # This test assumes ORD-1007 exists in the supplied dataset.
    result = lookup.lookup("  ord-1007  ")

    assert result["success"] is True
    assert result["order"]["order_id"] == "ORD-1007"


def test_internal_fields_are_not_exposed():
    lookup = get_lookup()

    result = lookup.lookup("ORD-1007")

    assert result["success"] is True

    order = result["order"]

    assert "customer_email" not in order
    assert "email" not in order
    assert "address" not in order
    assert "internal_notes" not in order
    assert "risk_score" not in order
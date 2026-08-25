import json
from pathlib import Path
from typing import Any


class OrderLookup:
    """Safe lookup service for mock order data."""

    def __init__(self, orders_path: str = "data/orders.json"):
        self.orders_path = Path(orders_path)
        self.orders = self._load_orders()

    def _load_orders(self) -> Any:
        """Load orders from the JSON file."""

        if not self.orders_path.exists():
            raise FileNotFoundError(
                f"Orders file not found: {self.orders_path}"
            )

        with self.orders_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def lookup(self, order_id: str) -> dict[str, Any]:
        """
        Look up an order and return only customer-safe information.
        """

        if not order_id or not order_id.strip():
            return {
                "success": False,
                "error": "missing_order_id",
                "message": "Please provide your order ID.",
            }

        normalized_id = order_id.strip().upper()

        if not normalized_id.startswith("ORD-"):
            return {
                "success": False,
                "error": "invalid_order_id",
                "message": "Please provide a valid order ID such as ORD-1007.",
            }

        order = self._find_order(normalized_id)

        if order is None:
            return {
                "success": False,
                "error": "order_not_found",
                "message": "I could not find an order with that ID.",
            }

        return {
            "success": True,
            "order": self._sanitize_order(order),
        }

    def _find_order(self, order_id: str) -> dict[str, Any] | None:
        """Find an order by normalized order ID."""

        if isinstance(self.orders, list):
            for order in self.orders:
                if str(order.get("order_id", "")).strip().upper() == order_id:
                    return order

        elif isinstance(self.orders, dict):
            # Handle either:
            # {"ORD-1007": {...}}
            # or {"orders": [{...}]}
            if order_id in self.orders:
                return self.orders[order_id]

            orders = self.orders.get("orders", [])

            if isinstance(orders, list):
                for order in orders:
                    if (
                        str(order.get("order_id", "")).strip().upper()
                        == order_id
                    ):
                        return order

        return None

    @staticmethod
    def _sanitize_order(order: dict[str, Any]) -> dict[str, Any]:
        """
        Return only fields that are safe to expose to a customer.
        """

        safe_fields = [
            "order_id",
            "status",
            "items",
            "estimated_delivery",
            "tracking_number",
        ]

        result = {}

        for field in safe_fields:
            if field in order and order[field] is not None:
                result[field] = order[field]

        # Never show stale delivery information for cancelled/returned orders.
        status = str(result.get("status", "")).lower()

        if status in {"cancelled", "canceled", "returned"}:
            result.pop("estimated_delivery", None)
            result.pop("tracking_number", None)

        return result
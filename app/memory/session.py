from typing import Any


class SessionMemory:
    """Simple in-memory conversation history."""

    def __init__(self):
        self.sessions: dict[str, list[dict[str, Any]]] = {}

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
    ) -> None:
        """Add a message to a session."""

        if session_id not in self.sessions:
            self.sessions[session_id] = []

        self.sessions[session_id].append(
            {
                "role": role,
                "content": content,
            }
        )

    def get_history(
        self,
        session_id: str,
    ) -> list[dict[str, Any]]:
        """Return conversation history for a session."""

        return self.sessions.get(session_id, []).copy()

    def clear_session(
        self,
        session_id: str,
    ) -> None:
        """Clear one conversation session."""

        self.sessions.pop(session_id, None)

    def get_recent_history(
        self,
        session_id: str,
        max_messages: int = 10,
    ) -> list[dict[str, Any]]:
        """Return only the most recent messages."""

        history = self.sessions.get(session_id, [])

        return history[-max_messages:]
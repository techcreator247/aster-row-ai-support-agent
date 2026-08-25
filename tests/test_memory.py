from app.memory.session import SessionMemory


def test_new_session_is_empty():
    memory = SessionMemory()

    assert memory.get_history("session-1") == []


def test_messages_are_saved():
    memory = SessionMemory()

    memory.add_message(
        "session-1",
        "user",
        "Do you ship internationally?",
    )

    memory.add_message(
        "session-1",
        "assistant",
        "Yes, international shipping is available.",
    )

    history = memory.get_history("session-1")

    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"


def test_sessions_are_isolated():
    memory = SessionMemory()

    memory.add_message(
        "session-1",
        "user",
        "Where is my order?",
    )

    memory.add_message(
        "session-2",
        "user",
        "What is your return policy?",
    )

    history_1 = memory.get_history("session-1")
    history_2 = memory.get_history("session-2")

    assert len(history_1) == 1
    assert len(history_2) == 1

    assert history_1[0]["content"] == "Where is my order?"
    assert history_2[0]["content"] == "What is your return policy?"


def test_recent_history_is_limited():
    memory = SessionMemory()

    for i in range(10):
        memory.add_message(
            "session-1",
            "user",
            f"message {i}",
        )

    history = memory.get_recent_history(
        "session-1",
        max_messages=3,
    )

    assert len(history) == 3
    assert history[0]["content"] == "message 7"
    assert history[-1]["content"] == "message 9"


def test_session_can_be_cleared():
    memory = SessionMemory()

    memory.add_message(
        "session-1",
        "user",
        "Hello",
    )

    memory.clear_session("session-1")

    assert memory.get_history("session-1") == []
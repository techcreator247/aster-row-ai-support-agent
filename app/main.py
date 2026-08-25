from app.agent import SupportAgent


def main():
    print("=" * 60)
    print("Aster & Row AI Support Agent")
    print("=" * 60)
    print("Type 'exit' to quit.")
    print()

    try:
        agent = SupportAgent()
    except Exception as error:
        print(f"Failed to start agent: {error}")
        return

    session_id = "cli-session"

    while True:
        try:
            user_message = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if user_message.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not user_message:
            continue

        try:
            response = agent.answer(
                session_id=session_id,
                user_message=user_message,
            )

            print()
            print("Agent:")
            print(response)
            print()

        except Exception as error:
            print()
            print(
                "Agent error. Please try again or contact "
                "human support."
            )
            print(f"Debug error: {error}")
            print()


if __name__ == "__main__":
    main()
import os

from dotenv import load_dotenv


load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


def validate_config() -> None:
    """Validate required application configuration."""

    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY is not configured. "
            "Add it to your .env file."
        )
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    PROJECT_NAME: str = "Speedboat API"
    PROJECT_VERSION: str = "1.0.0"
    AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_KEY: str = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_EMBED_MODEL: str = os.getenv("AZURE_OPENAI_EMBED_MODEL")
    AZURE_OPENAI_EMBED_API_ENDPOINT: str = os.getenv("AZURE_OPENAI_EMBED_API_ENDPOINT")
    AZURE_OPENAI_EMBED_API_KEY: str = os.getenv("AZURE_OPENAI_EMBED_API_KEY")
    AZURE_OPENAI_EMBED_VERSION: str = os.getenv("AZURE_OPENAI_EMBED_VERSION")



settings = Settings()

if (
    not settings.AZURE_OPENAI_ENDPOINT
    or not settings.AZURE_OPENAI_API_KEY
    or not settings.AZURE_OPENAI_EMBED_MODEL
    or not settings.AZURE_OPENAI_EMBED_API_ENDPOINT
    or not settings.AZURE_OPENAI_EMBED_API_KEY
    or not settings.AZURE_OPENAI_EMBED_VERSION
):
    raise ValueError(
        ".env is not set properly"
    )

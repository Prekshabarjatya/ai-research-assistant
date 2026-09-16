from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"

    # Enables the /api/research web-search step (see app/agents/graph.py).
    # Local retrieval and /api/query work with this empty.
    tavily_api_key: str = ""
    web_search_max_results: int = 4

    # Cap on the gather->write->review revision loop in the research graph.
    max_research_revisions: int = 1

    # Optional shared-password gate for public deployments (see app/main.py).
    # Empty (the local-dev default) means no auth is enforced.
    app_password: str = ""

    # Chunking. Characters, not tokens — good enough for the splitter's
    # separator-based strategy and avoids pulling in a tokenizer dependency.
    chunk_size: int = 900
    chunk_overlap: int = 150

    # Retrieval.
    top_k: int = 4
    min_relevance_score: float = 0.05

    max_upload_mb: int = 10


settings = Settings()

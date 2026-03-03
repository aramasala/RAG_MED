"""Configuration settings for RAG_MED."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    temperature: float = 0.7
    max_tokens: int = 5000

    # Text splitting settings
    chunk_size: int = 2000
    chunk_overlap: int = 200
    min_chunk_words: int = 20
    num_chunks_to_select: int = 6

    start_section_text: str = "Список литературы"
    end_section_text: str = "Приложение А2. Методология разработки клинических рекомендаций"

    # ValueAI RAG
    valueai_base_url: str = "https://ml-request-develop2.wavea.cc/api/external/v1"
    valueai_username: str | None = None
    valueai_password: str | None = None
    valueai_rag_id: int = 387
    valueai_model_name: str = "llm_qwen_3_32b_q8"
    valueai_instructions: str = "you are helpful assistant"
    valueai_poll_interval_seconds: float = 2.0
    valueai_timeout_seconds: float = 600  
    ragas_max_tokens: int = 8192

    # Metrics LLM
    metrics_llm_model_name: str = "llm_qwen_2_5_coder_32b_instruct_q8"
    metrics_llm_poll_interval_seconds: float = 2.0
    metrics_llm_timeout_seconds: float = 600

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()

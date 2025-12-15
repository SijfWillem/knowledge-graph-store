from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # LangFuse
    langfuse_public_key: str
    langfuse_secret_key: str
    langfuse_host: str = "https://cloud.langfuse.com"

    # Cognee
    cognee_llm_api_key: str

    # Database
    database_url: str

    # Redis
    redis_url: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

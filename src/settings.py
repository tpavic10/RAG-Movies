from pydantic_settings import BaseSettings


class _Settings(BaseSettings):
    HF_TOKEN: str

    class Config:
        env_file = ".env"
        case_sensitive = True


Settings = _Settings()

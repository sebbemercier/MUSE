# Copyright 2026 The OpenSLM Project
from pydantic_settings import BaseSettings, SettingsConfigDict

class MuseSettings(BaseSettings):
    # AI Settings
    TOKENIZER_PATH: str = "models/ecommerce_tokenizer.model"
    BASE_MODEL_ID: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    # SEO Settings
    DEFAULT_LANGUAGE: str = "fr"
    MAX_COPY_LENGTH: int = 500
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = MuseSettings()

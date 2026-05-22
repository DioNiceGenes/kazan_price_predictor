from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # Пути
    ALL_DATA_PATH: str = "data"
    RAW_DATA_PATH: str = "data/raw/kazan-avito-2024-2026.csv"
    IMAGES_BASE_PATH: str = "data/images"
    PROCESSED_DIR: str = "data/processed"
    
    # Названия сохраняемых файлов
    TDF_PATH: str = "data/processed/tdf.csv"
    IMAGE_FOLDERS_PATH: str = "data/processed/image_folders.csv"
    DESCRIPTIONS_PATH: str = "data/processed/descriptions.csv"
    
    class Config:
        env_file = ".env"
        extra = "allow"


settings = Settings()
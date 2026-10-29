"""Application configuration.

All tunables are environment-driven (prefix ``MEDDIAG_``) with sane defaults so
the service boots with zero configuration.  A ``.env`` file next to the project
root is picked up automatically.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# D:\minimaxc\meddiag\app\config.py  ->  project root is one level above `app`
PROJECT_ROOT = Path(__file__).resolve().parents[1]

Modality = Literal["ecg", "eeg", "xray"]


class Settings(BaseSettings):
    """Runtime settings for the MedDiag backend."""

    model_config = SettingsConfigDict(
        env_prefix="MEDDIAG_",
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------ app
    app_name: str = "MedDiag"
    app_title: str = "AI-Based Medical Diagnosis Assistant"
    version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    root_path: str = ""

    # --------------------------------------------------------------- paths
    data_dir: Path = PROJECT_ROOT / "data"
    model_dir: Path = PROJECT_ROOT / "models"
    report_dir: Path = PROJECT_ROOT / "reports"
    storage_dir: Path = PROJECT_ROOT / "storage"
    static_dir: Path = PROJECT_ROOT / "app" / "web" / "static"

    # ------------------------------------------------------------ behaviour
    api_key: str | None = None
    """Optional shared secret.  When set, every ``/api`` call needs ``X-API-Key``."""

    max_upload_mb: int = 25
    allowed_image_types: List[str] = ["image/png", "image/jpeg", "image/jpg", "image/dicom", "application/dicom"]

    # Signal handling defaults
    ecg_default_sampling_rate: float = 500.0
    eeg_default_sampling_rate: float = 256.0
    ecg_window_seconds: float = 10.0
    eeg_window_seconds: float = 10.0

    # Model behaviour
    device: Literal["auto", "cpu", "cuda"] = "auto"
    inference_batch_size: int = 8
    xai_enabled: bool = True
    xai_max_features: int = 8

    # Storage
    retain_analyses_days: int = 30
    audit_db_path: Path = PROJECT_ROOT / "storage" / "meddiag.db"

    @field_validator("allowed_image_types", mode="before")
    @classmethod
    def _split_types(cls, value):
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    def ensure_directories(self) -> None:
        """Create the writable directories the service needs at runtime."""
        for directory in (
            self.data_dir,
            self.model_dir,
            self.report_dir,
            self.storage_dir,
            self.audit_db_path.parent,
        ):
            Path(directory).mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings (singleton)."""
    return Settings()


settings = get_settings()

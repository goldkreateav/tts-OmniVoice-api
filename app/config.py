from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    api_path: str = Field(default="/v1/synthesize")
    tts_api_key: str | None = Field(default=None)

    # Response mode: binary | base64
    response_mode: str = Field(default="binary")

    model_name: str = Field(default="k2-fsa/OmniVoice")
    model_device_map: str = Field(default="auto")
    model_dtype: str = Field(default="float16")
    sample_rate_hz: int = Field(default=24000)

    # Generation quality knobs (help reduce occasional noisy output)
    gen_num_step: int = Field(default=32)
    gen_num_step_fallback: int = Field(default=48)
    gen_fallback_dtype: str = Field(default="float32")

    min_rate: float = Field(default=0.5)
    max_rate: float = Field(default=2.0)
    max_text_chars: int = Field(default=2000)

    max_concurrent_requests: int = Field(default=1)
    semaphore_acquire_timeout_sec: float = Field(default=0.05)
    synthesis_timeout_sec: float = Field(default=90.0)
    warmup_on_startup: bool = Field(default=False)

    ffmpeg_binary: str = Field(default="ffmpeg")
    voices_file: str = Field(default="app/voices.json")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def resolve_voices_path(base_dir: Path, settings: Settings) -> Path:
    return (base_dir / settings.voices_file).resolve()

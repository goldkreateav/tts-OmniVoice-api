from pydantic import BaseModel, Field, field_validator


class SynthesizeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Chunk text to synthesize")
    voice: str = Field(..., min_length=1, description="Voice id or 'default'")
    rate: float = Field(..., description="Speech rate, usually 0.5..2.0")
    format: str = Field(..., min_length=1, description="Audio format, e.g. wav/mp3/ogg")

    @field_validator("text")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("text must not be empty")
        return normalized

    @field_validator("voice")
    @classmethod
    def normalize_voice(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("voice must not be empty")
        return normalized

    @field_validator("format")
    @classmethod
    def normalize_format(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not normalized:
            raise ValueError("format must not be empty")
        return normalized


class ApiErrorBody(BaseModel):
    code: str
    message: str


class ApiErrorResponse(BaseModel):
    error: ApiErrorBody

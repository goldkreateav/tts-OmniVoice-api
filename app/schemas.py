from pydantic import BaseModel, Field, field_validator


class SynthesizeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Текст для озвучки (чанк)")
    voice: str = Field(..., min_length=1, description="Идентификатор голоса или 'default'")
    rate: float = Field(..., description="Скорость речи, обычно 0.5..2.0")
    format: str = Field(..., min_length=1, description="Формат аудио, например wav/mp3/ogg")

    @field_validator("text")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Поле text не должно быть пустым")
        return normalized

    @field_validator("voice")
    @classmethod
    def normalize_voice(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Поле voice не должно быть пустым")
        return normalized

    @field_validator("format")
    @classmethod
    def normalize_format(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not normalized:
            raise ValueError("Поле format не должно быть пустым")
        return normalized


class ApiErrorBody(BaseModel):
    code: str
    message: str


class ApiErrorResponse(BaseModel):
    error: ApiErrorBody

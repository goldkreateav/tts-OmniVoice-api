from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import ValidationError

from app.config import get_settings, resolve_voices_path
from app.engine_omnivoice import OmniVoiceEngine
from app.schemas import ApiErrorBody, ApiErrorResponse, SynthesizeRequest
from app.transcode import transcode_wav_bytes


def media_type_for_format(fmt: str) -> str:
    mapping = {
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
        "ogg": "audio/ogg",
    }
    if fmt not in mapping:
        raise ValueError(f"unsupported format: {fmt}")
    return mapping[fmt]


def make_error_response(status_code: int, code: str, message: str) -> JSONResponse:
    payload = ApiErrorResponse(error=ApiErrorBody(code=code, message=message)).model_dump()
    return JSONResponse(status_code=status_code, content=payload)


settings = get_settings()
app = FastAPI(title="OmniVoice TTS API", version="1.0.0")

app.state.base_dir = Path(__file__).resolve().parent.parent
app.state.settings = settings
app.state.engine = OmniVoiceEngine(
    model_name=settings.model_name,
    device_map=settings.model_device_map,
    dtype=settings.model_dtype,
    sample_rate_hz=settings.sample_rate_hz,
)
app.state.synthesis_semaphore = asyncio.Semaphore(settings.max_concurrent_requests)
app.state.voice_map = {}


@app.on_event("startup")
async def on_startup() -> None:
    voices_path = resolve_voices_path(app.state.base_dir, settings)
    if voices_path.exists():
        with voices_path.open("r", encoding="utf-8") as f:
            app.state.voice_map = json.load(f)

    if settings.warmup_on_startup:
        await asyncio.to_thread(app.state.engine.get_model)


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    code = f"http_{exc.status_code}"
    message = str(exc.detail)
    return make_error_response(exc.status_code, code, message)


@app.exception_handler(ValidationError)
async def validation_exception_handler(_: Request, exc: ValidationError) -> JSONResponse:
    return make_error_response(400, "invalid_request", exc.errors()[0]["msg"])


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    return make_error_response(500, "internal_error", str(exc))


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"ok": True}

@app.get("/v1/voices")
async def list_voices() -> dict[str, Any]:
    voices: list[dict[str, Any]] = [{"id": "default"}]
    for voice_id in sorted(app.state.voice_map.keys()):
        voices.append({"id": voice_id, "instruct": app.state.voice_map[voice_id]})
    return {"voices": voices}


async def synthesize(
    request: Request,
    authorization: str | None = Header(default=None),
    accept: str | None = Header(default=None),
) -> Response:
    content_type = request.headers.get("content-type", "")
    if not content_type.lower().startswith("application/json"):
        raise HTTPException(status_code=400, detail="content-type must be application/json")

    if accept and "audio/" not in accept and "application/json" not in accept and "*/*" not in accept:
        raise HTTPException(status_code=406, detail="not acceptable")

    if settings.tts_api_key:
        if not authorization:
            raise HTTPException(status_code=401, detail="missing bearer token")
        expected_value = f"Bearer {settings.tts_api_key}"
        if authorization.strip() != expected_value:
            raise HTTPException(status_code=403, detail="invalid bearer token")

    payload = SynthesizeRequest(**await request.json())

    if len(payload.text) > settings.max_text_chars:
        raise HTTPException(
            status_code=400,
            detail=f"text is too long: max {settings.max_text_chars} chars",
        )

    if payload.rate < settings.min_rate or payload.rate > settings.max_rate:
        raise HTTPException(
            status_code=400,
            detail=f"rate must be between {settings.min_rate} and {settings.max_rate}",
        )

    target_format = payload.format.lower()
    if target_format not in {"wav", "mp3", "ogg"}:
        raise HTTPException(status_code=400, detail="unsupported format")

    instruct: str | None = None
    voice_key = payload.voice.strip()
    if voice_key.lower() != "default":
        instruct = app.state.voice_map.get(voice_key)
        if not instruct:
            raise HTTPException(status_code=400, detail=f"unsupported voice: {voice_key}")

    try:
        await asyncio.wait_for(
            app.state.synthesis_semaphore.acquire(),
            timeout=settings.semaphore_acquire_timeout_sec,
        )
    except TimeoutError as exc:
        raise HTTPException(status_code=429, detail="server is busy, retry later") from exc

    try:
        wav_bytes = await asyncio.wait_for(
            asyncio.to_thread(
                app.state.engine.synthesize_wav_bytes,
                payload.text,
                payload.rate,
                instruct,
            ),
            timeout=settings.synthesis_timeout_sec,
        )
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail="synthesis timeout") from exc
    finally:
        app.state.synthesis_semaphore.release()

    if target_format == "wav":
        audio_bytes = wav_bytes
    else:
        audio_bytes = await asyncio.to_thread(
            transcode_wav_bytes,
            wav_bytes,
            target_format,
            settings.ffmpeg_binary,
        )

    if settings.response_mode == "base64":
        return JSONResponse({"audioBase64": base64.b64encode(audio_bytes).decode("ascii")})

    return Response(content=audio_bytes, media_type=media_type_for_format(target_format))


app.add_api_route(settings.api_path, synthesize, methods=["POST"])

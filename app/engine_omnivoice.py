from __future__ import annotations

import io
import threading
from typing import Any

import soundfile as sf


class OmniVoiceEngine:
    def __init__(self, model_name: str, device_map: str, dtype: str, sample_rate_hz: int) -> None:
        self.model_name = model_name
        self.device_map = device_map
        self.dtype = dtype
        self.sample_rate_hz = sample_rate_hz

        self._lock = threading.Lock()
        self._model: Any | None = None

    def _resolve_dtype(self) -> Any:
        import torch

        mapping = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        return mapping.get(self.dtype.lower(), torch.float16)

    def get_model(self) -> Any:
        if self._model is not None:
            return self._model

        with self._lock:
            if self._model is not None:
                return self._model

            from omnivoice import OmniVoice

            self._model = OmniVoice.from_pretrained(
                self.model_name,
                device_map=self.device_map,
                dtype=self._resolve_dtype(),
            )
            return self._model

    def synthesize_wav_bytes(self, text: str, speed: float, instruct: str | None = None) -> bytes:
        model = self.get_model()

        kwargs: dict[str, Any] = {
            "text": text,
            "speed": speed,
        }
        if instruct:
            kwargs["instruct"] = instruct

        audio_list = model.generate(**kwargs)
        if not audio_list:
            raise RuntimeError("OmniVoice returned no audio")

        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio_list[0], self.sample_rate_hz, format="WAV", subtype="PCM_16")
        return wav_buffer.getvalue()

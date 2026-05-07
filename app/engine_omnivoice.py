from __future__ import annotations

import io
import threading
from typing import Any

import numpy as np
import soundfile as sf


class OmniVoiceEngine:
    def __init__(self, model_name: str, device_map: str, dtype: str, sample_rate_hz: int) -> None:
        self.model_name = model_name
        self.device_map = device_map
        self.dtype = dtype
        self.sample_rate_hz = sample_rate_hz

        self._lock = threading.Lock()
        self._models: dict[str, Any] = {}

    def _resolve_dtype(self) -> Any:
        import torch

        mapping = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        return mapping.get(self.dtype.lower(), torch.float16)

    def _resolve_dtype_by_name(self, dtype_name: str) -> Any:
        import torch

        mapping = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        return mapping.get(dtype_name.lower(), torch.float32)

    def _effective_device_map(self) -> str:
        """
        OmniVoice/accelerate may shard weights across multiple CUDA devices when device_map="auto".
        Some internal ops can then end up mixing tensors across devices (e.g. cuda:0 + cuda:1),
        causing runtime errors. To keep inference stable, we force a single GPU when multiple are present.
        """
        try:
            import torch

            if self.device_map == "auto" and torch.cuda.is_available() and torch.cuda.device_count() > 1:
                return "cuda:0"
        except Exception:
            pass

        return self.device_map

    def get_model(self, dtype_name: str | None = None) -> Any:
        dtype_key = (dtype_name or self.dtype).lower()
        if dtype_key in self._models:
            return self._models[dtype_key]

        with self._lock:
            if dtype_key in self._models:
                return self._models[dtype_key]

            from omnivoice import OmniVoice

            model = OmniVoice.from_pretrained(
                self.model_name,
                device_map=self._effective_device_map(),
                dtype=self._resolve_dtype_by_name(dtype_key),
            )
            self._models[dtype_key] = model
            return model

    def _postprocess_audio(self, audio: Any) -> np.ndarray:
        arr = np.asarray(audio, dtype=np.float32)
        if arr.size == 0:
            raise ValueError("Пустой аудиосигнал")
        if not np.isfinite(arr).all():
            raise ValueError("Аудио содержит NaN/Inf")
        max_abs = float(np.max(np.abs(arr)))
        if max_abs > 1.0:
            arr = arr / max_abs
        arr = np.clip(arr, -1.0, 1.0)
        return arr

    def synthesize_wav_bytes(
        self,
        text: str,
        speed: float,
        num_step: int,
        instruct: str | None = None,
        *,
        dtype_name: str | None = None,
    ) -> bytes:
        model = self.get_model(dtype_name=dtype_name)

        kwargs: dict[str, Any] = {
            "text": text,
            "speed": speed,
            "num_step": num_step,
        }
        if instruct:
            kwargs["instruct"] = instruct

        try:
            import torch

            with torch.inference_mode():
                audio_list = model.generate(**kwargs)
        except Exception:
            audio_list = model.generate(**kwargs)

        if not audio_list:
            raise RuntimeError("OmniVoice не вернул аудио")

        audio = self._postprocess_audio(audio_list[0])

        wav_buffer = io.BytesIO()
        sr = int(getattr(model, "sampling_rate", self.sample_rate_hz))
        sf.write(wav_buffer, audio, sr, format="WAV", subtype="PCM_16")
        return wav_buffer.getvalue()

    def synthesize_wav_bytes_with_fallback(
        self,
        text: str,
        speed: float,
        num_step: int,
        fallback_num_step: int,
        fallback_dtype: str,
        instruct: str | None = None,
    ) -> bytes:
        try:
            return self.synthesize_wav_bytes(text, speed, num_step, instruct)
        except Exception:
            return self.synthesize_wav_bytes(
                text,
                speed,
                fallback_num_step,
                instruct,
                dtype_name=fallback_dtype,
            )

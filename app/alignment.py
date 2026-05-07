from __future__ import annotations

import tempfile
import threading
from dataclasses import dataclass
from typing import Any

import soundfile as sf


@dataclass(frozen=True)
class WordSpan:
    word: str
    startSec: float
    endSec: float


class WhisperXAligner:
    def __init__(self, language_code: str, device: str) -> None:
        self.language_code = language_code
        self.device = device

        self._lock = threading.Lock()
        self._model_a: Any | None = None
        self._metadata: Any | None = None

    def _get_model(self) -> tuple[Any, Any]:
        if self._model_a is not None and self._metadata is not None:
            return self._model_a, self._metadata

        with self._lock:
            if self._model_a is not None and self._metadata is not None:
                return self._model_a, self._metadata

            import whisperx

            model_a, metadata = whisperx.load_align_model(language_code=self.language_code, device=self.device)
            self._model_a = model_a
            self._metadata = metadata
            return model_a, metadata

    def align_words(self, text: str, wav_bytes: bytes) -> list[WordSpan]:
        """
        Best-effort forced alignment using WhisperX.

        Notes:
        - OmniVoice doesn't provide native alignment. WhisperX aligns text to audio.
        - The transcript MUST roughly match the audio; otherwise some words may be missing.
        """
        import whisperx

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
            f.write(wav_bytes)
            f.flush()

            # duration from container (more reliable than assuming sample rate)
            info = sf.info(f.name)
            duration_sec = float(info.duration)

            audio = whisperx.load_audio(f.name)  # typically 16 kHz float32
            # compute duration in seconds using WhisperX's resampled audio length
            if hasattr(audio, "__len__") and len(audio) > 0:
                duration_sec = max(duration_sec, float(len(audio)) / 16000.0)

            model_a, metadata = self._get_model()
            segments = [{"text": text, "start": 0.0, "end": duration_sec}]
            aligned = whisperx.align(segments, model_a, metadata, audio, self.device)

        words: list[WordSpan] = []
        for seg in (aligned.get("segments") or []):
            for w in (seg.get("words") or []):
                word = (w.get("word") or "").strip()
                start = w.get("start")
                end = w.get("end")
                if not word or start is None or end is None:
                    continue
                words.append(WordSpan(word=word, startSec=float(start), endSec=float(end)))
        return words


_whisperx_singletons: dict[tuple[str, str], WhisperXAligner] = {}
_whisperx_singletons_lock = threading.Lock()


def get_whisperx_aligner(language_code: str, device: str) -> WhisperXAligner:
    key = (language_code, device)
    if key in _whisperx_singletons:
        return _whisperx_singletons[key]

    with _whisperx_singletons_lock:
        if key in _whisperx_singletons:
            return _whisperx_singletons[key]
        aligner = WhisperXAligner(language_code=language_code, device=device)
        _whisperx_singletons[key] = aligner
        return aligner


from __future__ import annotations

import subprocess


def transcode_wav_bytes(wav_bytes: bytes, target_format: str, ffmpeg_binary: str = "ffmpeg") -> bytes:
    fmt = target_format.lower()
    if fmt == "wav":
        return wav_bytes

    codec_args: list[str]
    if fmt == "mp3":
        codec_args = ["-f", "mp3", "-codec:a", "libmp3lame"]
    elif fmt == "ogg":
        codec_args = ["-f", "ogg", "-codec:a", "libvorbis"]
    else:
        raise ValueError(f"unsupported format: {target_format}")

    cmd = [
        ffmpeg_binary,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        *codec_args,
        "pipe:1",
    ]

    try:
        completed = subprocess.run(
            cmd,
            input=wav_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is not installed or not found in PATH") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(f"ffmpeg transcode failed: {stderr or 'unknown error'}") from exc

    return completed.stdout

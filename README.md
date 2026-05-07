# OmniVoice TTS API

FastAPI сервис, реализующий контракт TTS API поверх `OmniVoice`.

## Что реализовано

- `POST /v1/synthesize` (путь можно сменить через `API_PATH`).
- Авторизация `Authorization: Bearer <key>` при заданном `TTS_API_KEY`.
- Request JSON:
  - `text` (required)
  - `voice` (required, `default` или ключ из `app/voices.json`)
  - `rate` (required, диапазон по умолчанию `0.5..2.0`)
  - `format` (required: `wav`, `mp3`, `ogg`)
- Response:
  - по умолчанию бинарное аудио `audio/*`
  - опционально JSON `{ "audioBase64": "..." }` через `RESPONSE_MODE=base64`
- Ошибки: `400/401/403/404/429/5xx` (в JSON формате `{"error":{"code","message"}}`).

## Установка (Windows / PowerShell)

```powershell
cd "c:\Users\Valentin\Downloads\TTS API"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Дополнительно для `mp3/ogg` нужен `ffmpeg` в `PATH`.

### Примечание про GPU (исправляет 500 с cuda:0/cuda:1)

Если на сервере несколько GPU, `MODEL_DEVICE_MAP=auto` может привести к шардированию модели по нескольким CUDA-устройствам и ошибке вида:

`Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cuda:1`

В этом случае задайте одну GPU явно, например:

`MODEL_DEVICE_MAP=cuda:0`

## Запуск

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Проверка:

```powershell
curl http://localhost:8000/health
```

Список голосов:

```powershell
curl http://localhost:8000/v1/voices
```

## Переменные окружения

- `API_PATH` (default: `/v1/synthesize`)
- `TTS_API_KEY` (optional)
- `RESPONSE_MODE` (`binary` or `base64`, default: `binary`)
- `MODEL_NAME` (default: `k2-fsa/OmniVoice`)
- `MODEL_DEVICE_MAP` (default: `auto`)
- `MODEL_DTYPE` (default: `float16`)
- `SAMPLE_RATE_HZ` (default: `24000`)
- `MIN_RATE` (default: `0.5`)
- `MAX_RATE` (default: `2.0`)
- `MAX_TEXT_CHARS` (default: `2000`)
- `MAX_CONCURRENT_REQUESTS` (default: `1`)
- `SEMAPHORE_ACQUIRE_TIMEOUT_SEC` (default: `0.05`)
- `SYNTHESIS_TIMEOUT_SEC` (default: `90`)
- `WARMUP_ON_STARTUP` (default: `false`)
- `FFMPEG_BINARY` (default: `ffmpeg`)
- `VOICES_FILE` (default: `app/voices.json`)

## Примеры запросов

### WAV

```powershell
curl -X POST "http://localhost:8000/v1/synthesize" `
  -H "Content-Type: application/json" `
  -H "Accept: audio/*, application/json" `
  -o out.wav `
  -d "{\"text\":\"Привет! Это тест TTS.\",\"voice\":\"default\",\"rate\":1.0,\"format\":\"wav\"}"
```

### MP3

```powershell
curl -X POST "http://localhost:8000/v1/synthesize" `
  -H "Content-Type: application/json" `
  -o out.mp3 `
  -d "{\"text\":\"Привет! Это тест MP3.\",\"voice\":\"assistant_friendly\",\"rate\":1.0,\"format\":\"mp3\"}"
```

### OGG

```powershell
curl -X POST "http://localhost:8000/v1/synthesize" `
  -H "Content-Type: application/json" `
  -o out.ogg `
  -d "{\"text\":\"Привет! Это тест OGG.\",\"voice\":\"assistant_formal\",\"rate\":1.0,\"format\":\"ogg\"}"
```

### Проверка ошибок контракта

- Пустой `text` -> `400`
- Неверный `rate` (< 0.5 или > 2.0) -> `400`
- Неподдерживаемый `format` -> `400`
- При заданном `TTS_API_KEY`:
  - без `Authorization` -> `401`
  - неверный токен -> `403`
- При перегрузке (занят семафор) -> `429`

## Замечания

- В первой итерации реализован основной режим ответа A (бинарное аудио).
- Режим B (`audioBase64`) включается через `RESPONSE_MODE=base64`.
- Режим C (`audioUrl`) не включен в эту версию.

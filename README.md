# VoiceFlow AI — Транскрибатор + Саммари

Локальный пайплайн для **транскрибации встреч и генерации саммари**:

1. Загружаете аудио → **faster-whisper** распознаёт речь (русский по умолчанию).
2. **NeMo TitaNet + спектральная кластеризация** размечают спикеров (диаризация).
3. **Ollama** (локальная LLM) генерирует саммари по выбранному промпту-роли.
4. Транскрипт, саммари и метрики качества сохраняются в ваш **Obsidian vault** как Markdown.

Долгие задачи выполняются асинхронно через **Celery + Redis**, а фронтенд
опрашивает статус и показывает прогресс, метрики и превью.

---

## Структура проекта

```
.
├── app/
│   ├── __init__.py
│   ├── config.py        # вся конфигурация из .env (без хардкодов)
│   ├── prompts.py       # 5 ролевых промптов + безопасный рендер
│   ├── obsidian.py      # построение Markdown + сохранение в vault
│   ├── worker.py        # Celery: Whisper + диаризация + Ollama (ленивые ML-импорты)
│   └── main.py          # FastAPI: /upload /summarize-file /status /cancel /prompts /health
├── static/
│   └── index.html       # фронтенд (Tailwind), привязан к API
├── tests/
│   └── test_smoke.py    # smoke-тесты (без GPU/Redis/Ollama)
├── uploads/             # загруженные аудио (gitignored)
├── reports/             # JSONL/CSV-метрики MVP для отчёта (gitignored)
├── .github/workflows/   # GitHub Actions smoke tests
├── requirements.txt     # полный стек (вкл. torch/NeMo)
├── requirements-min.txt # лёгкий стек (API+очередь+саммари, без ASR/диаризации)
├── pyproject.toml       # зависимости и extras для запуска через uv
├── Dockerfile           # контейнер API/worker
├── docker-compose.yml   # api + worker + redis + ollama
├── docker-compose.gpu.yml # optional GPU override для NVIDIA Docker
├── .env.example         # шаблон переменных окружения
├── .env.docker.example  # шаблон окружения для Docker Compose
├── .env.windows.example # Windows-friendly шаблон + CUDA PyTorch подсказки
├── run_api.sh           # запуск FastAPI
└── run_worker.sh        # запуск Celery-воркера
```

---

## Быстрый старт

### Вариант A. Локальный запуск через uv

`uv` — рекомендуемый способ для локального dev/prod-запуска без ручного
создания virtualenv.

```bash
# Системный ffmpeg (обязательно для конвертации аудио)
# macOS:   brew install ffmpeg
# Ubuntu:  sudo apt update && sudo apt install ffmpeg
# Windows: choco install ffmpeg

# Установка uv, если его ещё нет:
# macOS/Linux: curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows:     powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

cp .env.example .env
uv sync --extra dev

# Если нужна NeMo-диаризация:
uv sync --extra dev --extra diarization

# Терминал 1 — Redis, если не установлен локально:
docker run --rm -p 6379:6379 redis:7-alpine

# Терминал 2 — воркер:
uv run celery -A app.worker.app worker --loglevel=info --pool=solo

# Терминал 3 — API + фронтенд:
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Откройте <http://localhost:8000>. Для саммари нужен запущенный Ollama:

```bash
ollama pull gemma3:12b
ollama serve
```

Полезные команды:

```bash
uv run pytest -q
uv run python -m app.main
uv pip install -r requirements.txt
```

#### GPU/CUDA на Windows: точная установка PyTorch

Для этого проекта есть два разных CUDA-потребителя:

1. **PyTorch / NeMo** — использует `torch`.
2. **Whisper-бэкенды** — сначала используется `faster-whisper / CTranslate2`,
   а при CUDA runtime ошибках автоматически пробуется **WhisperX**.

Для RTX 50xx/Blackwell принципиально не опускайтесь ниже **CUDA 13.x / cu130**:
карты RTX 5080/5090 используют архитектуру `sm_120`, а сборки PyTorch CUDA 12.x
могут видеть GPU, но падать при выполнении CUDA kernels с ошибкой `CUDA error:
no kernel image is available for execution on the device`.

Перед переустановкой проверьте текущий `torch`:

```powershell
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

##### Рекомендуемый вариант: CUDA 13.0 / cu130 через uv

В `pyproject.toml` уже настроен индекс `pytorch-cu130`, поэтому основной сценарий
установки такой:

```powershell
uv sync --extra dev

# Если нужна NeMo-диаризация:
uv sync --extra dev --extra diarization
```

Проверьте:

```powershell
uv run python -c "import torch; print('torch:', torch.__version__); print('torch cuda:', torch.version.cuda); print('cuda available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
uv run python -c "import ctranslate2; print('ctranslate2:', ctranslate2.__version__); print('cuda devices:', ctranslate2.get_cuda_device_count())"
uv run python -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cuda', compute_type='float16'); print('faster-whisper CUDA OK')"
uv run python -c "import whisperx; print('whisperx OK')"
uv run python -c "import silero_vad; print('silero-vad OK')"
```

Ожидаемый результат:

```text
torch cuda: 13.0
cuda available: True
cuda devices: 1
faster-whisper CUDA OK или автоматический fallback на WhisperX при обработке
whisperx OK
silero-vad OK
```

Если при обработке `faster-whisper` падает с:

```text
RuntimeError: Library cublas64_12.dll is not found or cannot be loaded
```

По умолчанию worker не пробует WhisperX, потому что на Windows + RTX 50xx +
`cu130` он может зависать даже на коротких файлах после загрузки Silero VAD.
Практический MVP-режим: сразу уйти на Faster-Whisper CPU/int8 fallback.

Если всё же нужно экспериментально включить второй CUDA-бэкенд, выставьте:

```env
WHISPERX_FALLBACK_ENABLED=true
```

Тогда ожидаемый лог:

```text
Whisper CUDA failed (...). Trying WhisperX as fallback...
WhisperX loaded successfully on CUDA
```

Если WhisperX на свежем `torchaudio` падает с:

```text
module 'torchaudio' has no attribute 'AudioMetaData'
```

в проекте есть compatibility patch перед импортом WhisperX: worker восстановит
`torchaudio.AudioMetaData` из внутренних модулей torchaudio или создаст
совместимый dataclass, после чего повторит загрузку WhisperX в рамках того же
fallback-сценария. После обновления файлов полностью перезапустите Celery worker,
потому что уже импортированные Python-модули не патчатся в старом процессе.

Тот же compatibility patch восстанавливает удалённые legacy helpers:
`torchaudio.list_audio_backends`, `torchaudio.get_audio_backend` и
`torchaudio.set_audio_backend`. Они нужны некоторым версиям WhisperX/pyannote на
свежем `torchaudio`.

Если WhisperX доходит до VAD и падает на:

```text
Lazy import of ... speechbrain.integrations.k2_fsa ... failed
```

worker предпочитает `vad_method="silero"` при загрузке WhisperX. Это обходит
проблемный Pyannote/SpeechBrain VAD в CUDA 13.x окружениях и оставляет WhisperX
как резервный CUDA-бэкенд именно для транскрибации.

Не заменяйте это на `vad_options=None`: в некоторых версиях WhisperX это не
отключает VAD, а приводит к выбору default Pyannote VAD. В проекте default VAD
специально не используется как fallback: если установленная версия WhisperX не
принимает `vad_method="silero"`, задача безопасно уйдёт на Faster-Whisper
CPU/int8 вместо повторного падения в Pyannote/SpeechBrain.

Для загрузки аудио в WhisperX используется `soundfile.read(...)`, а не
`whisperx.load_audio` и не `librosa.load`. Файл к этому моменту уже подготовлен
через ffmpeg как mono/16 kHz WAV, поэтому `soundfile` достаточно. Это снижает
зависимость от legacy `torchaudio` I/O API и не триггерит lazy-import цепочки
`librosa -> inspect -> speechbrain`.

Если после успешной загрузки Silero появляется:

```text
Lazy import of ... speechbrain.integrations.k2_fsa ... failed
```

это optional-интеграция SpeechBrain с K2-FSA, которая не нужна для нашего
WhisperX ASR fallback. Worker заранее ставит compatibility stub для
`speechbrain.integrations.k2_fsa`, чтобы отсутствие/несовместимость K2 не
роняла транскрибацию.

Аналогично, если всплывает:

```text
Lazy import of ... speechbrain.integrations.nlp ... failed
No module named 'flair'
```

worker ставит compatibility stub для `speechbrain.integrations.nlp`. Flair/NLP
интеграция не нужна для распознавания речи в этом пайплайне.

`silero-vad` добавлен в зависимости проекта. После обновления выполните:

```powershell
uv sync --extra dev --extra diarization
```

```powershell
uv pip uninstall torch torchvision torchaudio -y
uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130
```

После этого снова выполните:

```powershell
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

##### Трёхступенчатый fallback Whisper

Стабильный MVP-режим на Windows + RTX 50xx + `cu130`:

```env
WHISPER_DEVICE=auto
WHISPER_COMPUTE_TYPE=auto
WHISPER_CUDA_ENABLED=false
WHISPERX_FALLBACK_ENABLED=false
WHISPER_FALLBACK_TO_CPU=true
```

В этом режиме worker видит CUDA, но не запускает Faster-Whisper/CTranslate2 на
CUDA, потому что в этой связке он может искать `cublas64_12.dll` или зависать
ещё до исключения. Фактически транскрибация идёт через Faster-Whisper CPU/int8,
а NeMo-диаризация может отдельно использовать свой `DIARIZATION_DEVICE`.

Экспериментальный порядок распознавания, если включить CUDA:

1. `faster-whisper` на CUDA.
2. Если CUDA runtime/CUBLAS/CUDNN падает и `WHISPERX_FALLBACK_ENABLED=true` —
   `WhisperX` на CUDA.
3. Если WhisperX тоже падает и `WHISPER_FALLBACK_TO_CPU=true` —
   `faster-whisper` на CPU/int8.
4. Если `WHISPERX_FALLBACK_ENABLED=false`, шаг 2 пропускается и CPU/int8
   fallback запускается сразу.

##### Переменные `.env` для GPU

После установки CUDA-версии PyTorch выставьте:

```env
WHISPER_DEVICE=auto
WHISPER_COMPUTE_TYPE=auto
WHISPER_CUDA_ENABLED=false
WHISPER_FALLBACK_TO_CPU=true
WHISPERX_FALLBACK_ENABLED=false
DIARIZATION_DEVICE=cuda
DIARIZATION_FALLBACK_TO_CPU=true
```

Во время обработки можно смотреть реальную загрузку GPU:

```powershell
nvidia-smi -l 1
```

Проект также выводит фактические устройства в лог worker, `/health` и блок
«Метрики качества» в UI: `Whisper device`, `NeMo device`, `CUDA доступна`,
`CUDA устройство`, `PyTorch CUDA runtime`. В метриках транскрибации также
сохраняется `whisper_backend`: `faster_whisper` или `whisperx`.

### Вариант B. Классический pip/venv

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Для лёгкого dev-режима без тяжёлого ML-стека:

```bash
pip install -r requirements-min.txt
```

### Вариант C. Docker Compose

Compose поднимает `api`, `worker`, `redis`, `ollama` и named volumes для
загрузок, модели Ollama и Obsidian vault.

```bash
cp .env.docker.example .env.docker

# Собрать API/worker образ. По умолчанию используется requirements.txt.
docker compose build

# Запустить Redis/Ollama/API/worker:
docker compose up -d redis ollama api worker

# Один раз подтянуть модель Ollama внутри docker-сети:
docker compose --profile pull run --rm ollama-pull

# Логи:
docker compose logs -f api worker
```

Откройте <http://localhost:8000>.

Для быстрого CI/dev-образа без тяжёлых ML-зависимостей можно собрать так:

```bash
REQUIREMENTS_FILE=requirements-min.txt docker compose build
```

В таком режиме API/тесты стартуют, но полноценная обработка аудио через
`faster-whisper` недоступна. Для реальной транскрибации используйте обычный
`requirements.txt`.

#### Docker на новом ПК без GPU

Этот сценарий самый предсказуемый и подходит для ноутбуков/ПК без NVIDIA GPU.

```powershell
git clone https://github.com/Mortelnik/ubiquitous-enigma.git
cd ubiquitous-enigma
Copy-Item .env.docker.example .env.docker
```

В `.env.docker` оставьте стабильные значения:

```env
WHISPER_MODEL=base
WHISPER_DEVICE=auto
WHISPER_COMPUTE_TYPE=auto
WHISPER_CUDA_ENABLED=false
WHISPERX_FALLBACK_ENABLED=false
WHISPER_FALLBACK_TO_CPU=true
ENABLE_DIARIZATION=false
```

Запуск:

```powershell
docker compose --env-file .env.docker build
docker compose --env-file .env.docker up -d redis ollama api worker
docker compose --env-file .env.docker --profile pull run --rm ollama-pull
docker compose --env-file .env.docker logs -f worker
```

Откройте <http://localhost:8000>. В этом режиме транскрибация идёт через
Faster-Whisper CPU/int8, диаризация отключена, поэтому запуск максимально
надёжный.

#### Docker если нет RTX 50xx

Сначала установите:

- NVIDIA Driver;
- Docker Desktop;
- включённую поддержку WSL2/Docker GPU;
- проверьте на хосте:

```powershell
nvidia-smi
```

Клонирование и env:

```powershell
git clone https://github.com/<your-user>/<repo>.git
cd <repo>
Copy-Item .env.docker.example .env.docker
```

Для первого запуска начните со стабильного режима:

```env
WHISPER_MODEL=base
WHISPER_CUDA_ENABLED=false
WHISPERX_FALLBACK_ENABLED=false
WHISPER_FALLBACK_TO_CPU=true
ENABLE_DIARIZATION=false
```

Запуск с GPU override:

```powershell
docker compose --env-file .env.docker -f docker-compose.yml -f docker-compose.gpu.yml build
docker compose --env-file .env.docker -f docker-compose.yml -f docker-compose.gpu.yml up -d redis ollama api worker
docker compose --env-file .env.docker -f docker-compose.yml -f docker-compose.gpu.yml --profile pull run --rm ollama-pull
```

Проверка GPU внутри worker:

```powershell
docker compose --env-file .env.docker -f docker-compose.yml -f docker-compose.gpu.yml exec worker python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Если `torch.cuda.is_available()` показывает `True`, можно экспериментально
включить Whisper CUDA:

```env
WHISPER_CUDA_ENABLED=true
WHISPER_DEVICE=auto
WHISPER_COMPUTE_TYPE=auto
WHISPERX_FALLBACK_ENABLED=false
WHISPER_FALLBACK_TO_CPU=true
```

После изменения `.env.docker` перезапустите worker:

```powershell
docker compose --env-file .env.docker -f docker-compose.yml -f docker-compose.gpu.yml up -d --force-recreate worker
docker compose --env-file .env.docker -f docker-compose.yml -f docker-compose.gpu.yml logs -f worker
```

Если появятся ошибки CUDA/CUBLAS/CUDNN или зависания, верните:

```env
WHISPER_CUDA_ENABLED=false
```

Это не ошибка деплоя: CPU/int8 режим является штатным стабильным режимом MVP.
GPU override при этом всё ещё полезен для Ollama и будущих экспериментов с
NeMo/Whisper на совместимых версиях CUDA-библиотек.

### Конфигурация

Для локального запуска:

```bash
cp .env.example .env
# отредактируйте .env: путь к Obsidian vault, модель Whisper, модель Ollama и т.д.
```

Для Docker Compose:

```bash
cp .env.docker.example .env.docker
# внутри Docker используйте REDIS_URL=redis://redis:6379/0 и OLLAMA_HOST=http://ollama:11434
```

Секреты в репозиторий не коммитятся — `.env` и `.env.docker` уже в `.gitignore`.

---

## Переменные окружения

| Переменная           | По умолчанию                                          | Назначение                                              |
| -------------------- | ----------------------------------------------------- | ------------------------------------------------------- |
| `UPLOAD_DIR`         | `uploads`                                             | Каталог для загруженных аудио                           |
| `OBSIDIAN_VAULT`     | `obsidian_vault`                                      | Путь к Obsidian vault                            |
| `OBSIDIAN_FOLDER`    | `Встречи`                                             | Подпапка внутри vault для заметок                       |
| `REPORT_DIR`         | `reports`                                             | JSONL/CSV-метрики обработок для отчёта MVP              |
| `REDIS_URL`          | `redis://localhost:6379/0`                            | Брокер и backend Celery                                 |
| `WHISPER_MODEL`      | `large-v3`                                            | Модель faster-whisper (`tiny`…`large-v3`)               |
| `WHISPER_DEVICE`     | `auto`                                                | `auto`, `cuda` или `cpu`; фактическое устройство видно в логах/UI |
| `WHISPER_COMPUTE_TYPE`| `auto`                                               | `auto`, `float16`, `int8`, `int8_float16`, `float32`    |
| `WHISPER_CUDA_ENABLED` | `false`                                            | `true` — разрешить Faster-Whisper/CTranslate2 CUDA; `false` — сразу CPU/int8 для стабильного MVP |
| `WHISPERX_FALLBACK_ENABLED` | `false`                                      | `true` — пробовать WhisperX CUDA после падения Faster-Whisper CUDA; `false` — сразу CPU/int8 fallback |
| `LANGUAGE`           | `ru`                                                  | Язык распознавания                                      |
| `ENABLE_DIARIZATION` | `true`                                                | `false` — пропустить NeMo (один спикер, без GPU)        |
| `MAX_SPEAKERS`       | `10`                                                  | Максимум спикеров при кластеризации                     |
| `DIARIZATION_MODEL`  | `nvidia/speakerverification_en_titanet_large`         | Модель эмбеддингов спикеров                             |
| `DIARIZATION_DEVICE` | `auto`                                                | `auto`, `cuda` или `cpu`; фактическое устройство видно в логах/UI |
| `OLLAMA_MODEL`       | `gemma3:12b`                                           | Модель Ollama для саммари                               |
| `OLLAMA_HOST`        | `http://localhost:11434`                              | Адрес сервера Ollama                                    |
| `API_HOST`/`API_PORT`| `0.0.0.0` / `8000`                                    | Хост и порт FastAPI                                     |
| `CORS_ORIGINS`       | `*`                                                   | Разрешённые origin'ы (через запятую)                    |

---

## API

| Метод | Путь                | Описание                                              |
| ----- | ------------------- | ----------------------------------------------------- |
| GET   | `/`                 | Фронтенд (панель управления)                          |
| GET   | `/health`           | Проверка готовности и текущей конфигурации            |
| GET   | `/prompts`          | Список доступных промптов для UI                      |
| GET   | `/report/summary`   | Агрегированные MVP-метрики для защиты                 |
| GET   | `/report/runs`      | Последние обработки в JSON                            |
| GET   | `/report/export.csv`| CSV-выгрузка обработок для Excel/Google Sheets        |
| POST  | `/upload`           | Загрузка аудио (`file`, `prompt`) → `task_id`         |
| POST  | `/summarize-file`   | Саммари из готового транскрипта (`transcript_file`, `prompt`) |
| GET   | `/status/{task_id}` | Статус/прогресс задачи и результат                    |
| POST  | `/cancel/{task_id}` | Отмена задачи                                         |

---

## Тесты

```bash
pytest -q
```

Smoke-тесты не требуют GPU, Redis или Ollama: проверяют загрузку конфигурации,
рендер промптов (включая защиту от падения real-time промпта), построение
Markdown для Obsidian, ленивый импорт воркера и эндпоинты `/health`, `/prompts`.

В репозиторий добавлен GitHub Actions workflow `.github/workflows/tests.yml`.
Он ставит `requirements-min.txt`, отключает диаризацию и запускает `pytest -q`,
поэтому должен проходить на GitHub без Redis, Ollama, CUDA и NeMo.

---

## Сбор данных

После каждой успешной или ошибочной обработки сервис добавляет строку в
`reports/runs.jsonl`. В запись входят:

- длительность аудио и время обработки;
- `processing_ratio = processing_time_sec / audio_duration_sec`;
- число сегментов, спикеров, silhouette, доля неизвестных спикеров;
- модель Whisper, фактический `Whisper device`, `NeMo device`, CUDA/PyTorch;
- успешность сохранения в Obsidian;
- ошибка, если задача завершилась неуспешно.

Открыть сводку можно прямо в UI в блоке «Отчёт MVP» или через API:

```bash
curl http://localhost:8000/report/summary
curl http://localhost:8000/report/runs
curl -o voiceflow_report.csv http://localhost:8000/report/export.csv
```

---
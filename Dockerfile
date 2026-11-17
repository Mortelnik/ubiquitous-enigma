# syntax=docker/dockerfile:1
FROM python:3.12-slim-trixie

COPY --from=ghcr.io/astral-sh/uv:0.11.14 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      ffmpeg \
      patchelf \
      libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# API/bot use the base dependencies; worker adds audio by default.
# Example for speaker separation: UV_EXTRAS="audio diarization".
ARG UV_EXTRAS=audio
# Some optional NeMo dependencies are built from source.
RUN set -eu; \
    for extra in $UV_EXTRAS; do \
      if [ "$extra" = "diarization" ]; then \
        apt-get update \
        && apt-get install -y --no-install-recommends build-essential git \
        && rm -rf /var/lib/apt/lists/*; \
      fi; \
    done
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    set -eu; \
    set --; \
    for extra in $UV_EXTRAS; do set -- "$@" --extra "$extra"; done; \
    uv sync --locked --no-dev --no-install-project "$@"

# Check native libraries on the actual image/glibc before accepting a build.
# Old CTranslate2 wheels have an incorrect executable-stack flag (upstream #1849).
COPY app/audio_runtime.py /tmp/audio_runtime.py
RUN python /tmp/audio_runtime.py && rm /tmp/audio_runtime.py

COPY app ./app
COPY bot ./bot
COPY static ./static
RUN mkdir -p uploads obsidian_vault reports

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

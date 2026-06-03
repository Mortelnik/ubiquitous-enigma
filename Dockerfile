FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      build-essential \
      curl \
      ffmpeg \
      git \
      libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

# По умолчанию ставим полный requirements.txt. Для лёгкого dev/CI образа:
# docker compose build --build-arg REQUIREMENTS_FILE=requirements-min.txt
ARG REQUIREMENTS_FILE=requirements.txt
COPY requirements.txt requirements-min.txt ./
RUN uv pip install --system -r "${REQUIREMENTS_FILE}"

COPY app ./app
COPY static ./static
COPY tests ./tests
COPY run_api.sh run_worker.sh README.md pyproject.toml ./
RUN chmod +x run_api.sh run_worker.sh \
    && mkdir -p uploads obsidian_vault reports

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

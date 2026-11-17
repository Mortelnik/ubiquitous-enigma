# Obsidian, CUDA и результаты Telegram в Docker Desktop

Все команды выполняются из корня проекта в PowerShell. GPU требует Docker Desktop
с Linux-контейнерами и WSL2, NVIDIA GPU и драйвер с поддержкой WSL.

Контейнерный Ollama доступен сервисам по `http://ollama:11434` без публикации порта
на Windows. Это позволяет запускать проект при установленном Windows Ollama,
который занимает локальный порт 11434. После изменения Compose выполните `up -d`,
чтобы пересоздать контейнер: кнопка Start не применяет новые настройки портов.

## Открыть заметки в Windows

В базовом Compose заметки находятся в Docker volume `obsidian_data`, а не в папке
Windows. Сначала скопируйте существующие заметки, если ещё не сделали этого:

```powershell
New-Item -ItemType Directory -Force .\obsidian_vault | Out-Null
docker compose --env-file .env.docker cp worker:/app/obsidian_vault/. .\obsidian_vault
```

Копирование в непустую папку может заменить одноимённые файлы: сначала сохраните
их копию.

В Obsidian выберите «Открыть папку как хранилище» и укажите папку
`obsidian_vault` внутри вашего клона репозитория. Заметки находятся в подпапке «Встречи».
Для сохранения новых заметок прямо в эту папку запускайте сервисы с override:

```powershell
docker compose --env-file .env.docker -f docker-compose.yml -f docker-compose.storage.yml --profile bot up -d --build --force-recreate
```

Override заменяет только mount `/app/obsidian_vault` у API и worker. Старый том
сохраняется. Не используйте `down -v`, если хотите сохранить Docker-данные.

## Whisper на GPU вместе с локальным Obsidian

```powershell
docker compose --env-file .env.docker -f docker-compose.yml -f docker-compose.gpu.yml -f docker-compose.storage.yml --profile bot up -d --build --force-recreate
```

GPU override устанавливает дополнение uv `cuda`, включает доступ к GPU у worker
и Ollama, задаёт Whisper `cuda / float16` и пути CUDA-библиотек. Для закреплённого
CTranslate2 4.4 используются CUDA 12/cuDNN 8. PyTorch для Faster-Whisper не нужен.
TF-IDF/LogisticRegression продолжает работать на CPU. Диаризация по умолчанию
выключена; NeMo имеет отдельные зависимости и настройки.

Проверка GPU и динамических библиотек без скачивания модели:

```powershell
docker compose --env-file .env.docker -f docker-compose.yml -f docker-compose.gpu.yml -f docker-compose.storage.yml exec worker python -m app.gpu_check
```

Успешная диагностика выводит `CUDA runtime check OK`. После этого отправьте новое
аудио и проверьте метрики `Whisper device: cuda / float16`, `Fallback Whisper: нет`.
Проверка окружения не заменяет запуск настоящего распознавания на GPU.

Если GPU или библиотеки недоступны, при `WHISPER_FALLBACK_TO_CPU=true` обработка
может перейти на CPU/int8. Смотрите фактическое устройство и причину в журнале:

```powershell
docker compose --env-file .env.docker -f docker-compose.yml -f docker-compose.gpu.yml -f docker-compose.storage.yml --profile bot logs -f --tail 100 worker bot
```

Используйте те же файлы `-f` при следующих пересозданиях контейнеров. Запуск только
с базовым Compose вернёт CPU и внутренний том Obsidian. Для helper-скрипта есть
эквивалент `-Gpu -LocalVault -Bot`.

## Стиль и ответ Telegram

В `.env.docker`:

```dotenv
BOT_SEND_RESULTS=true
```

В боте через `/prompt` выберите классификатор стиля. Он возвращает категорию
«разговорный» или «официальный», оценку модели, длительность, число сегментов,
метрики Whisper, информацию о диаризации, время обработки, фрагменты текста
и имена сохранённых заметок. Оценка классификатора не является измеренной точностью.
Полные тексты остаются в файлах Obsidian; Telegram получает короткие фрагменты.
При `false` бот отвечает только о локальном сохранении.

В веб-интерфейсе результат показывается в отдельном блоке «Стиль речи». Если
выбрана другая роль, классификация не запускается. Требуется обученный файл
`models/style_classifier.joblib`; см. `DZ_03/RUNBOOK.md`.

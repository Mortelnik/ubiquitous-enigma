# VoiceFlow AI (см ReadME ветки vanilla)

добавлен бинарный классификатор стиля речи для телеграм-бота: определяет, является ли голосовое
сообщение разговорным или официально-деловым, и выбирает промпт для саммари.

**Кейс 1 — Классификация аудиосообщений на «разговорные» vs «официальные»**
Команда 7: Шелест М.А., Иванова Ю.С., Романенко Д.Н., Ишбулатова С.С., Буянов Е.Э

## О проекте

Помимо загрузки через бразуер со своего диска или сетевой папки, пользователь получил можность отправлять сообщения в тг-бота, который

1. Отправляет аудио на Whisper, получает транскрипт
2. Классифицирует стиль: 0 — разговорный, 1 — официально-деловой
3. Выбирает промпт для саммари в зависимости от стиля
4. Отправляет аудио в Ollama (Qwen) с выбранным промптом
5. Возвращает пользователю краткое содержание

Классификатор реализован классическим методом: TF-IDF + LogisticRegression с ансамблем CNN.

## Структура репозитория

```
├── .gitignore                  # исключения для GitHub
├── README.md                   # этот файл
├── requirements.txt            # зависимости Python
├── bot/                        # телеграм-бот (aiogram)
│   ├── main.py
│   ├── api_client.py
│   ├── config.py
│   └── Dockerfile
├── api/                        # FastAPI для транскрипции
│   ├── main.py
│   └── Dockerfile
├── worker/                     # воркер для Ollama
│   ├── main.py
│   └── Dockerfile
├── scripts/                    # скрипты для данных
│   ├── download_balalaika.py   # скачивание датасета
│   ├── prepare_dataset.py      # подготовка train/val/test
│   ├── download_golos_farfield.py
│   └── prepare_duma.py
├── dz03/                        # ДЗ 3: классификация стиля
│   ├── model.ipynb             # ноутбук с экспериментами
│   ├── prepare_dataset.py      # скрипт подготовки данных
│   ├── отчет.md                # отчёт (markdown)
│   └── data/                   # данные (в .gitignore)
├── docker-compose.yml
├── docker-compose.gpu.yml
└── .env.docker.example         # пример переменных окружения
```

## Запуск

### Требования

- Python 3.12
- 16+ ГБ ОЗУ для Whisper large-v3
- 22+ ГБ VRAM для Ollama с 27B-моделью (или меньше для 8B)
- Docker Desktop для контейнерной сборки

### Установка зависимостей

```bash
# создаём окружение
python -m venv .venv312
source .venv312/bin/activate    # Linux/Mac
# .venv312\Scripts\activate    # Windows

pip install -r requirements.txt
```

### Скачивание данных

Датасет OpenSTT Balalaika (~150 МБ текстовых аннотаций, полные аудио ~10,6 ГБ):

```bash
python scripts/download_balalaika.py --out data/raw
```

Датасет Golos farfield (для расширения класса «разговорный»):

```bash
python scripts/download_golos_farfield.py --out data/golos_farfield
```

Корпус стенограмм Госдумы для синтеза аудио (класс «официальный»):

```bash
python scripts/prepare_duma.py --raw data/duma/raw --out data/duma
python scripts/tts_synth.py --phrases data/duma/tts_phrases.csv --out data/duma/tts_audio
```

### Подготовка данных

```bash
python hw3/prepare_dataset.py
```

Создаёт `hw3/data/processed/{dataset,train,val,test}.parquet` и `dataset_stats.json`.

### Обучение классификатора

`model.ipynb`

### Запуск через Docker

```bash
# с GPU
docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile bot up -d

# без GPU
docker compose --profile bot up -d
```

Переменные окружения: скопируй `.env.docker.example` в `.env.docker` и заполни
токены Telegram, Ollama, Redis.

## Результаты (ДЗ 3)

| Модель | Accuracy (val) | Accuracy (test) |
|---|---|---|
| LogisticRegression (TF-IDF) | 0,9453 | — |
| CNN | 0,9384 | — |
| **Ансамбль LR (0,45) + CNN** | **0,9599** | **0,9581** |

## Датасеты

| Датасет | Часы | Записей | Класс | Ссылка |
|---|---|---|---|---|
| OpenSTT Balalaika | 431,4 | 450 415 | оба | [HF](https://huggingface.co/datasets/lab260/openstt_balalaika) |
| Golos farfield | 132,4 | 12 419 | 0 (разговорный) | [HF](https://huggingface.co/datasets/bond005/sberdevices_golos_100h_farfield) |
| Корпус Госдумы (текст) | — | 132 291 | 1 (официальный) | [GitHub](https://github.com/Chepenkoroman/duma_analysis) |

## Ограничения и известные проблемы

- Разметка по доменам — прокси-разметка (слабые метки), не ручная. 458 ошибок
  вида 1→0 — это реплики в аудиокнигах, по стилю действительно разговорные.
- Класс 0 в тесте представлен только 334 записями — следствие маленького размера
  домена телефонных звонков в Balalaika.

## Дополнительные материалы

- Отчёт ДЗ 3: `dz03/отчет.md`
- Ноутбук экспериментов: `dz03/model.ipynb`
- Пайплайн телеграм-бота: `bot/`, `api/`, `worker/`

"""Подготовка датасета для ДЗ 3: классификация стиля речи (разговорный vs официальный).

Домены Balalaika размечаются по стилю:
- класс 0 (разговорный): телефонные разговоры, радио, YouTube, сериалы
- класс 1 (официальный): публичные выступления, лекции, аудиокниги
- TTS-адреса (робот) исключаются как неестественная речь
"""

import pandas as pd
import numpy as np
import re
import json
from pathlib import Path

RAW_DIR = Path("../data/raw")
OUT_DIR = Path("../data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DOMAIN_LABELS = {
    "asr_public_phone_calls_1": 0,
    "asr_public_phone_calls_2": 0,
    "public_series_1": 0,
    "public_youtube1120": 0,
    "public_youtube1120_hq": 0,
    "public_youtube700": 0,
    "radio-v4": 0,
    "radio_2": 0,
    "radio_v4_add": 0,
    "public_lecture_1": 1,
    "public_speech": 1,
    "asr_public_stories_1": 1,
    "asr_public_stories_2": 1,
    "private_buriy_audiobooks_2": 1,
    "tts_russian_addresses_rhvoice_4voices": None,  # робот — исключаем
}

CLASS_NAMES = {0: "разговорный", 1: "официальный"}


def load_and_label():
    frames = []
    for pq in sorted(RAW_DIR.glob("*.parquet")):
        d = pd.read_parquet(pq)
        d["domain"] = pq.stem.replace("_balalaika", "")
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["label"] = df["domain"].map(DOMAIN_LABELS)
    return df


def clean_text(t: str) -> str:
    """Нормализация текста: нижний регистр, только буквы/цифры/пробелы."""
    if not isinstance(t, str):
        return ""
    t = re.sub(r"[^\w\s]", " ", t, flags=re.UNICODE)
    t = re.sub(r"\s+", " ", t)
    return t.lower().strip()


def main():
    df = load_and_label()
    n0 = (df.label == 0).sum()
    n1 = (df.label == 1).sum()
    print(f"Всего записей: {len(df)}")
    print(f"Класс 0 (разговорный): {n0}")
    print(f"Класс 1 (официальный): {n1}")
    print(f"Исключено (робот/прочее): {len(df) - n0 - n1}")

    # Базовая фильтрация качества: MOS >= 4.0, один диктор, непустой текст
    df["text"] = df["punct"].fillna("")
    df["text_clean"] = df["text"].apply(clean_text)
    df["n_words"] = df["text_clean"].str.split().str.len()

    keep = (
        df.label.notna()
        & (df.mos_pred >= 4.0)
        & (df.is_single_speaker == True)
        & (df.n_words >= 3)
    )
    filtered = df[keep].copy()
    print(f"\nПосле фильтрации (MOS>=4, single speaker, >=3 слов): {len(filtered)}")
    print(f"Класс 0: {(filtered.label==0).sum()}, класс 1: {(filtered.label==1).sum()}")

    # Ограничим максимальную длину и уберём дубликаты
    filtered = filtered[filtered.n_words <= 300].drop_duplicates("text_clean")
    drop_cols = [c for c in ("end", "start", "playlist_id", "podcast_id", "accent", "rover", "phonemes", "model") if c in filtered.columns]
    filtered = filtered.drop(columns=drop_cols)
    print(f"После ограничения длины и дедупликации: {len(filtered)}")

    # Сохраняем полный датасет и stratified split
    filtered.to_parquet(OUT_DIR / "dataset.parquet", index=False)

    rng = np.random.RandomState(42)
    for split_name, frac in [("train", 0.7), ("val", 0.15), ("test", 0.15)]:
        parts = []
        for label in (0, 1):
            sub = filtered[filtered.label == label]
            n = int(len(sub) * frac)
            parts.append(sub.sample(n=n, random_state=rng))
        pd.concat(parts).to_parquet(OUT_DIR / f"{split_name}.parquet", index=False)

    for split_name in ("train", "val", "test"):
        d = pd.read_parquet(OUT_DIR / f"{split_name}.parquet")
        print(f"{split_name}: {len(d)} (0: {(d.label==0).sum()}, 1: {(d.label==1).sum()})")

    # Сохраняем описание датасета
    desc = {
        "total": int(len(df)),
        "after_filter": int(len(filtered)),
        "class_0": int((filtered.label == 0).sum()),
        "class_1": int((filtered.label == 1).sum()),
        "mos_median": float(filtered.mos_pred.median()),
        "mos_min": float(filtered.mos_pred.min()),
        "mos_max": float(filtered.mos_pred.max()),
        "duration_median_sec": float(filtered.duration.median()),
        "words_median": float(filtered.n_words.median()),
        "domains": filtered.domain.value_counts().to_dict(),
    }
    with open(OUT_DIR / "dataset_stats.json", "w") as f:
        json.dump(desc, f, ensure_ascii=False, indent=2)
    print(json.dumps(desc, ensure_ascii=False, indent=2)[:2000])


if __name__ == "__main__":
    main()

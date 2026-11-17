"""Prepare disjoint style datasets; paths are independent of the working directory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from app.classifier import preprocess

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data/raw"
OUT_DIR = ROOT / "data/processed"

# Weak domain labels only. Mixed domains and fiction are deliberately excluded.
DOMAIN_LABELS = {
    "asr_public_phone_calls_1": 0,
    "asr_public_phone_calls_2": 0,
    "public_series_1": 0,
    "public_lecture_1": 1,
    "public_speech": 1,
}
GROUP_COLUMNS = ("source_id", "recording_id", "speaker_id", "playlist_id", "podcast_id", "group_id")
clean_text = preprocess


def load_and_label(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    frames = []
    for path in sorted(raw_dir.glob("*.parquet")):
        frame = pd.read_parquet(path)
        if "domain" not in frame:
            frame["domain"] = path.stem.removesuffix("_balalaika")
        if "label" not in frame:
            frame["label"] = frame.domain.map(DOMAIN_LABELS)
            frame["label_source"] = "weak_domain"
        else:
            frame["label_source"] = frame.get("label_source", "provided")
        frames.append(frame)
    if not frames:
        raise ValueError(f"В {raw_dir} нет parquet-файлов")
    return pd.concat(frames, ignore_index=True)


def prepare_rows(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "text" not in frame:
        frame["text"] = frame["punct"]
    frame["text"] = frame.text.fillna(frame.get("punct", pd.Series("", index=frame.index))).fillna("")
    frame["text_clean"] = frame.text.map(preprocess)
    frame["n_words"] = frame.text_clean.str.split().str.len()
    frame = frame[frame.label.notna()].copy()
    if not frame.label.isin([0, 1]).all():
        raise ValueError("Метки должны быть 0 (разговорный) или 1 (официальный)")
    frame["label"] = frame.label.astype(int)
    keep = frame.n_words.between(3, 300)
    if "mos_pred" in frame:
        keep &= frame.mos_pred.ge(4.0) | frame.mos_pred.isna()
    if "is_single_speaker" in frame:
        keep &= frame.is_single_speaker.eq(True) | frame.is_single_speaker.isna()
    frame = frame[keep].reset_index(drop=True)
    if frame.groupby("text_clean").label.nunique().gt(1).any():
        raise ValueError("У одинаковых текстов противоречащие метки; исправьте разметку")
    # Union shared sources before deduplication so duplicates cannot sever groups.
    parent = list(range(len(frame)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for column in ("text_clean", "filepath", *GROUP_COLUMNS):
        if column not in frame:
            continue
        seen = {}
        domains = frame.get("domain", pd.Series("", index=frame.index))
        for i, (value, domain) in enumerate(zip(frame[column], domains)):
            if pd.isna(value) or not str(value).strip():
                continue
            namespace = str(domain) if column in {"playlist_id", "podcast_id"} else ""
            key = (namespace, str(value))
            if key in seen:
                parent[find(i)] = find(seen[key])
            else:
                seen[key] = i
    members = {}
    for i, text in enumerate(frame.text_clean):
        members.setdefault(find(i), []).append(text)
    group_ids = {key: hashlib.sha256("\n".join(sorted(set(texts))).encode()).hexdigest()
                 for key, texts in members.items()}
    frame["group_id"] = [group_ids[find(i)] for i in range(len(frame))]
    return frame.drop_duplicates("text_clean").reset_index(drop=True)


def _split(frame: pd.DataFrame, fraction: float, seed: int):
    best, score = None, float("inf")
    splitter = GroupShuffleSplit(n_splits=100, test_size=fraction, random_state=seed)
    for left, right in splitter.split(frame, frame.label, groups=frame.group_id):
        a, b = frame.iloc[left], frame.iloc[right]
        if a.label.nunique() != 2 or b.label.nunique() != 2:
            continue
        error = abs(len(b) / len(frame) - fraction) + abs(a.label.mean() - b.label.mean())
        if error < score:
            best, score = (a, b), error
    if best is None:
        raise ValueError("Недостаточно независимых групп обоих классов для разбиения")
    return best


def assert_disjoint(splits: dict[str, pd.DataFrame]) -> None:
    for name, frame in splits.items():
        if frame.empty or frame.label.nunique() != 2:
            raise ValueError(f"{name}: нужны непустые данные обоих классов")
        if not frame.label.isin([0, 1]).all():
            raise ValueError(f"{name}: недопустимые метки")
        if not frame.text.fillna("").map(preprocess).str.len().gt(0).all():
            raise ValueError(f"{name}: пустой нормализованный текст")
    names = list(splits)
    for i, name in enumerate(names):
        for other in names[i + 1:]:
            a, b = splits[name], splits[other]
            if set(a.text.map(preprocess)) & set(b.text.map(preprocess)):
                raise ValueError(f"Пересечение текстов: {name}/{other}")
            for column in ("filepath", *GROUP_COLUMNS):
                if column in a and column in b:
                    def keys(frame):
                        valid = frame[column].notna() & frame[column].astype(str).str.strip().ne("")
                        values = frame.loc[valid, column].astype(str)
                        if column in {"playlist_id", "podcast_id"} and "domain" in frame:
                            values = frame.loc[valid, "domain"].astype(str) + ":" + values
                        return set(values)
                    if keys(a) & keys(b):
                        raise ValueError(f"Пересечение {column}: {name}/{other}")


def split_dataset(frame: pd.DataFrame, seed: int = 42) -> dict[str, pd.DataFrame]:
    train, held_out = _split(frame, 0.3, seed)
    val, test = _split(held_out, 0.5, seed + 1)
    splits = {"train": train, "val": val, "test": test}
    assert_disjoint(splits)
    return splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW_DIR)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    frame = prepare_rows(load_and_label(args.raw))
    splits = split_dataset(frame)
    args.out.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out / "dataset.parquet", index=False)
    for name, data in splits.items():
        data.to_parquet(args.out / f"{name}.parquet", index=False)
    stats = {name: {"rows": len(data), "classes": data.label.value_counts().to_dict(),
                    "groups": data.group_id.nunique()} for name, data in splits.items()}
    (args.out / "dataset_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

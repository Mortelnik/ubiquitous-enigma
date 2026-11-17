"""Download the public text annotations used by the style classifier (no audio)."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile

from DZ_03.model_pre import DOMAIN_LABELS, RAW_DIR

REPO_ID = "lab260/openstt_balalaika"


def download_dataset(out_dir: Path, *, revision: str = "main", dry_run: bool = False,
                     api=None, download_file=None) -> dict:
    if api is None or (download_file is None and not dry_run):
        from huggingface_hub import HfApi, hf_hub_download
        api = api or HfApi()
        download_file = download_file or hf_hub_download

    # Resolve a mutable branch once; every file in this run uses the same commit.
    info = api.dataset_info(REPO_ID, revision=revision, files_metadata=True)
    if not info.sha:
        raise ValueError("Источник не вернул номер версии датасета")
    remote = {item.rfilename: item for item in info.siblings}
    names = sorted(f"{domain}_balalaika.parquet" for domain in DOMAIN_LABELS)
    missing = set(names) - remote.keys()
    if missing:
        raise ValueError(f"В датасете отсутствуют ожидаемые файлы: {', '.join(sorted(missing))}")
    manifest = {
        "repo_id": REPO_ID,
        "revision": info.sha,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "files": [{"name": name, "bytes": remote[name].size} for name in names],
    }
    if dry_run:
        return manifest

    out_dir.mkdir(parents=True, exist_ok=True)
    # Existing unrelated Parquet files must not silently join a different dataset.
    unexpected = {path.name for path in out_dir.glob("*.parquet")} - set(names)
    if unexpected:
        raise ValueError("Выберите отдельный каталог --out: в нём уже есть другие Parquet-файлы")
    for name in names:
        print(f"Загрузка: {name}", flush=True)
        download_file(repo_id=REPO_ID, repo_type="dataset", filename=name,
                      revision=info.sha, local_dir=str(out_dir))
    # Record provenance only after all downloads complete successfully.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=out_dir,
                                     suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    try:
        os.replace(temporary, out_dir / "download_manifest.json")
    finally:
        temporary.unlink(missing_ok=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=RAW_DIR)
    parser.add_argument("--revision", default="main", help="Commit/tag/branch of the public dataset")
    parser.add_argument("--dry-run", action="store_true", help="List files without downloading them")
    args = parser.parse_args()
    try:
        manifest = download_dataset(args.out, revision=args.revision, dry_run=args.dry_run)
    except Exception as exc:
        parser.exit(1, f"Не удалось скачать датасет: {exc}\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

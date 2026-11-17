from types import SimpleNamespace

import pytest

from scripts.download_balalaika import REPO_ID, download_dataset
from DZ_03.model_pre import DOMAIN_LABELS


def dataset_api():
    entries = [SimpleNamespace(rfilename=f"{domain}_balalaika.parquet", size=100)
               for domain in DOMAIN_LABELS]
    # Archives and unrelated annotations must never be downloaded.
    entries += [SimpleNamespace(rfilename="public_speech.tar.gz", size=999999),
                SimpleNamespace(rfilename="radio_2_balalaika.parquet", size=100)]
    return SimpleNamespace(dataset_info=lambda *args, **kwargs:
                           SimpleNamespace(sha="fixed-commit", siblings=entries))


def test_annotation_download_pins_revision_and_skips_audio(tmp_path):
    calls = []
    manifest = download_dataset(tmp_path, api=dataset_api(),
                                download_file=lambda **kwargs: calls.append(kwargs))
    assert len(calls) == len(DOMAIN_LABELS)
    assert all(call["revision"] == "fixed-commit" and call["repo_id"] == REPO_ID for call in calls)
    assert {call["filename"] for call in calls} == {item["name"] for item in manifest["files"]}
    assert (tmp_path / "download_manifest.json").is_file()


def test_dry_run_does_not_create_download_directory(tmp_path):
    output = tmp_path / "not-created"
    result = download_dataset(output, dry_run=True, api=dataset_api())
    assert len(result["files"]) == len(DOMAIN_LABELS)
    assert not output.exists()


def test_incomplete_download_does_not_mark_success(tmp_path):
    def fail(**kwargs):
        raise OSError("interrupted")
    with pytest.raises(OSError):
        download_dataset(tmp_path, api=dataset_api(), download_file=fail)
    assert not (tmp_path / "download_manifest.json").exists()


def test_unrelated_local_dataset_is_preserved(tmp_path):
    unrelated = tmp_path / "private.parquet"
    unrelated.write_bytes(b"local data")
    with pytest.raises(ValueError, match="отдельный каталог"):
        download_dataset(tmp_path, api=dataset_api(), download_file=lambda **kwargs: pytest.fail())
    assert unrelated.read_bytes() == b"local data"

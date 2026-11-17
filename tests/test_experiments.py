import json

import joblib
import pandas as pd
import pytest

from DZ_03.compare_experiments import compare, select_candidate
from DZ_03.train_classifier import augmented_train, load_splits
from scripts.tts_synth import select_phrases


def experiment_data(tmp_path):
    train = pd.DataFrame([
        {"text": f"{prefix} {suffix}", "label": label, "group_id": f"train-{label}-{i}"}
        for label, prefix in [(0, "привет друг давай гулять"), (1, "уважаемые коллеги прошу направить отчёт")]
        for i, suffix in enumerate(["сегодня", "завтра", "утром", "вечером"])
    ])
    held_out = {
        "val": ["привет дружище пойдём гулять в парк", "уважаемые коллеги направьте официальный документ"],
        "test": ["эй друг давай прогуляемся после школы", "коллеги прошу подготовить письменный официальный ответ"],
    }
    train.to_parquet(tmp_path / "train.parquet", index=False)
    for name, texts in held_out.items():
        pd.DataFrame({"text": texts, "label": [0, 1], "group_id": [f"{name}-0", f"{name}-1"]}).to_parquet(
            tmp_path / f"{name}.parquet", index=False)
    synthetic = train.iloc[[0, 4]].copy()
    synthetic["asr_text"] = ["привет друг пойдём гулять сегодня", "уважаемые коллеги направьте отчёт сегодня"]
    manifest = tmp_path / "manifest.csv"
    synthetic.to_csv(manifest, index=False)
    return manifest


def test_comparison_preserves_splits_and_exports_deployable_candidate(tmp_path):
    manifest = experiment_data(tmp_path)
    before = {name: (tmp_path / f"{name}.parquet").read_bytes() for name in ("train", "val", "test")}
    runtime = tmp_path / "runtime/style.joblib"
    result = compare(tmp_path, manifest, tmp_path / "experiments", runtime)
    assert result["selection_metric"] == "validation_macro_f1"
    assert result["metadata"]["real_plus_tts"]["synthetic_rows"] == 2
    assert result["metadata"]["real_only"]["synthetic_rows"] == 0
    assert result["metadata"]["real_only"]["split_fingerprints"] == result["metadata"]["real_plus_tts"]["split_fingerprints"]
    for name, contents in before.items():
        assert (tmp_path / f"{name}.parquet").read_bytes() == contents
    for name in ("val", "test"):
        assert result["metrics"]["real_only"][name]["baseline"] == result["metrics"]["real_plus_tts"][name]["baseline"]
    assert joblib.load(runtime)["metadata"]["experiment"] == result["selected"]
    assert json.loads(runtime.with_suffix(".metrics.json").read_text()) == result["metrics"][result["selected"]]
    assert (tmp_path / "experiments/comparison.md").is_file()


def test_selection_ignores_test_and_prefers_real_only_on_tie():
    def metric(value):
        return {"model": {"macro_f1": value}}
    experiments = {"real_only": {"val": metric(.8), "test": metric(.1)},
                   "real_plus_tts": {"val": metric(.7), "test": metric(.99)}}
    assert select_candidate(experiments) == "real_only"
    experiments["real_plus_tts"]["val"] = metric(.8)
    assert select_candidate(experiments) == "real_only"
    experiments["real_plus_tts"]["val"] = metric(.9)
    assert select_candidate(experiments) == "real_plus_tts"


@pytest.mark.parametrize("violation", ["source", "label", "asr", "group", "empty_asr"])
def test_tts_invalid_or_held_out_rows_are_rejected(tmp_path, violation):
    manifest = experiment_data(tmp_path)
    splits = load_splits(tmp_path)
    synthetic = pd.read_csv(manifest)
    if violation == "source":
        synthetic.loc[0, "text"] = splits["test"].iloc[0].text
    elif violation == "label":
        synthetic.loc[0, "label"] = 1
    elif violation == "asr":
        synthetic.loc[0, "asr_text"] = splits["val"].iloc[0].text
    elif violation == "group":
        synthetic.loc[0, "group_id"] = "another-recording"
    else:
        synthetic.loc[0, "asr_text"] = "123 ..."
    synthetic.to_csv(manifest, index=False)
    with pytest.raises(ValueError):
        augmented_train(splits, manifest)


def test_tts_sampling_is_repeatable_and_includes_both_classes(tmp_path):
    experiment_data(tmp_path)
    train = load_splits(tmp_path)["train"]
    selected = select_phrases(train, max_per_class=2)
    assert selected.label.value_counts().to_dict() == {0: 2, 1: 2}
    assert selected.equals(select_phrases(train, max_per_class=2))
    assert set(selected.group_id).issubset(train.group_id)

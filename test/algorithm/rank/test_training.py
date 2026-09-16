import pandas as pd
import pytest

from algorithm.rank import training as trainer
from algorithm.rank.training import TrainingRequest


def test_materialized_rows_are_aligned_and_reject_incomplete_samples():
    events = pd.DataFrame(
        [
            {"_sample_id": "later", "time": 20},
            {"_sample_id": "earlier", "time": 10},
        ]
    )
    users = pd.DataFrame(
        [
            {"_sample_id": "earlier", "id": "u", "city": "old"},
            {"_sample_id": "later", "id": "u", "city": "new"},
        ]
    )
    items = pd.DataFrame(
        [
            {"_sample_id": "earlier", "id": "i", "weight": 1},
            {"_sample_id": "later", "id": "i", "weight": 2},
        ]
    )

    aligned_users, aligned_items = trainer._align_materialized_rows(
        events, users, items
    )
    assert aligned_users["city"].tolist() == ["new", "old"]
    assert aligned_items["weight"].tolist() == [2, 1]
    assert (
        trainer._latest_feature_rows(events, aligned_users).iloc[0]["city"]
        == "new"
    )

    with pytest.raises(ValueError, match="do not match"):
        trainer._align_materialized_rows(events, users.iloc[:1], items)


def test_materialized_rows_reject_duplicate_sample_identity():
    events = pd.DataFrame(
        [
            {"_sample_id": "same", "time": 10},
            {"_sample_id": "same", "time": 20},
        ]
    )
    rows = pd.DataFrame(
        [
            {"_sample_id": "same", "id": "u"},
            {"_sample_id": "other", "id": "u"},
        ]
    )
    with pytest.raises(ValueError, match="duplicate"):
        trainer._align_materialized_rows(events, rows, rows)


def test_train_request_requires_auditable_feature_cutoff():
    with pytest.raises(ValueError):
        TrainingRequest(
            scene="home",
            version="20260824-r001",
            business_date="2026-08-24",
            revision="r001",
            dataset_dir="/models/training/home/run",
        )

    request = TrainingRequest(
        scene="home",
        version="20260824-r001",
        business_date="2026-08-24",
        revision="r001",
        dataset_dir="/models/training/home/run",
        feature_cutoff_time=123,
        label_observation_cutoff=456,
        input_label_count=10,
        constructed_label_count=8,
    )
    assert request.feature_cutoff_time == 123
    assert request.constructed_label_count == 8


def training_request(dataset):
    return TrainingRequest(
        scene="global",
        version="20260916-r001",
        business_date="2026-09-16",
        revision="r001",
        dataset_dir=str(dataset),
        feature_cutoff_time=123,
        label_observation_cutoff=456,
        input_label_count=10,
    )


def test_failed_training_cleans_staging_without_publishing(tmp_path):
    dataset = tmp_path / "training" / "run"
    dataset.mkdir(parents=True)
    releases = tmp_path / "releases"
    with pytest.raises(FileNotFoundError):
        trainer.train_release(training_request(dataset), releases)
    assert list((releases / "item" / "global").iterdir()) == []
    assert dataset.exists()


def test_existing_release_is_immutable(tmp_path):
    dataset = tmp_path / "training" / "run"
    dataset.mkdir(parents=True)
    releases = tmp_path / "releases"
    release = releases / "item" / "global" / "20260916-r001"
    release.mkdir(parents=True)
    manifest = release / "manifest.json"
    manifest.write_text('{"version": "retained"}')
    with pytest.raises(ValueError, match="already exists"):
        trainer.train_release(training_request(dataset), releases)
    assert manifest.read_text() == '{"version": "retained"}'
    assert dataset.exists()

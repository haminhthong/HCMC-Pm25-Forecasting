import json

import joblib

from src.artifacts.loader import load_artifact_bundle
from src.artifacts.writer import save_artifacts


def test_artifacts_use_one_current_bundle(tmp_path):
    model = {"name": "test-model"}
    config = {
        "data": {
            "timestamp_column": "timestamp",
            "station_column": "station",
            "target_column": "PM2.5",
        },
        "artifacts": {"directory": str(tmp_path)},
    }
    metadata = {"features": ["PM2.5_lag_1"], "forecast_strategy": "persistence"}
    evaluation = {"model_selection": {"forecast_strategy": "persistence"}}

    save_artifacts(model, metadata, evaluation, config)

    assert (tmp_path / "model.joblib").is_file()
    assert (tmp_path / "metadata.json").is_file()
    assert (tmp_path / "evaluation.json").is_file()
    assert (tmp_path / "feature_schema.json").is_file()
    assert not (tmp_path / "active_release.json").exists()
    assert not (tmp_path / "models").exists()
    assert joblib.load(tmp_path / "model.joblib") == model
    assert json.loads((tmp_path / "metadata.json").read_text()) == metadata


def test_loader_rejects_missing_current_model(tmp_path):
    try:
        load_artifact_bundle(tmp_path)
    except FileNotFoundError as error:
        assert "model.joblib" in str(error)
    else:
        raise AssertionError("Loader phải báo lỗi khi thiếu model hiện tại.")

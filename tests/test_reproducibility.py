import pytest

from src.pipeline import run_train_pipeline


def test_training_is_reproducible():
    """Kiểm tra hai lần huấn luyện độc lập sinh ra cùng kết quả metric trên cùng seed."""
    res1 = run_train_pipeline("configs/config.yaml", persist_artifacts=False)
    res2 = run_train_pipeline("configs/config.yaml", persist_artifacts=False)

    assert res1["metrics"]["mae"] == pytest.approx(res2["metrics"]["mae"])
    assert res1["metrics"]["rmse"] == pytest.approx(res2["metrics"]["rmse"])
    assert res1["forecast_strategy"] == res2["forecast_strategy"]

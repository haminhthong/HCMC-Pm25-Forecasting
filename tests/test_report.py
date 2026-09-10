from src.report import build_markdown


def test_report_contains_backtest_and_model_selection():
    evaluation = {
        "backtest": {"ridge": {"mae_mean": 1.2, "mae_std": 0.2, "rmse_mean": 1.5}},
        "model_selection": {
            "best_cv_model": "ridge",
            "forecast_strategy": "persistence",
            "status": "persistence",
        },
        "model_test": {"mae": 1.1, "rmse": 1.4},
        "persistence_test": {"mae": 1.0, "rmse": 1.2},
        "seasonal_naive_test": {"mae": 1.3, "rmse": 1.5},
        "selected_strategy_test": {"mae": 1.0, "rmse": 1.2},
    }
    report = build_markdown(evaluation)
    assert "ridge" in report
    assert "Forecast strategy: `persistence`" in report


def test_report_does_not_claim_production_gate():
    report = build_markdown({"backtest": {}})
    assert "Quality gate" not in report
    assert "production" not in report.lower()

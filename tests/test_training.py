from src.forecasting.selection import select_forecast_strategy


def test_model_selection_compares_against_persistence():
    config = {
        "model_selection": {
            "minimum_mae_improvement": 0.05,
            "maximum_cv_mae_std": 1.0,
        }
    }
    passed_metrics = {"mae": 1.0}
    persistence_metrics = {"mae": 2.0}

    passed = select_forecast_strategy("ridge", passed_metrics, persistence_metrics, 0.5, config)
    assert passed["forecast_strategy"] == "ridge"

    failed = select_forecast_strategy("ridge", {"mae": 1.98}, persistence_metrics, 0.5, config)
    assert failed["forecast_strategy"] == "persistence"

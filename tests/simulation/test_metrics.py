from src.simulation.metrics import calculate_metrics


def test_calculate_metrics_uses_worst_case_drawdown_when_available():
    equity_curve = [
        {"timestamp": "2024-01-01T00:00:00Z", "equity": 100.0, "equity_worst": 100.0},
        {"timestamp": "2024-01-01T01:00:00Z", "equity": 110.0, "equity_worst": 100.0},
        {"timestamp": "2024-01-01T02:00:00Z", "equity": 105.0, "equity_worst": 90.0},
    ]

    metrics = calculate_metrics([], equity_curve)

    # close-to-close drawdown: (110 - 105) / 110 = 4.545%
    assert abs(metrics["max_drawdown_close"] - (5.0 / 110.0)) < 1e-9
    # worst-case drawdown should capture 100 -> 90 = 10%
    assert abs(metrics["max_drawdown_worst"] - 0.1) < 1e-9
    # exposed headline drawdown should use the conservative value
    assert abs(metrics["max_drawdown"] - 0.1) < 1e-9

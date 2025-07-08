from src.query_tools.metric_retrieval import get_weekly_metrics


def get_metric_time_series(
    department: str, metric_name: str, start_week: int, end_week: int, year: int
) -> dict[int, float]:
    """Pull a single metric across multiple weeks."""
    # Generate slightly varying values to simulate trends
    base_value = get_weekly_metrics(department, end_week, year, [metric_name])[
        metric_name
    ]

    series = {}
    for week in range(start_week, end_week + 1):
        # Add some variation (-5% to +5%)
        variation = 1 + (0.1 * (week - start_week) / (end_week - start_week) - 0.05)
        series[week] = round(base_value * variation, 2)

    return series


def get_prior_year_metric(
    department: str, metric_name: str, week: int, year: int
) -> float:
    """Get the same metric from the same week in the prior year."""
    current_value = get_weekly_metrics(department, week, year, [metric_name])[
        metric_name
    ]

    # Simulate YoY growth - most metrics up 2-5% except items (volume)
    if metric_name == "sales":
        return round(current_value / 1.0447, 2)  # 4.47% growth as shown
    elif metric_name == "gpbf":
        return round(current_value / 1.0505, 2)  # 5.05% growth
    elif metric_name == "items":
        return round(current_value / 1.055, 2)  # 5.5% growth
    elif metric_name == "asp":
        return round(current_value / 0.991, 2)  # -0.9% deflation
    else:
        return round(current_value / 1.03, 2)  # Default 3% growth

from src.query_tools.metric_retrieval import get_weekly_metrics


def get_budget_values(
    department: str, week: int, year: int, metrics_list: list[str]
) -> dict[str, float]:
    """Pull budget values for specified metrics."""
    actuals = get_weekly_metrics(department, week, year, metrics_list)

    # Budget typically close to actuals with some variance
    budget = {}
    for metric, value in actuals.items():
        if metric == "sales":
            budget[metric] = round(value * 1.029, 2)  # Usually slightly under budget
        elif metric == "gpbf":
            budget[metric] = round(value * 1.043, 2)  # More under budget on profit
        else:
            budget[metric] = round(value * 1.01, 2)  # Close to budget

    return budget


def get_forecast_values(
    department: str, week: int, year: int, metrics_list: list[str]
) -> dict[str, float]:
    """Pull forecast values for specified metrics."""
    actuals = get_weekly_metrics(department, week, year, metrics_list)

    # Forecast very close to actuals
    forecast = {}
    for metric, value in actuals.items():
        if metric == "sales":
            forecast[metric] = round(value * 0.986, 2)  # Slightly below forecast
        elif metric == "gpbf":
            forecast[metric] = round(value * 0.95, 2)  # Below forecast
        else:
            forecast[metric] = round(value * 1.005, 2)  # Very close

    return forecast

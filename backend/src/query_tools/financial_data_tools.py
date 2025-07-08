from typing import Union, Any

from langchain.tools import tool

from src.query_tools import (
    # get_brand_type_metrics,
    get_budget_values,
    get_category_sales,
    get_claims_amount,
    get_event_impacts,
    get_forecast_values,
    get_metric_time_series,
    get_mix_variance,
    # get_mtd_sum,
    # get_price_changes,
    get_prior_year_metric,
    get_promo_metrics,
    get_promo_sales,
    get_rsa_amount,
    get_trade_variance,
    get_weekly_metrics,
    get_weekly_metrics_by_channel,
    # get_ytd_sum,
)


@tool
def fetch_weekly_metrics(
    department: str, week: int, year: int, metrics: str
) -> dict[str, float]:
    """Fetch specified metrics for a department for a single week.

    Args:
        department: Department name (e.g., "Everyday Chilled", "Bakery")
        week: Week number (1-52)
        year: Year (e.g., 2024)
        metrics: Comma-separated list of metrics (e.g., "sales,gpbf,items,asp")

    Returns:
        dictionary of metric names to values
    """
    metrics_list = [m.strip() for m in metrics.split(",")]
    return get_weekly_metrics(department, week, year, metrics_list)


@tool
def fetch_channel_metrics(
    department: str, week: int, year: int, channel: str, metrics: str
) -> dict[str, float]:
    """Fetch metrics filtered by sales channel (B&M or eCom).

    Args:
        department: Department name
        week: Week number
        year: Year
        channel: Either "B&M" or "eCom"
        metrics: Comma-separated list of metrics

    Returns:
        dictionary of metric names to values for the specified channel
    """
    metrics_list = [m.strip() for m in metrics.split(",")]
    return get_weekly_metrics_by_channel(department, week, year, channel, metrics_list)


@tool
def fetch_time_series(
    department: str, metric: str, start_week: int, end_week: int, year: int
) -> dict[int, float]:
    """Fetch a time series of a single metric across multiple weeks.

    Args:
        department: Department name
        metric: Single metric name (e.g., "sales")
        start_week: Starting week number
        end_week: Ending week number (inclusive)
        year: Year

    Returns:
        dictionary mapping week numbers to metric values
    """
    return get_metric_time_series(department, metric, start_week, end_week, year)


@tool
def fetch_variance_data(
    department: str, week: int, year: int, metric: str
) -> dict[str, float]:
    """Fetch variance data for a metric comparing actual vs LY, Budget, and Forecast.

    Args:
        department: Department name
        week: Week number
        year: Year
        metric: Metric name to get variances for

    Returns:
        dictionary with actual, prior_year, budget, and forecast values
    """
    metrics_list = [metric]
    actual = get_weekly_metrics(department, week, year, metrics_list)[metric]
    prior_year = get_prior_year_metric(department, metric, week, year)
    budget = get_budget_values(department, week, year, metrics_list)[metric]
    forecast = get_forecast_values(department, week, year, metrics_list)[metric]

    return {
        "actual": actual,
        "prior_year": prior_year,
        "budget": budget,
        "forecast": forecast,
        "vs_ly": actual - prior_year,
        "vs_budget": actual - budget,
        "vs_forecast": actual - forecast,
    }


@tool
def fetch_promotional_analysis(department: str, week: int, year: int) -> dict[str, Any]:
    """Fetch comprehensive promotional metrics and sales breakdown.

    Args:
        department: Department name
        week: Week number
        year: Year

    Returns:
        dictionary containing promotional metrics and promo/non-promo sales split
    """
    promo_metrics = get_promo_metrics(department, week, year)
    promo_sales = get_promo_sales(department, week, year)

    return {"metrics": promo_metrics, "sales_breakdown": promo_sales}


@tool
def fetch_category_performance(
    department: str, week: int, year: int
) -> dict[str, float]:
    """Fetch sales performance by category within a department.

    Args:
        department: Department name
        week: Week number
        year: Year

    Returns:
        dictionary mapping category names to sales values
    """
    return get_category_sales(department, week, year)


@tool
def fetch_external_impacts(
    department: str, week: int, year: int
) -> list[dict[str, Union[str, float]]]:
    """Fetch any external events impacting performance (floods, closures, etc).

    Args:
        department: Department name
        week: Week number
        year: Year

    Returns:
        list of impact events with descriptions and dollar impacts
    """
    return get_event_impacts(department, week, year)


@tool
def fetch_variance_bridge_components(
    department: str, week: int, year: int, comparison: str
) -> dict[str, float]:
    """Fetch variance bridge components (trade, mix, claims, RSA) for analysis.

    Args:
        department: Department name
        week: Week number
        year: Year
        comparison: Type of comparison ("LY", "Budget", or "Forecast")

    Returns:
        dictionary with trade, mix, claims, and RSA variance components
    """
    return {
        "trade": get_trade_variance(department, week, year, comparison),
        "mix": get_mix_variance(department, week, year, comparison),
        "claims": get_claims_amount(department, week, year)
        if comparison == "LY"
        else 0,
        "rsa": get_rsa_amount(department, week, year) if comparison == "LY" else 0,
    }


# Create a list of all financial tools
financial_analysis_tools = [
    fetch_weekly_metrics,
    fetch_channel_metrics,
    fetch_time_series,
    fetch_variance_data,
    fetch_promotional_analysis,
    fetch_category_performance,
    fetch_external_impacts,
    fetch_variance_bridge_components,
]

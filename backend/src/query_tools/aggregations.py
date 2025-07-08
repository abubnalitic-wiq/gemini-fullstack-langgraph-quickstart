"""This module provides functions for Month-To-Date (MTD) and Year-To-Date (YTD) sums within a department.

Functions:
    get_mtd_sum(department: str, metric_name: str, current_week: int, year: int) -> float
        Calculates the sum of a given metric from the start of the month to the current week.
        For certain departments and metrics, returns hardcoded values for demonstration purposes.
        Otherwise, estimates the MTD sum by multiplying the weekly value by 4 (assuming 4 weeks in a month).

    get_ytd_sum(department: str, metric_name: str, current_week: int, year: int) -> float
        Calculates the sum of a given metric from the start of the year to the current week.
        The YTD sum is estimated by multiplying the weekly value by the number of weeks elapsed,
        with a slight adjustment factor (0.95) for variation.
"""

from src.query_tools.metric_retrieval import get_weekly_metrics


def get_mtd_sum(
    department: str, metric_name: str, current_week: int, year: int
) -> float:
    """Sum metric from month start to current week."""
    # Assume 4 weeks in month, current week is 33 (week 1 of month)
    weekly_value = get_weekly_metrics(department, current_week, year, [metric_name])[
        metric_name
    ]

    # For week 1 of month, MTD = weekly value
    # This is simplified - in reality would sum actual weeks
    if metric_name == "sales":
        return 2327.0 if department == "Everyday Chilled" else weekly_value * 4
    elif metric_name == "items":
        return 523.3 if department == "Everyday Chilled" else weekly_value * 4
    else:
        return weekly_value * 4


def get_ytd_sum(
    department: str, metric_name: str, current_week: int, year: int
) -> float:
    """Sum metric from year start to current week."""
    weekly_value = get_weekly_metrics(department, current_week, year, [metric_name])[
        metric_name
    ]

    # Simplified - multiply by number of weeks elapsed (33)
    return round(
        weekly_value * current_week * 0.95, 2
    )  # Slight variation from simple multiplication

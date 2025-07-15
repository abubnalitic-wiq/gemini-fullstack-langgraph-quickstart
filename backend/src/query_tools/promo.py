from src.query_tools.metric_retrieval import get_weekly_metrics


def get_promo_metrics(department: str, week: int, year: int) -> dict[str, float]:
    """Pull promo penetration, frequency, depth, and scan funding."""
    # Base metrics by department
    if department == "Everyday Chilled":
        base_metrics = {
            "promo_frequency_avg": 20.3,
            "promo_pen_percent": 17.2,
            "promo_gp_percent": 22.3,
            "average_promo_depth_percent": 26.3,
            "scan_funding_amount": 4.8,
            "scan_funding_rate_percent": 57.0,
        }
    elif department == "Bakery":
        base_metrics = {
            "promo_frequency_avg": 14.5,
            "promo_pen_percent": 12.8,
            "promo_gp_percent": 18.5,
            "average_promo_depth_percent": 22.0,
            "scan_funding_amount": 1.8,
            "scan_funding_rate_percent": 45.0,
        }
    else:  # Default/other departments
        base_metrics = {
            "promo_frequency_avg": 18.5,
            "promo_pen_percent": 15.0,
            "promo_gp_percent": 20.0,
            "average_promo_depth_percent": 25.0,
            "scan_funding_amount": 2.5,
            "scan_funding_rate_percent": 55.0,
        }

    # Apply week-specific adjustments
    adjusted_metrics = base_metrics.copy()

    if week == 51:  # Pre-Christmas week
        if department == "Everyday Chilled":
            # Heavy promotional activity for Christmas entertaining
            adjusted_metrics["promo_frequency_avg"] = 32.5
            adjusted_metrics["promo_pen_percent"] = 28.5
            adjusted_metrics["promo_gp_percent"] = (
                26.8  # Better GP% on premium promo items
            )
            adjusted_metrics["average_promo_depth_percent"] = (
                22.0  # Shallower discounts
            )
            adjusted_metrics["scan_funding_amount"] = 8.2  # Suppliers invest heavily
            adjusted_metrics["scan_funding_rate_percent"] = 68.0
        elif department == "Bakery":
            # Increased promo for Christmas baking
            adjusted_metrics["promo_frequency_avg"] = 24.0
            adjusted_metrics["promo_pen_percent"] = 22.5
            adjusted_metrics["promo_gp_percent"] = 21.0
            adjusted_metrics["average_promo_depth_percent"] = (
                20.0  # Less deep discounts
            )
            adjusted_metrics["scan_funding_amount"] = 3.5
            adjusted_metrics["scan_funding_rate_percent"] = 55.0
        else:  # Other departments
            adjusted_metrics["promo_frequency_avg"] = 28.0
            adjusted_metrics["promo_pen_percent"] = 24.0
            adjusted_metrics["promo_gp_percent"] = 23.0
            adjusted_metrics["average_promo_depth_percent"] = 23.0
            adjusted_metrics["scan_funding_amount"] = 4.5
            adjusted_metrics["scan_funding_rate_percent"] = 62.0

    elif week == 33:  # Mid-August, back-to-school period
        if department == "Everyday Chilled":
            # Value-focused promotions
            adjusted_metrics["promo_frequency_avg"] = 23.0
            adjusted_metrics["promo_pen_percent"] = 20.5
            adjusted_metrics["promo_gp_percent"] = 20.0  # Lower GP% on deeper discounts
            adjusted_metrics["average_promo_depth_percent"] = 30.0  # Deeper discounts
            adjusted_metrics["scan_funding_amount"] = 5.5
            adjusted_metrics["scan_funding_rate_percent"] = 60.0
        elif department == "Bakery":
            # Lunch box promotions
            adjusted_metrics["promo_frequency_avg"] = 18.0
            adjusted_metrics["promo_pen_percent"] = 16.0
            adjusted_metrics["promo_gp_percent"] = 17.0
            adjusted_metrics["average_promo_depth_percent"] = 25.0
            adjusted_metrics["scan_funding_amount"] = 2.2
            adjusted_metrics["scan_funding_rate_percent"] = 48.0
        else:  # Other departments
            adjusted_metrics["promo_frequency_avg"] = 21.0
            adjusted_metrics["promo_pen_percent"] = 18.0
            adjusted_metrics["promo_gp_percent"] = 18.5
            adjusted_metrics["average_promo_depth_percent"] = 28.0
            adjusted_metrics["scan_funding_amount"] = 3.0
            adjusted_metrics["scan_funding_rate_percent"] = 58.0

    # Round all values to 1 decimal place for consistency
    return {metric: round(value, 1) for metric, value in adjusted_metrics.items()}


def get_promo_sales(department: str, week: int, year: int) -> dict[str, float]:
    """Pull sales split between promotional and non-promotional."""
    total_sales = get_weekly_metrics(department, week, year, ["sales"])["sales"]
    promo_pen = get_promo_metrics(department, week, year)["promo_pen_percent"]

    promo_sales = round(total_sales * (promo_pen / 100), 2)
    non_promo_sales = round(total_sales - promo_sales, 2)

    return {
        "promo_sales": promo_sales,
        "non_promo_sales": non_promo_sales,
        "total_sales": total_sales,
    }

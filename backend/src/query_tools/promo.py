from src.query_tools.metric_retrieval import get_weekly_metrics


def get_promo_metrics(department: str, week: int, year: int) -> dict[str, float]:
    """Pull promo penetration, frequency, depth, and scan funding."""
    if department == "Everyday Chilled":
        return {
            "promo_frequency_avg": 20.3,
            "promo_pen_percent": 17.2,
            "promo_gp_percent": 22.3,
            "average_promo_depth_percent": 26.3,
            "scan_funding_amount": 4.8,
            "scan_funding_rate_percent": 57.0,
        }
    else:
        return {
            "promo_frequency_avg": 18.5,
            "promo_pen_percent": 15.0,
            "promo_gp_percent": 20.0,
            "average_promo_depth_percent": 25.0,
            "scan_funding_amount": 2.5,
            "scan_funding_rate_percent": 55.0,
        }


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

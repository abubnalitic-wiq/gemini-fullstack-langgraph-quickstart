def get_weekly_metrics(
    department: str, week: int, year: int, metrics_list: list[str]
) -> dict[str, float]:
    """Pull specified metrics for a single week, scaled for a large Australian retailer."""
    # Canned responses based on department, scaled up for a large AU retailer
    if department == "Everyday Chilled":
        base_metrics = {
            "sales": 1583000.0,
            "gpbf": 622000.0,
            "items": 310000.0,
            "asp": 5.10,
            "stock_loss_rate": 0.8,
            "gpbf_percent": 39.3,
            "loyalty_discount_percent": 1.2,
            "own_brand_pen_percent": 32.0,
            "own_brand_gp_percent": 34.5,
            "promo_pen_percent": 24.0,
            "inflation": 1.1,
        }
    elif department == "Bakery":
        base_metrics = {
            "sales": 920000.0,
            "gpbf": 370000.0,
            "items": 210000.0,
            "asp": 4.38,
            "stock_loss_rate": 1.5,
            "gpbf_percent": 40.2,
            "loyalty_discount_percent": 1.0,
            "own_brand_pen_percent": 55.0,
            "own_brand_gp_percent": 60.0,
            "promo_pen_percent": 18.0,
            "inflation": 0.8,
        }
    else:  # Default/other departments
        base_metrics = {
            "sales": 600000.0,
            "gpbf": 180000.0,
            "items": 120000.0,
            "asp": 3.50,
            "stock_loss_rate": 0.9,
            "gpbf_percent": 30.0,
            "loyalty_discount_percent": 1.1,
            "own_brand_pen_percent": 25.0,
            "own_brand_gp_percent": 28.0,
            "promo_pen_percent": 20.0,
            "inflation": 1.0,
        }

    # Return only requested metrics
    return {metric: base_metrics.get(metric, 0.0) for metric in metrics_list}


def get_weekly_metrics_by_channel(
    department: str, week: int, year: int, channel: str, metrics_list: list[str]
) -> dict[str, float]:
    """Pull metrics filtered by channel (B&M or eCom)."""
    base_metrics = get_weekly_metrics(department, week, year, metrics_list)

    if channel == "B&M":
        # B&M typically 90% of sales in AU
        return {metric: value * 0.90 for metric, value in base_metrics.items()}
    elif channel == "eCom":
        # eCom typically 10% of sales in AU
        return {metric: value * 0.10 for metric, value in base_metrics.items()}
    else:
        return base_metrics

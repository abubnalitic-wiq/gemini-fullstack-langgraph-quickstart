def get_weekly_metrics(
    department: str, week: int, year: int, metrics_list: list[str]
) -> dict[str, float]:
    """Pull specified metrics for a single week, scaled for a large Australian retailer."""
    # Base metrics by department
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

    # Apply week-specific adjustments
    adjusted_metrics = base_metrics.copy()

    if week == 51:  # Pre-Christmas week
        if department == "Everyday Chilled":
            # Higher sales for entertaining, premium products
            adjusted_metrics["sales"] *= 1.45
            adjusted_metrics["gpbf"] *= 1.50  # Better margin on premium items
            adjusted_metrics["items"] *= 1.35
            adjusted_metrics["asp"] *= 1.07  # Higher ASP for premium items
            adjusted_metrics["promo_pen_percent"] = 32.0  # More promotions
            adjusted_metrics["stock_loss_rate"] = 1.2  # Higher loss due to volume
            adjusted_metrics["gpbf_percent"] = 40.8  # Better margin
        elif department == "Bakery":
            # Massive increase for Christmas baking
            adjusted_metrics["sales"] *= 1.85
            adjusted_metrics["gpbf"] *= 1.90
            adjusted_metrics["items"] *= 1.65
            adjusted_metrics["asp"] *= 1.12  # Premium Christmas items
            adjusted_metrics["promo_pen_percent"] = 28.0  # More promotions
            adjusted_metrics["stock_loss_rate"] = 2.2  # Higher wastage
            adjusted_metrics["gpbf_percent"] = 41.5
        else:  # Other departments
            adjusted_metrics["sales"] *= 1.30
            adjusted_metrics["gpbf"] *= 1.32
            adjusted_metrics["items"] *= 1.25
            adjusted_metrics["asp"] *= 1.04
            adjusted_metrics["promo_pen_percent"] = 28.0

    elif week == 33:  # Mid-August, back-to-school period
        if department == "Everyday Chilled":
            # Slight dip as families prepare for school
            adjusted_metrics["sales"] *= 0.92
            adjusted_metrics["gpbf"] *= 0.90
            adjusted_metrics["items"] *= 0.94
            adjusted_metrics["asp"] *= 0.98  # Value seeking
            adjusted_metrics["promo_pen_percent"] = 28.0  # More promo hunting
            adjusted_metrics["own_brand_pen_percent"] = 36.0  # More own brand
        elif department == "Bakery":
            # Increase for lunch box items
            adjusted_metrics["sales"] *= 1.15
            adjusted_metrics["gpbf"] *= 1.12
            adjusted_metrics["items"] *= 1.20
            adjusted_metrics["asp"] *= 0.96  # Smaller pack sizes
            adjusted_metrics["promo_pen_percent"] = 22.0
            adjusted_metrics["own_brand_pen_percent"] = 58.0
        else:  # Other departments
            adjusted_metrics["sales"] *= 0.95
            adjusted_metrics["gpbf"] *= 0.94
            adjusted_metrics["items"] *= 0.97
            adjusted_metrics["promo_pen_percent"] = 24.0

    # Recalculate derived metrics after adjustments
    if "sales" in adjusted_metrics and "gpbf" in adjusted_metrics:
        # Update GPBF% if both sales and GPBF are present
        adjusted_metrics["gpbf_percent"] = (
            adjusted_metrics["gpbf"] / adjusted_metrics["sales"]
        ) * 100

    if "sales" in adjusted_metrics and "items" in adjusted_metrics:
        # Update ASP if both sales and items are present
        adjusted_metrics["asp"] = adjusted_metrics["sales"] / adjusted_metrics["items"]

    # Return only requested metrics
    return {
        metric: round(adjusted_metrics.get(metric, 0.0), 2) for metric in metrics_list
    }


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

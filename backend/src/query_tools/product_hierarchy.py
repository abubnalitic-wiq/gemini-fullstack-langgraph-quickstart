from src.query_tools.metric_retrieval import get_weekly_metrics


def get_category_sales(department: str, week: int, year: int) -> dict[str, float]:
    """Pull sales by category within department."""
    total_sales = get_weekly_metrics(department, week, year, ["sales"])["sales"]

    if department == "Everyday Chilled":
        return {
            "Eggs": round(total_sales * 0.25, 2),
            "Yoghurt": round(total_sales * 0.20, 2),
            "Cheese": round(total_sales * 0.30, 2),
            "Plant Based": round(total_sales * 0.10, 2),
            "Other Chilled": round(total_sales * 0.15, 2),
        }
    elif department == "Bakery":
        return {
            "Bread": round(total_sales * 0.40, 2),
            "Rolls": round(total_sales * 0.15, 2),
            "Sweet Bakery": round(total_sales * 0.25, 2),
            "Cakes": round(total_sales * 0.20, 2),
        }
    else:
        return {
            "Category A": round(total_sales * 0.40, 2),
            "Category B": round(total_sales * 0.35, 2),
            "Category C": round(total_sales * 0.25, 2),
        }


def get_brand_type_metrics(
    department: str, week: int, year: int, brand_type: str, metrics_list: list[str]
) -> dict[str, float]:
    """Pull metrics filtered by own_brand or national_brand."""
    base_metrics = get_weekly_metrics(department, week, year, metrics_list)
    own_brand_pen = (
        get_weekly_metrics(department, week, year, ["own_brand_pen_percent"])[
            "own_brand_pen_percent"
        ]
        / 100
    )

    if brand_type == "own_brand":
        return {
            metric: round(value * own_brand_pen, 2)
            for metric, value in base_metrics.items()
        }
    elif brand_type == "national_brand":
        return {
            metric: round(value * (1 - own_brand_pen), 2)
            for metric, value in base_metrics.items()
        }
    else:
        return base_metrics

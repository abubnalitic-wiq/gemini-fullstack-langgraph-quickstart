from typing import Union


def get_event_impacts(
    department: str, week: int, year: int
) -> list[dict[str, Union[str, float]]]:
    """Pull any logged external impacts with dollar amounts."""
    if department == "Everyday Chilled" and week == 33:
        return [
            {
                "event": "QLD Floods",
                "impact_amount": -0.75,
                "impact_type": "sales",
                "description": "Store closures and supply chain disruption",
            },
            {
                "event": "IA DC Closure",
                "impact_amount": -0.8,
                "impact_type": "gpbf",
                "description": "Distribution center temporary closure",
            },
        ]
    else:
        return []


def get_price_changes(
    department: str, week: int, year: int
) -> list[dict[str, Union[str, float]]]:
    """Pull CPI (cost price increase) data."""
    if department == "Everyday Chilled":
        return [
            {
                "category": "Eggs",
                "cpi_amount": -150000,
                "cpi_percent": -8.5,
                "description": "Recent CPIs in Eggs",
            },
            {
                "category": "Yoghurt",
                "vendor": "WW Kids",
                "cpi_amount": -50000,
                "cpi_percent": -3.2,
                "description": "WW Kids Yoghurt price increase",
            },
        ]
    else:
        return []

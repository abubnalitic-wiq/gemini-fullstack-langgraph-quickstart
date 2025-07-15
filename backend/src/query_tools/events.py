from typing import Union


def get_event_impacts(
    department: str, week: int, year: int
) -> list[dict[str, Union[str, float]]]:
    """Pull any logged external impacts with dollar amounts."""
    events = []

    if week == 51:  # Pre-Christmas week
        if department == "Everyday Chilled":
            events = [
                {
                    "event": "NSW Transport Strike",
                    "impact_amount": -1.2,
                    "impact_type": "sales",
                    "description": "Delivery delays affecting fresh stock availability",
                },
                {
                    "event": "Heatwave VIC/SA",
                    "impact_amount": -0.5,
                    "impact_type": "gpbf",
                    "description": "Increased shrinkage and energy costs",
                },
                {
                    "event": "Port Congestion",
                    "impact_amount": -0.3,
                    "impact_type": "sales",
                    "description": "Import delays for premium Christmas lines",
                },
            ]
        elif department == "Bakery":
            events = [
                {
                    "event": "Flour Mill Fire",
                    "impact_amount": -0.8,
                    "impact_type": "gpbf",
                    "description": "Major supplier disruption requiring expensive alternatives",
                },
                {
                    "event": "Heatwave VIC/SA",
                    "impact_amount": -0.4,
                    "impact_type": "sales",
                    "description": "Reduced foot traffic and product quality issues",
                },
            ]
        else:  # Other departments
            events = [
                {
                    "event": "National Supply Chain Disruption",
                    "impact_amount": -0.6,
                    "impact_type": "sales",
                    "description": "General logistics challenges pre-Christmas",
                }
            ]

    elif week == 33:  # Back-to-school period
        if department == "Everyday Chilled":
            events = [
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
        elif department == "Bakery":
            events = [
                {
                    "event": "QLD Floods",
                    "impact_amount": -0.45,
                    "impact_type": "sales",
                    "description": "Reduced deliveries to North QLD stores",
                },
                {
                    "event": "Wheat Shortage",
                    "impact_amount": -0.3,
                    "impact_type": "gpbf",
                    "description": "Poor harvest affecting flour costs",
                },
            ]
        else:  # Other departments
            events = [
                {
                    "event": "Regional Store Closures",
                    "impact_amount": -0.4,
                    "impact_type": "sales",
                    "description": "Weather-related temporary closures",
                }
            ]

    # For other weeks, might have occasional events
    elif week % 13 == 0:  # Quarterly events
        if department == "Everyday Chilled":
            events = [
                {
                    "event": "Competitor Store Opening",
                    "impact_amount": -0.3,
                    "impact_type": "sales",
                    "description": "New ALDI in major trade area",
                }
            ]

    return events


def get_price_changes(
    department: str, week: int, year: int
) -> list[dict[str, Union[str, float]]]:
    """Pull CPI (cost price increase) data."""
    price_changes = []

    if week == 51:  # Pre-Christmas week
        if department == "Everyday Chilled":
            price_changes = [
                {
                    "category": "Premium Cheese",
                    "cpi_amount": -180000,
                    "cpi_percent": -12.0,
                    "description": "European import cost increases",
                },
                {
                    "category": "Smoked Salmon",
                    "vendor": "Ocean Blue",
                    "cpi_amount": -95000,
                    "cpi_percent": -15.0,
                    "description": "Peak season pricing for Christmas",
                },
                {
                    "category": "Champagne/Sparkling",
                    "cpi_amount": -120000,
                    "cpi_percent": -8.0,
                    "description": "Currency fluctuation and demand surge",
                },
            ]
        elif department == "Bakery":
            price_changes = [
                {
                    "category": "Premium Flour",
                    "cpi_amount": -85000,
                    "cpi_percent": -18.0,
                    "description": "Specialty flour for Christmas baking",
                },
                {
                    "category": "Butter",
                    "vendor": "Western Star",
                    "cpi_amount": -65000,
                    "cpi_percent": -22.0,
                    "description": "Dairy commodity price spike",
                },
                {
                    "category": "Dried Fruit & Nuts",
                    "cpi_amount": -45000,
                    "cpi_percent": -15.0,
                    "description": "Import costs for Christmas cakes",
                },
            ]
        else:  # Other departments
            price_changes = [
                {
                    "category": "General Grocery",
                    "cpi_amount": -50000,
                    "cpi_percent": -5.0,
                    "description": "Broad-based supplier increases",
                }
            ]

    elif week == 33:  # Back-to-school period
        if department == "Everyday Chilled":
            price_changes = [
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
        elif department == "Bakery":
            price_changes = [
                {
                    "category": "Sandwich Bread",
                    "cpi_amount": -40000,
                    "cpi_percent": -4.5,
                    "description": "Wheat price flow-through",
                },
                {
                    "category": "Lunch Box Snacks",
                    "vendor": "Uncle Tobys",
                    "cpi_amount": -25000,
                    "cpi_percent": -6.0,
                    "description": "Muesli bar range CPI",
                },
                {
                    "category": "Wraps & Rolls",
                    "cpi_amount": -18000,
                    "cpi_percent": -3.8,
                    "description": "Input cost increases",
                },
            ]
        else:  # Other departments
            price_changes = [
                {
                    "category": "General Packaged",
                    "cpi_amount": -30000,
                    "cpi_percent": -3.0,
                    "description": "Standard quarterly price review",
                }
            ]

    # Regular weeks might have standard CPIs
    elif week % 4 == 0:  # Monthly price reviews
        if department == "Everyday Chilled":
            price_changes = [
                {
                    "category": "Milk",
                    "cpi_amount": -25000,
                    "cpi_percent": -2.0,
                    "description": "Farmgate price adjustment",
                }
            ]
        elif department == "Bakery":
            price_changes = [
                {
                    "category": "Standard Bread",
                    "cpi_amount": -15000,
                    "cpi_percent": -1.8,
                    "description": "Regular price adjustment",
                }
            ]

    return price_changes
